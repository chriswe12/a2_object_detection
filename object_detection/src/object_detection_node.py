#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
from sensor_msgs_py import point_cloud2
import time
import cv2
from os.path import join
from collections import deque
from numpy.lib.recfunctions import unstructured_to_structured
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, PointCloud2, CameraInfo, PointField
from geometry_msgs.msg import PoseArray, Pose, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener, TransformException
from scipy.spatial.transform import Rotation

from object_detection_msgs.msg import (
    PointCloudArray,
    ObjectDetectionInfo,
    ObjectDetectionInfoArray,
)
from foxglove_msgs.msg import (
    ImageAnnotations,
    PointsAnnotation,
    TextAnnotation,
    Point2,
    Color,
)
from std_msgs.msg import Header

from object_detection.objectdetectorONNX import ObjectDetectorONNX
from object_detection.pointprojector import PointProjector
from object_detection.objectlocalizer import ObjectLocalizer
from object_detection.detection_logger import DetectionLogger
from object_detection.utils import *

# from object_detection.ros_numpy import *

from ament_index_python.packages import get_package_share_directory


class GlobalObject:
    """A fused best-guess of one physical object's location in the map frame.

    Repeated detections of the same class that land within the merge distance
    are averaged into a single running estimate, so the published object map
    stays stable as the robot re-observes things from different viewpoints.
    """

    def __init__(self, obj_id, name, position, confidence):
        self.id = obj_id
        self.name = name
        self.position = position  # np.array([x, y, z]) in the map frame
        self.count = 1            # number of fused observations
        self.confidence = confidence

    def update(self, position, confidence):
        self.count += 1
        # Incremental mean — equal weight per observation.
        self.position = self.position + (position - self.position) / self.count
        self.confidence = max(self.confidence, confidence)


class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__("object_detection_node")

        self.get_logger().info(
            "[ObjectDetection Node] Object Detector initilization starts ..."
        )

        # ---------- Initialize parameters ----------
        self.declare_parameters(
            namespace="",
            parameters=[
                ("verbose", True),
                ("project_object_points_to_image", True),
                ("project_all_points_to_image", False),
                ("camera_topic", "/rgb_camera/undistorted"),
                ("camera_info_topic", "/rgb_camera/camera_info"),
                ("lidar_topic", "/rslidar/points"),
                ("object_detection_pose_topic", "object_poses"),
                ("object_detection_output_image_topic", "detections_in_image"),
                ("object_detection_point_clouds_topic", "detection_point_clouds"),
                ("object_detection_info_topic", "detection_info"),
                ("image_annotations_topic", "detection_annotations"),
                ("camera_lidar_sync_queue_size", 10),
                # Max stamp difference (s) for the synchronizer to pair an image
                # with a lidar scan. With a 5 Hz camera and 10 Hz lidar there is
                # always a scan within ~0.05 s of each image, so 0.05 is the
                # practical floor: lower starts dropping images that have no
                # close-enough scan. The residual gap is removed by the
                # motion-compensated lidar->camera lookup (see sync_callback).
                ("camera_lidar_sync_slop", 0.05),
                ("architecture", "yolo"),
                ("task", ""),  # ""/"auto" -> infer from ONNX; "detect"/"segment" forces it
                ("model", "yolov5n6"),
                ("model_dir_path", ""),
                ("device", "cpu"),
                ("confident", 0.4),
                ("iou", 0.1),
                ("model_method", "hdbscan"),
                ("ground_percentage", 25),
                ("bb_contract_percentage", 10),
                ("distance_estimator_type", "none"),
                ("distance_estimator_save_data", False),
                ("object_specific_file", "object_specific.yaml"),
                ("min_cluster_size", 5),
                ("cluster_selection_epsilon", 0.08),
                ("max_object_depth", 0.25),
                ("classes", [11, 24, 25, 39, 74]),
                ("log_detections", True),
                ("log_file_path", ""),
                # --- global "best guess" object map (odometry-anchored) ---
                ("publish_global_objects", True),
                ("map_frame", "map"),
                # Fixed (non-moving) frame the lidar->camera transform is routed
                # through so ego-motion between the lidar scan time and the image
                # exposure time is removed. Empty -> fall back to map_frame. Under
                # RESPLE this is "map" (static-identity to "world").
                ("lidar_camera_fixed_frame", ""),
                # Map-frame fusion is deferred: each frame's detections are
                # queued and resolved against the odometry pose at their *exact*
                # stamp once it is available (tf2 interpolates between the two
                # surrounding poses), so nothing is dropped because the pose
                # hadn't arrived yet. This is the longest a frame waits in that
                # queue for its pose before we give up on its map position (the
                # detection is still logged). Keep it below the TF buffer cache.
                ("map_tf_max_wait", 2.0),
                # How often (s) the deferred-resolution timer runs.
                ("map_tf_poll_period", 0.05),
                ("object_merge_distance", 1.0),
                ("global_object_markers_topic", "global_object_markers"),
                ("global_object_info_topic", "global_detection_info"),
            ],
        )

        all_coco_ids = self.get_parameter("classes").value

        # ---------- Setup publishers ----------
        self.object_pose_pub = self.create_publisher(
            PoseArray, self.get_parameter("object_detection_pose_topic").value, 10
        )

        self.object_detection_img_pub = self.create_publisher(
            Image, self.get_parameter("object_detection_output_image_topic").value, 10
        )

        self.object_point_clouds_pub = self.create_publisher(
            PointCloudArray,
            self.get_parameter("object_detection_point_clouds_topic").value,
            10,
        )

        self.detection_info_pub = self.create_publisher(
            ObjectDetectionInfoArray,
            self.get_parameter("object_detection_info_topic").value,
            10,
        )

        # Foxglove image annotations (boxes + projected points + text labels),
        # overlaid on the raw camera image in Foxglove's Image panel.
        self.image_annotations_pub = self.create_publisher(
            ImageAnnotations,
            self.get_parameter("image_annotations_topic").value,
            10,
        )

        self.marker_pub = self.create_publisher(MarkerArray, "object_markers", 10)

        # Persistent best-guess object map, anchored in the odometry/map frame.
        self.map_frame = self.get_parameter("map_frame").value
        # Fixed frame for motion-compensating the lidar->camera transform.
        self.fixed_frame = (
            self.get_parameter("lidar_camera_fixed_frame").value or self.map_frame
        )
        self.global_marker_pub = self.create_publisher(
            MarkerArray, self.get_parameter("global_object_markers_topic").value, 10
        )
        self.global_info_pub = self.create_publisher(
            ObjectDetectionInfoArray,
            self.get_parameter("global_object_info_topic").value,
            10,
        )
        self.global_objects = []
        self._global_id_counter = 0

        # ---------- Setup subscribers ----------
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        camera_topic = self.get_parameter("camera_topic").value
        self.camera_sub = Subscriber(self, Image, camera_topic)
        self.lidar_sub = Subscriber(
            self,
            PointCloud2,
            self.get_parameter("lidar_topic").value,
            qos_profile=qos_profile,
        )

        # ---------- Setup synchronizer ----------
        self.synchronizer = ApproximateTimeSynchronizer(
            [self.camera_sub, self.lidar_sub],
            queue_size=self.get_parameter("camera_lidar_sync_queue_size").value,
            slop=self.get_parameter("camera_lidar_sync_slop").value,
        )
        self.synchronizer.registerCallback(self.sync_callback)

        # ---------- Config Directory ----------
        self.config_dir = join(get_package_share_directory("object_detection"), "cfg")

        # ---------- Setup TF ----------
        # spin_thread=True gives the listener its own executor thread so the TF
        # buffer keeps filling even while the main thread is busy with YOLO
        # inference, so the deferred-resolution timer always sees fresh poses.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer, self, spin_thread=True
        )

        # ---------- Deferred map-frame fusion ----------
        # Detections wait here (in arrival order) until the odometry pose at
        # their stamp is available; a timer drains them. This decouples the
        # map-frame position from when the pose arrives, so no frame is dropped
        # for being ahead of the SLAM estimate, and every frame gets the exact
        # interpolated pose rather than whatever happened to be latest.
        self._pending = deque()
        self.map_tf_max_wait = Duration(
            seconds=float(self.get_parameter("map_tf_max_wait").value)
        )
        self.create_timer(
            float(self.get_parameter("map_tf_poll_period").value),
            self._drain_pending,
        )

        # ---------- Setup PointProjector ----------
        self.point_projector = PointProjector(self)

        # ---------- Setup 2D Object Detection ----------
        self.object_detector = ObjectDetectorONNX(
            {
                "architecture": self.get_parameter("architecture").value,
                "task": self.get_parameter("task").value,
                "model": self.get_parameter("model").value,
                "model_dir_path": self.get_parameter("model_dir_path").value,
                "device": self.get_parameter("device").value,
                "confident": self.get_parameter("confident").value,
                "iou": self.get_parameter("iou").value,
                "checkpoint": None,
                "classes": all_coco_ids,
                "multiple_instance": False,
            },
        )

        # ---------- Setup 3D Object Localizer ----------
        self.object_localizer = ObjectLocalizer(
            self,
            {
                "model_method": self.get_parameter("model_method").value,
                "ground_percentage": self.get_parameter("ground_percentage").value,
                "bb_contract_percentage": self.get_parameter(
                    "bb_contract_percentage"
                ).value,
                "distance_estimator_type": self.get_parameter(
                    "distance_estimator_type"
                ).value,
                "distance_estimator_save_data": self.get_parameter(
                    "distance_estimator_save_data"
                ).value,
                "object_specific_file": self.get_parameter(
                    "object_specific_file"
                ).value,
                "min_cluster_size": self.get_parameter("min_cluster_size").value,
                "cluster_selection_epsilon": self.get_parameter(
                    "cluster_selection_epsilon"
                ).value,
                "max_object_depth": self.get_parameter("max_object_depth").value,
            },
            self.config_dir,
        )

        # ---------- Initialize components ----------
        self.image_reader = CvBridge()
        self.image_info_received = False
        self._K = None
        self._dist_coeffs = None
        self._det_logger = None

        self.get_logger().info(
            "[ObjectDetection Node] Object Detector initilization done."
        )
        self.get_logger().info("[ObjectDetection Node] Waiting for image info ...")
        self.get_logger().info(
            "[ObjectDetection Node] If this takes longer than a few seconds, make sure {self.camera_info_topic} is published."
        )

        # ---------- Check camera info ----------
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.get_parameter("camera_info_topic").value,
            self.image_info_callback,
            10,
        )

    def image_info_callback(self, msg):
        """Handle camera info message"""
        h = msg.height
        w = msg.width
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

        if check_validity_image_info(K, w, h):
            self._K = K
            self._dist_coeffs = np.array(msg.d, dtype=np.float64) if msg.d else None
            self.point_projector.set_intrinsic_params(K, [w, h], self._dist_coeffs)
            self.object_localizer.set_intrinsic_camera_param(K)
            self.optical_frame_id = msg.header.frame_id
            self.get_logger().info(
                "[ObjectDetection Node] Image info is set! Detection will start...",
                once=True,
            )
            self.image_info_received = True

            if self.get_parameter("log_detections").value and self._det_logger is None:
                log_path = self.get_parameter("log_file_path").value
                self._det_logger = DetectionLogger(log_path)
                self._det_logger.log_metadata(K, w, h, msg.header.frame_id)
                self.get_logger().info(
                    f"[ObjectDetection] Logging detections to {self._det_logger.path}, "
                    f"global objects to {self._det_logger.global_path}"
                )
        else:
            self.get_logger().error(
                " ------------------ camera_info not valid ------------------------"
            )

    def sync_callback(self, image_msg, lidar_msg):
        """Synchronized callback for image and point cloud"""
        if not self.image_info_received:
            self.get_logger().warn("Waiting for camera info...", once=True)
            return

        start_time = time.time()
        self.get_logger().info(
            "Got first image / pointcloud pair",
            once=True,
        )

        # If Image and Lidar messages are not empty
        if not image_msg.height > 0:
            self.get_logger().fatal(
                "[ObjectDetection Node] Image message is empty. Object detecion is on hold."
            )
            return
        if not lidar_msg.width > 0:
            self.get_logger().fatal(
                "[ObjectDetection Node] Lidar message is empty. Object detecion is on hold."
            )
            return

        try:
            # Read message — keep raw distorted image so YOLO bounding boxes are
            # in distorted pixel space. LiDAR projection applies the same distortion
            # model (via PointProjector), so the two align without any information loss.
            cv_image = self.image_reader.imgmsg_to_cv2(image_msg, "bgr8")

            point_cloud_xyz = pointcloud2_to_xyz_array(lidar_msg)
            # Strip NaN points (some lidars encode invalid returns as NaN)
            point_cloud_xyz = point_cloud_xyz[~np.isnan(point_cloud_xyz).any(axis=1)]
            # Validate point cloud data
            if point_cloud_xyz is None or point_cloud_xyz.shape[0] == 0:
                self.get_logger().warn("Empty point cloud received")
                return

            # TODO: ground filter is disabled — it runs before the TF transform so the
            # upward axis assumption (Z) is wrong for front_lidar_link (upward is X there).
            # If re-enabled, apply after the transform and use upward=-1 (camera optical -Y).
            # point_cloud_xyz = filter_ground(
            #     point_cloud_xyz, self.get_parameter("ground_percentage").value
            # )

            # Get the lidar->camera transform. The lidar scan (lidar_msg.stamp)
            # and the image (image_msg.stamp) are captured up to `slop` apart, so
            # we route the lookup through a fixed frame to remove the robot's
            # motion in that interval (the extrinsic is static, but the world has
            # moved relative to the rig). Returns (None, None) if it can't be
            # resolved even with the static fallback.
            rotation_matrix, translation = self._lidar_to_camera_transform(
                lidar_msg.header.frame_id,
                lidar_msg.header.stamp,
                image_msg.header.stamp,
            )
            if rotation_matrix is None:
                return

            # Transform points
            transformed_points = np.dot(point_cloud_xyz[:, :3], rotation_matrix.T) + translation
            point_cloud_xyz[:, :3] = transformed_points

            # Project points and validate results
            points_on_image, in_fov_indices = (
                self.point_projector.project_points_on_image(point_cloud_xyz[:, :3])
            )
            if len(in_fov_indices) == 0:
                self.get_logger().debug("No points projected within image frame")
                return

            # Get points in field of view
            pointcloud_in_fov = point_cloud_xyz[in_fov_indices]

            # Detect objects
            infer_start = time.time()
            object_detection_result, object_detection_image, object_masks = (
                self.object_detector.detect(cv_image)
            )
            infer_ms = (time.time() - infer_start) * 1000
            if (
                object_detection_result is None
                or len(object_detection_result.get("name", [])) == 0
            ):
                self.get_logger().debug("No objects detected")
                total_ms = (time.time() - start_time) * 1000
                fps = 1000.0 / total_ms if total_ms > 0 else 0.0
                self.get_logger().info(
                    f"0 obj | infer {infer_ms:.0f}ms | total {total_ms:.0f}ms | {fps:.1f} FPS",
                    throttle_duration_sec=1.0,
                )
                return

            # Localize objects
            object_list = self.object_localizer.localize(
                object_detection_result,
                points_on_image,
                point_cloud_xyz[in_fov_indices],
                cv_image,
                masks=object_masks,
            )
            # Create and publish results
            header = Header()
            header.stamp = image_msg.header.stamp
            header.frame_id = self.optical_frame_id

            # Map-frame data (logging + global fusion) is resolved later by
            # _drain_pending, once the odometry pose at this stamp is available.
            # We collect just what that step needs per detection here.
            pending_items = []

            object_pose_array = PoseArray(header=header)
            object_info_array = ObjectDetectionInfoArray(header=header)
            point_cloud_array = PointCloudArray(header=header)
            # Foxglove image annotations: one LINE_LOOP box + text per detection,
            # plus a single POINTS annotation accumulating projected lidar points.
            image_annotations = ImageAnnotations()
            points_annotation = PointsAnnotation()
            points_annotation.timestamp = header.stamp
            points_annotation.type = PointsAnnotation.POINTS
            points_annotation.thickness = 3.0

            # Build marker array — clear stale markers first, then add one
            # sphere + one text label per detected object.
            marker_array = MarkerArray()
            clear = Marker()
            clear.header.frame_id = self.optical_frame_id
            clear.header.stamp = image_msg.header.stamp
            clear.action = Marker.DELETEALL
            marker_array.markers.append(clear)

            # Populate messages
            for i, obj in enumerate(object_list):
                # Create pose
                object_pose = Pose()
                object_pose.position.x = float(obj.pos[0])
                object_pose.position.y = float(obj.pos[1])
                object_pose.position.z = float(obj.pos[2])
                object_pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                object_pose_array.poses.append(object_pose)

                # Create detection info
                object_information = ObjectDetectionInfo()
                object_information.class_id = str(
                    object_detection_result["name"][i]
                )  # Ensure string
                object_information.id = int(obj.id)  # Ensure integer

                object_information.position.x = float(obj.pos[0])
                object_information.position.y = float(obj.pos[1])
                object_information.position.z = float(obj.pos[2])
                object_information.pose_estimation_type = str(obj.estimation_type)
                object_information.confidence = float(
                    object_detection_result["confidence"][i]
                )
                object_information.bounding_box_min_x = int(
                    object_detection_result["xmin"][i]
                )
                object_information.bounding_box_min_y = int(
                    object_detection_result["ymin"][i]
                )
                object_information.bounding_box_max_x = int(
                    object_detection_result["xmax"][i]
                )
                object_information.bounding_box_max_y = int(
                    object_detection_result["ymax"][i]
                )
                object_info_array.info.append(object_information)

                # Bounding-box pixel coords + label, used by the Foxglove annotation.
                xmin = int(object_detection_result["xmin"][i])
                ymin = int(object_detection_result["ymin"][i])
                xmax = int(object_detection_result["xmax"][i])
                ymax = int(object_detection_result["ymax"][i])
                cls = str(object_detection_result["name"][i])
                score = float(object_detection_result["confidence"][i])

                # Defer the map-frame position (log + global fusion) until the
                # odometry pose at this stamp is available. Keep the minimal data
                # the deferred step needs.
                pending_items.append({
                    "name": cls,
                    "confidence": score,
                    "point_cam": np.array(
                        [float(obj.pos[0]), float(obj.pos[1]), float(obj.pos[2])]
                    ),
                    "bbox": [xmin, ymin, xmax, ymax],
                    "estimation_type": str(obj.estimation_type),
                })

                # Foxglove annotation: bounding box (LINE_LOOP) + label text
                box_ann = PointsAnnotation()
                box_ann.timestamp = header.stamp
                box_ann.type = PointsAnnotation.LINE_LOOP
                box_ann.thickness = 2.0
                box_ann.outline_color = Color(r=0.0, g=1.0, b=0.0, a=1.0)
                box_ann.points = [
                    Point2(x=float(xmin), y=float(ymin)),
                    Point2(x=float(xmax), y=float(ymin)),
                    Point2(x=float(xmax), y=float(ymax)),
                    Point2(x=float(xmin), y=float(ymax)),
                ]
                image_annotations.points.append(box_ann)

                text_ann = TextAnnotation()
                text_ann.timestamp = header.stamp
                text_ann.position = Point2(x=float(xmin), y=float(max(ymin - 4, 0)))
                text_ann.text = f"{cls} {score:.2f}"
                text_ann.font_size = 14.0
                text_ann.text_color = Color(r=1.0, g=1.0, b=1.0, a=1.0)
                text_ann.background_color = Color(r=0.0, g=1.0, b=0.0, a=0.5)
                image_annotations.texts.append(text_ann)
                # Create point cloud
                object_point_cloud = pointcloud_in_fov[obj.pt_indices]
                point_cloud_msg = array_to_pointcloud2(
                    object_point_cloud,
                    frame_id=self.optical_frame_id,
                    stamp=image_msg.header.stamp,
                )
                point_cloud_array.point_clouds.append(point_cloud_msg)

                # --- RViz markers ---
                class_name = str(object_detection_result["name"][i])
                c255 = CLASS_COLOR.get(class_name, (255, 255, 0))
                color = (c255[0] / 255.0, c255[1] / 255.0, c255[2] / 255.0)

                sphere = marker_(
                    ns="objects",
                    marker_id=i * 2,
                    pos=[float(obj.pos[0]), float(obj.pos[1]), float(obj.pos[2])],
                    stamp=image_msg.header.stamp,
                    color=color,
                    frame_id=self.optical_frame_id,
                )
                marker_array.markers.append(sphere)

                label = Marker()
                label.header.frame_id = self.optical_frame_id
                label.header.stamp = image_msg.header.stamp
                label.ns = "object_labels"
                label.id = i * 2 + 1
                label.type = Marker.TEXT_VIEW_FACING
                label.action = Marker.ADD
                label.pose.position.x = float(obj.pos[0])
                label.pose.position.y = float(obj.pos[1]) - 0.15  # above sphere
                label.pose.position.z = float(obj.pos[2])
                label.pose.orientation.w = 1.0
                label.scale.z = 0.15
                label.color.r = 1.0
                label.color.g = 1.0
                label.color.b = 1.0
                label.color.a = 1.0
                label.text = class_name
                marker_array.markers.append(label)

                # Visualize if enabled
                if (
                    not self.get_parameter("project_all_points_to_image").value
                    and self.get_parameter("project_object_points_to_image").value
                ):
                    object_points = points_on_image[obj.pt_indices]

                    if len(object_points.shape) == 1:
                        # Calculate number of points (total length must be even)
                        n_points = len(object_points) // 2
                        # Reshape to (n_points, 2) array
                        object_points = object_points.reshape(n_points, 2)
                    elif object_points.shape[1] != 2:
                        # If 2D but wrong shape, try to fix it
                        object_points = object_points.reshape(-1, 2)

                    for idx, pt in enumerate(object_points):
                        try:
                            dist = object_point_cloud[idx, 2]
                            color = depth_color(dist, min_d=0.5, max_d=20)
                            points_annotation.points.append(
                                Point2(x=float(pt[0]), y=float(pt[1]))
                            )
                            points_annotation.outline_colors.append(
                                Color(r=color[0] / 255.0, g=color[1] / 255.0,
                                      b=color[2] / 255.0, a=1.0)
                            )
                            # Make a copy of the image before drawing
                            object_detection_image = object_detection_image.copy()

                            cv2.circle(
                                object_detection_image,
                                pt[:2].astype(np.int32),
                                2,
                                color,
                                -1,
                            )
                        except Exception as e:
                            self.get_logger().warn(f"Could not draw circle: {str(e)}")

            # Publish all points if enabled
            if self.get_parameter("project_all_points_to_image").value:
                for idx, pt in enumerate(points_on_image):
                    dist = pointcloud_in_fov[idx, 2]
                    color = depth_color(dist, min_d=0.5, max_d=30)
                    points_annotation.points.append(
                        Point2(x=float(pt[0]), y=float(pt[1]))
                    )
                    points_annotation.outline_colors.append(
                        Color(r=color[0] / 255.0, g=color[1] / 255.0,
                              b=color[2] / 255.0, a=1.0)
                    )
                    try:
                        cv2.circle(
                            object_detection_image,
                            pt[:2].astype(np.int32),
                            3,
                            color,
                            -1,
                        )
                    except Exception as e:
                        self.get_logger().warn(f"Could not draw circle: {str(e)}")
            # Attach the accumulated projected lidar points to the annotation.
            if points_annotation.points:
                image_annotations.points.append(points_annotation)

            # Publish results
            self.marker_pub.publish(marker_array)
            self.object_pose_pub.publish(object_pose_array)
            self.detection_info_pub.publish(object_info_array)
            self.image_annotations_pub.publish(image_annotations)
            self.object_point_clouds_pub.publish(point_cloud_array)
            det_img_msg = self.image_reader.cv2_to_imgmsg(object_detection_image, "bgr8")
            # cv2_to_imgmsg leaves the header empty — stamp it with the input
            # image's frame_id + time so it matches camera_info / TF and the
            # detections (which share `header`). Fall back to the camera optical
            # frame if the source image header has no frame_id.
            det_img_msg.header = image_msg.header
            if not det_img_msg.header.frame_id:
                det_img_msg.header.frame_id = self.optical_frame_id
            self.object_detection_img_pub.publish(det_img_msg)

            # Queue this frame's detections for map-frame logging + fusion. The
            # timer resolves them against the odometry pose at header.stamp once
            # it is available, so they're never dropped for being ahead of it.
            if pending_items:
                self._pending.append(
                    {"stamp": header.stamp, "items": pending_items}
                )

            total_ms = (time.time() - start_time) * 1000
            fps = 1000.0 / total_ms if total_ms > 0 else 0.0
            self.get_logger().info(
                f"{len(object_list)} obj | infer {infer_ms:.0f}ms | total {total_ms:.0f}ms | {fps:.1f} FPS",
                throttle_duration_sec=1.0,
            )

        except Exception as e:
            self.get_logger().error(f"Error in sync_callback: {str(e)}")


    def _lidar_to_camera_transform(self, lidar_frame, lidar_stamp, image_stamp):
        """Return (R, t) mapping lidar points (at scan time) into the camera
        optical frame (at image time), compensated for ego-motion in between.

        The lidar->camera extrinsic is static, but the lidar scan and the image
        are captured at slightly different times; while the robot moves, the
        world shifts relative to the rig in that interval. Routing the lookup
        through ``self.fixed_frame`` (e.g. ``map``) at the two distinct stamps
        removes that shift, so projected points land on the object as seen in the
        image rather than where it was when the lidar swept it.

        Falls back to the time-agnostic static extrinsic if the motion-
        compensated lookup can't be resolved yet (e.g. odometry not yet covering
        both stamps), so frames are never dropped for a missing pose. Returns
        (None, None) only if even the static lookup fails.
        """
        t = None
        try:
            t = self.tf_buffer.lookup_transform_full(
                self.optical_frame_id,
                rclpy.time.Time.from_msg(image_stamp),
                lidar_frame,
                rclpy.time.Time.from_msg(lidar_stamp),
                self.fixed_frame,
            )
        except TransformException as ex:
            # Motion-compensated lookup unavailable: use the static extrinsic so
            # we still publish, but warn since projection may smear during motion.
            try:
                t = self.tf_buffer.lookup_transform(
                    self.optical_frame_id, lidar_frame, rclpy.time.Time()
                )
            except TransformException as ex2:
                self.get_logger().info(
                    f"Could not transform points from {lidar_frame} to "
                    f"{self.optical_frame_id}: {ex2}",
                    throttle_duration_sec=1.0,
                )
                return None, None
            self.get_logger().warn(
                f"Motion-compensated lidar->camera lookup via '{self.fixed_frame}' "
                f"unavailable ({ex}); using static extrinsic — projection may "
                "smear while the robot moves.",
                throttle_duration_sec=5.0,
            )

        translation = np.array([
            t.transform.translation.x,
            t.transform.translation.y,
            t.transform.translation.z,
        ])
        quaternion = [
            t.transform.rotation.x,
            t.transform.rotation.y,
            t.transform.rotation.z,
            t.transform.rotation.w,
        ]
        return Rotation.from_quat(quaternion).as_matrix(), translation

    def _map_transform_at(self, query):
        """Non-blocking optical->map transform (R, t) at the exact time `query`.

        Returns (None, None) if tf2 can't resolve that stamp yet (the surrounding
        odometry poses haven't both arrived). No 'latest pose' fallback: the
        caller waits for the real interpolated pose instead of using a
        temporally-misaligned one.
        """
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.optical_frame_id, query
            )
        except TransformException:
            return None, None
        tr = np.array([
            tf.transform.translation.x,
            tf.transform.translation.y,
            tf.transform.translation.z,
        ])
        q = [
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
            tf.transform.rotation.w,
        ]
        return Rotation.from_quat(q).as_matrix(), tr

    def _drain_pending(self):
        """Resolve queued detections against their stamp's odometry pose.

        Processes the queue in arrival (≈stamp) order. A frame whose pose isn't
        available yet but is still younger than ``map_tf_max_wait`` is left in
        place (we wait for the estimator to catch up) and, since the queue is
        ordered, so is everything behind it. A frame that ages past that without
        a pose is resolved anyway with no map position -- the detection is still
        logged, never dropped.
        """
        if not self._pending:
            return
        now = self.get_clock().now()
        while self._pending:
            rec = self._pending[0]
            query = rclpy.time.Time.from_msg(rec["stamp"])
            R, t = self._map_transform_at(query)
            if R is None:
                if (now - query) < self.map_tf_max_wait:
                    break  # pose may still arrive; keep order, try again later
                self.get_logger().warn(
                    f"No {self.map_frame} <- {self.optical_frame_id} pose within "
                    f"{self.map_tf_max_wait.nanoseconds / 1e9:.2f}s of the image "
                    "stamp; logging the detection without a map position.",
                    throttle_duration_sec=2.0,
                )
            self._pending.popleft()
            self._resolve_frame(rec, R, t)

    def _resolve_frame(self, rec, R, t):
        """Log + fuse one frame's detections using its resolved pose (R, t).

        R may be None (pose never arrived): detections are still logged, just
        without a map-frame position, and global fusion is skipped.
        """
        stamp = rec["stamp"]
        if self._det_logger is not None:
            for it in rec["items"]:
                point_map = (R @ it["point_cam"] + t) if R is not None else None
                self._det_logger.log_detection(
                    stamp=stamp,
                    class_id=it["name"],
                    confidence=it["confidence"],
                    point_cam=it["point_cam"],
                    bbox=it["bbox"],
                    camera_frame=self.optical_frame_id,
                    estimation_type=it["estimation_type"],
                    point_map=point_map,
                    map_frame=self.map_frame,
                )
        if R is not None:
            self.update_global_objects(rec["items"], stamp, R, t)

    def update_global_objects(self, items, stamp, R, t):
        """Transform localized detections into the map frame and fuse them in."""
        if not self.get_parameter("publish_global_objects").value:
            return

        merge_distance = self.get_parameter("object_merge_distance").value
        for it in items:
            # Only fuse real lidar measurements; bbox-only distance estimates and
            # un-localized detections (z == -1) carry no reliable 3D position.
            if it["estimation_type"] != "measurement":
                continue
            pos_map = R @ it["point_cam"] + t
            self._merge_global_object(
                it["name"], pos_map, it["confidence"], merge_distance
            )

        self._publish_global_objects(stamp)

    def _merge_global_object(self, name, position, confidence, merge_distance):
        """Associate a detection with an existing best-guess or start a new one."""
        match = None
        best_distance = merge_distance
        for g in self.global_objects:
            if g.name != name:
                continue
            distance = np.linalg.norm(g.position - position)
            if distance < best_distance:
                best_distance = distance
                match = g

        if match is None:
            self._global_id_counter += 1
            self.global_objects.append(
                GlobalObject(self._global_id_counter, name, position.copy(), confidence)
            )
        else:
            match.update(position, confidence)

    def _publish_global_objects(self, stamp):
        """Publish the full best-guess object map as markers + detection info."""
        marker_array = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self.map_frame
        clear.header.stamp = stamp
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        info_array = ObjectDetectionInfoArray()
        info_array.header.frame_id = self.map_frame
        info_array.header.stamp = stamp

        for k, g in enumerate(self.global_objects):
            c255 = CLASS_COLOR.get(g.name, (255, 255, 0))
            color = (c255[0] / 255.0, c255[1] / 255.0, c255[2] / 255.0)

            sphere = marker_(
                ns="global_objects",
                marker_id=k * 2,
                pos=[float(g.position[0]), float(g.position[1]), float(g.position[2])],
                stamp=stamp,
                color=color,
                frame_id=self.map_frame,
            )
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.3
            marker_array.markers.append(sphere)

            label = Marker()
            label.header.frame_id = self.map_frame
            label.header.stamp = stamp
            label.ns = "global_object_labels"
            label.id = k * 2 + 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = float(g.position[0])
            label.pose.position.y = float(g.position[1])
            label.pose.position.z = float(g.position[2]) + 0.3
            label.pose.orientation.w = 1.0
            label.scale.z = 0.25
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            label.text = f"{g.name} #{g.id} ({g.count})"
            marker_array.markers.append(label)

            info = ObjectDetectionInfo()
            info.class_id = g.name
            info.id = int(g.id)
            info.position.x = float(g.position[0])
            info.position.y = float(g.position[1])
            info.position.z = float(g.position[2])
            info.confidence = float(g.confidence)
            info.pose_estimation_type = "global"
            info_array.info.append(info)

        self.global_marker_pub.publish(marker_array)
        self.global_info_pub.publish(info_array)

        if self._det_logger is not None and self.global_objects:
            self._det_logger.write_global_objects(self.global_objects, stamp)

    def destroy_node(self):
        # Flush anything still queued: resolve against the pose if it's there by
        # now, otherwise log without a map position so no detection is lost.
        while self._pending:
            rec = self._pending.popleft()
            R, t = self._map_transform_at(rclpy.time.Time.from_msg(rec["stamp"]))
            self._resolve_frame(rec, R, t)
        if self._det_logger is not None:
            if self.global_objects:
                self._det_logger.write_global_objects(self.global_objects, stamp=None)
            self._det_logger.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    try:
        node = ObjectDetectionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down")
    except Exception as e:
        node.get_logger().fatal(f"Fatal error: {str(e)}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
