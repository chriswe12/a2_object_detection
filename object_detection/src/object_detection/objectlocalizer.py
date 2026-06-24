#!/usr/bin/env python3
import os
import yaml
import hdbscan
import numpy as np
from scipy.signal import find_peaks
from dataclasses import dataclass
from rclpy.logging import get_logger

NO_POSE = -1
MAX_DIST = 999999

AXIS_X = 0
AXIS_Y = 1
AXIS_Z = 2

DEFAULT_MAX_OBJECT_DEPTH = 0.25


@dataclass
class DetectedObject:
    """Class for keeping the info of objects"""

    id: int
    idx: int
    pos: np.array
    pt_indices: np.array
    estimation_type: str


class ObjectLocalizer:
    """Class for localizing objects in 3D space using 2D detections and point cloud data."""

    def __init__(self, node, config, config_dir):
        """Initialize the ObjectLocalizer.

        Args:
            node: ROS2 node instance
            config: Configuration dictionary containing model parameters
            config_dir: Directory containing configuration files
        """
        self.node = node
        self.logger = node.get_logger() if node else get_logger("object_localizer")

        try:
            self.object_specific_file_dir = os.path.join(
                config_dir, config["object_specific_file"]
            )
            with open(self.object_specific_file_dir) as file:
                self.obj_conf = yaml.safe_load(file)
            self.logger.info("Loaded object specific configuration")
        except Exception as e:
            self.logger.info(
                f"Object specific file not found: {str(e)}. This is fine if using version 1."
            )

        self.model_method = config["model_method"].lower()
        self.distance_estimator_type = config["distance_estimator_type"].lower()
        self.distance_estimator_save_data = config["distance_estimator_save_data"]
        self.bb_contract_percentage = config["bb_contract_percentage"]
        self.min_cluster_size = config["min_cluster_size"]
        self.cluster_selection_epsilon = config["cluster_selection_epsilon"]
        # Depth slice (m) kept around the chosen cluster's front; falls back to
        # the module default when not supplied.
        self.max_object_depth = config.get("max_object_depth", DEFAULT_MAX_OBJECT_DEPTH)

        self.distance_estimator = self.estimate_dist_default
        self.id_dict = {}
        self.masks = None

        if self.distance_estimator_type != "none":
            try:
                with open(
                    os.path.join(config_dir, f"{self.distance_estimator_type}.yaml")
                ) as file:
                    self.estimate_dist_cfg = yaml.safe_load(file)
                self.distance_estimator = getattr(
                    self, f"estimate_dist_{self.distance_estimator_type}"
                )
                self.logger.info(
                    f"Distance estimator {self.distance_estimator_type} has been set"
                )
            except Exception as e:
                self.logger.error(
                    f"Distance estimator {self.distance_estimator_type} is not defined: {str(e)}"
                )

            if self.distance_estimator_save_data:
                self.learner_data_dir = os.path.join(
                    config_dir, "data", self.distance_estimator_type
                )
                self.create_save_directory()
                self.data_saver = getattr(
                    self, f"save_data_{self.distance_estimator_type}"
                )
                self.distance_estimator = self.estimate_dist_default
                self.logger.info(
                    f"Data collection enabled for {self.distance_estimator_type}"
                )
        else:
            self.distance_estimator_save_data = False
            self.logger.info("No estimator will be used (estimator type is None)")

    def set_scene(self, objects, points2D, points3D, image=None, masks=None):
        """Set the scene info such as objects, points2D, points3D, image.

        Args:
            objects     : 2D object detection results in Panda Dataframe
            points2D    : 2D Point cloud in camera frame on the image
            points3D    : 3D Point cloud in camera frame
            masks       : Optional [N, H, W] boolean instance masks aligned with
                          ``objects`` rows. When given, points are associated to
                          an object by mask membership instead of its bounding box.
        """
        self.objects = objects
        self.points2D = points2D
        self.points3D = points3D
        self.image = image
        self.masks = masks

    def set_intrinsic_camera_param(self, K):
        """Set intrinsic camera parameters.

        Args:
            K     : intrinsic camera parameters
        """
        self.K = K

    def create_save_directory(self):
        """Create directory for saving learner data if it doesn't exist."""
        if not os.path.exists(self.learner_data_dir):
            os.makedirs(self.learner_data_dir)
            self.logger.debug(f"Created data directory: {self.learner_data_dir}")

    def save_data_bb2dist(self, input):
        """Save bounding box to distance data for training.

        Args:
            input: Tuple containing (index, pose) of the detected object
        """
        ind, pose = input[0], input[1]

        for i in range(len(self.objects)):
            if ind == i or self.is_overlapping(ind, i):
                continue

        obj_class = self.objects["name"][ind]
        bb_size = self.object_unique_size(ind, self.obj_conf[obj_class]["unique"])

        with open(os.path.join(self.learner_data_dir, f"{obj_class}.txt"), "a") as file:
            file.write(f"{bb_size} {pose[2]}\n")

    def object_unique_size(self, ind, unique):
        """Calculate the unique size of an object based on its bounding box.

        Args:
            ind: Index of the object
            unique: Dimension to use ('x', 'y', or other for diagonal)
        Returns:
            float: Size measurement in pixels
        """
        if unique == "x":
            return self.objects["xmax"][ind] - self.objects["xmin"][ind]
        elif unique == "y":
            return self.objects["ymax"][ind] - self.objects["ymin"][ind]
        else:
            min_p = np.array([self.objects["xmin"][ind], self.objects["ymin"][ind]])
            max_p = np.array([self.objects["xmax"][ind], self.objects["ymax"][ind]])
            return np.linalg.norm(max_p - min_p)

    def is_overlapping(self, ind1, ind2):
        """Check if two bounding boxes are overlapping.

        Args:
            ind1: Index of first bounding box
            ind2: Index of second bounding box
        Returns:
            bool: True if boxes overlap, False otherwise
        """
        return (
            (self.objects["xmax"][ind1] >= self.objects["xmin"][ind2])
            and (self.objects["xmax"][ind2] >= self.objects["xmin"][ind1])
            and (self.objects["ymax"][ind1] >= self.objects["ymin"][ind2])
            and (self.objects["ymax"][ind2] >= self.objects["ymin"][ind1])
        )

    def object_id(self, class_id):
        """Generate unique ID for object of given class.

        Args:
            class_id: Class identifier of the object
        Returns:
            int: Unique object ID
        """
        if class_id in self.id_dict:
            object_id = self.id_dict[class_id] + 1
            self.id_dict[class_id] += 1
        else:
            object_id = 0
            self.id_dict[class_id] = 0
        return object_id

    def estimate_pos_with_BB_center(self, center, est_dist):
        """Estimate 3D position using bounding box center and estimated distance.

        Args:
            center: 2D center point of bounding box
            est_dist: Estimated distance to object
        Returns:
            list: [X, Y, Z] coordinates in camera frame
        """
        X = (center[0] - self.K[0, 2]) * est_dist / self.K[0, 0]
        Y = (center[1] - self.K[1, 2]) * est_dist / self.K[1, 1]
        return [X, Y, est_dist]

    def estimate_dist_default(self, input):
        """Default distance estimation method.

        Args:
            input: Input data for estimation
        Returns:
            float: Default distance value (0)
        """
        return 0

    def estimate_dist_bb2dist(self, input):
        """Estimate distance using bounding box size relationship.

        Args:
            input: Tuple of (index, class_id)
        Returns:
            float: Estimated distance to object
        """
        idx, class_id = input[0], input[1]
        p = np.poly1d(self.estimate_dist_cfg[class_id])
        return max(
            p(self.object_unique_size(idx, self.obj_conf[class_id]["unique"])), 0.5
        )

    def points_in_BB(self, index):
        """Find points that lie within the contracted bounding box.

        Args:
            index: Index of the bounding box
        Returns:
            tuple: (points indices inside BB, center point index, BB center)
        """
        x_diff = self.objects["xmax"][index] - self.objects["xmin"][index]
        y_diff = self.objects["ymax"][index] - self.objects["ymin"][index]

        inside_BB_x = np.logical_and(
            (
                self.points2D[:, 0]
                >= self.objects["xmin"][index]
                + x_diff * self.bb_contract_percentage / 100
            ),
            (
                self.points2D[:, 0]
                <= self.objects["xmax"][index]
                - x_diff * self.bb_contract_percentage / 100
            ),
        )
        inside_BB_y = np.logical_and(
            (
                self.points2D[:, 1]
                >= self.objects["ymin"][index]
                + y_diff * self.bb_contract_percentage / 100
            ),
            (
                self.points2D[:, 1]
                <= self.objects["ymax"][index]
                - y_diff * self.bb_contract_percentage / 100
            ),
        )
        inside_BB = np.nonzero(np.logical_and(inside_BB_x, inside_BB_y))[0]

        center = np.array(
            [
                (self.objects["xmin"][index] + self.objects["xmax"][index]) / 2.0,
                (self.objects["ymin"][index] + self.objects["ymax"][index]) / 2.0,
            ]
        )

        if len(inside_BB) == 0:
            return np.array([NO_POSE]), NO_POSE, center
        else:
            center_ind = np.argmin(
                np.linalg.norm(self.points2D[inside_BB, :] - center, axis=1)
            )
            return inside_BB, center_ind, center

    def points_in_mask(self, index):
        """Find points whose image projection falls inside the instance mask.

        Args:
            index: Index of the object / mask
        Returns:
            tuple: (point indices inside mask, center point index, BB center)
        """
        mask = self.masks[index]
        h, w = mask.shape

        px = self.points2D[:, 0].astype(np.int64)
        py = self.points2D[:, 1].astype(np.int64)
        in_frame = (px >= 0) & (px < w) & (py >= 0) & (py < h)

        inside = np.zeros(len(px), dtype=bool)
        inside[in_frame] = mask[py[in_frame], px[in_frame]]
        inside_mask = np.nonzero(inside)[0]

        center = np.array(
            [
                (self.objects["xmin"][index] + self.objects["xmax"][index]) / 2.0,
                (self.objects["ymin"][index] + self.objects["ymax"][index]) / 2.0,
            ]
        )

        if len(inside_mask) == 0:
            return np.array([NO_POSE]), NO_POSE, center

        center_ind = np.argmin(
            np.linalg.norm(self.points2D[inside_mask, :] - center, axis=1)
        )
        return inside_mask, center_ind, center

    def method_hdbscan(self, in_BB_3D, obj_class, estimated_dist):
        """Apply HDBSCAN clustering to find object points.

        Args:
            in_BB_3D: 3D points within bounding box
            obj_class: Object class name
            estimated_dist: Estimated distance to object
        Returns:
            tuple: (average position, point indices)
        """
        cluster = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
        ).fit(in_BB_3D[:, [AXIS_X, AXIS_Z]])

        unique = np.unique(cluster.labels_)
        min_val = MAX_DIST
        indices = None

        for i in unique:
            if i == -1:
                continue

            indices_ = np.nonzero(cluster.labels_ == i)[0]
            min_val_ = np.abs(estimated_dist - min(in_BB_3D[indices_, AXIS_Z]))

            if min_val_ < min_val:
                indices = indices_
                min_val = min_val_

        if indices is None:
            indices_ = np.nonzero(cluster.labels_ == -1)[0]
            indices = np.argmin(np.abs(estimated_dist - in_BB_3D[indices_, AXIS_Z]))
            avg = in_BB_3D[indices]
        else:
            distances = np.squeeze(in_BB_3D[indices, AXIS_Z])
            in_range_indices = np.nonzero(
                (
                    np.abs(distances - estimated_dist)
                    - min(np.abs(distances - estimated_dist))
                )
                < self.max_object_depth
            )[0]
            indices = indices[in_range_indices]
            avg = np.mean(in_BB_3D[indices], axis=0)

        return avg, indices

    def method_histogram(self, in_BB_3D, method="distance", bins=100):
        """Use histogram analysis to find object points.

        Args:
            in_BB_3D: 3D points within bounding box
            method: Analysis method ('distance' or other)
            bins: Number of histogram bins
        Returns:
            tuple: (mean position, point indices)
        """
        if method == "distance":
            hist, bin_edges = np.histogram(np.linalg.norm(in_BB_3D, axis=1), bins)
        else:
            hist, bin_edges = np.histogram(in_BB_3D[:, 2], bins)

        bin_edges = bin_edges[:-1]
        hist = np.insert(hist, 0, 0)
        bin_edges = np.insert(bin_edges, 0, 0)
        peaks, _ = find_peaks(hist)

        inside_peak = np.logical_and(
            (in_BB_3D[:, 2] >= bin_edges[peaks[0] - 1]),
            (in_BB_3D[:, 2] <= bin_edges[peaks[0] + 1]),
        )

        return np.mean(in_BB_3D[inside_peak, :], axis=0), inside_peak

    def get_object_pos(self, index):
        """Get 3D position of object from its 2D detection.

        Args:
            index: Index of the object in detection results
        Returns:
            DetectedObject: Object with position and point information
        """
        obj_class = self.objects["name"][index]
        new_obj_id = self.object_id(obj_class)
        new_obj = DetectedObject(
            id=new_obj_id, idx=index, pos=None, pt_indices=None, estimation_type=None
        )

        # Prefer mask membership when a segmentation mask is available; fall back
        # to the bounding box if the mask captures no projected points.
        use_mask = (
            self.masks is not None
            and index < len(self.masks)
            and self.masks[index] is not None
        )
        if use_mask:
            in_BB_indices, center_ind, center = self.points_in_mask(index)
            if center_ind == NO_POSE:
                in_BB_indices, center_ind, center = self.points_in_BB(index)
                use_mask = False
        else:
            in_BB_indices, center_ind, center = self.points_in_BB(index)

        # Mask membership is trusted directly, so a single point is enough; the
        # bounding-box path still needs a cluster's worth of points to be robust.
        min_points = 1 if use_mask else self.min_cluster_size

        if center_ind == NO_POSE or len(in_BB_indices) < min_points:
            if self.distance_estimator_type == "none":
                new_obj.pt_indices = np.array([NO_POSE])
                new_obj.pos = np.array([0, 0, NO_POSE])
                new_obj.estimation_type = "none"
            else:
                estimated_dist = self.distance_estimator([index, obj_class])
                new_obj.pt_indices = np.array([NO_POSE])
                new_obj.pos = self.estimate_pos_with_BB_center(center, estimated_dist)
                new_obj.estimation_type = "estimation"
        else:
            in_BB_3D = self.points3D[in_BB_indices, :]

            if use_mask:
                # Segmentation already isolated the object in 2D, so take all
                # masked points and use their mean (no extra clustering).
                pos, on_object = np.mean(in_BB_3D, axis=0), np.arange(in_BB_3D.shape[0])
            elif self.model_method == "hdbscan":
                try:
                    estimated_dist = self.distance_estimator([index, obj_class])
                    pos, on_object = self.method_hdbscan(
                        in_BB_3D, obj_class, estimated_dist
                    )
                except Exception as e:
                    self.logger.warn(f"Estimation failed for {obj_class}: {str(e)}")
                    estimated_dist = 0
                    pos, on_object = np.mean(in_BB_3D, axis=0), np.arange(
                        in_BB_3D.shape[0]
                    )
            elif self.model_method == "mean":
                pos, on_object = np.mean(in_BB_3D, axis=0), np.arange(in_BB_3D.shape[0])
            elif self.model_method == "median":
                pos, on_object = np.median(in_BB_3D, axis=0), np.arange(
                    in_BB_3D.shape[0]
                )
            elif self.model_method == "center":
                pos, on_object = in_BB_3D[center_ind, :], np.arange(in_BB_3D.shape[0])
            elif self.model_method == "histogram":
                pos, on_object = self.method_histogram(in_BB_3D)
            else:
                self.logger.error(f"Unknown method: {self.model_method}")
                pos, on_object = np.mean(in_BB_3D, axis=0), np.arange(in_BB_3D.shape[0])

            # Align with bounding box center
            center_point = in_BB_3D[center_ind]
            center_point[[AXIS_X, AXIS_Y]] = center_point[[AXIS_X, AXIS_Y]] * (
                pos[AXIS_Z] / center_point[AXIS_Z]
            )
            pos[[AXIS_X, AXIS_Y]] = center_point[[AXIS_X, AXIS_Y]]

            new_obj.pt_indices = (
                np.array([in_BB_indices[on_object]])
                if isinstance(on_object, np.int64)
                else in_BB_indices[on_object]
            )
            new_obj.pos = pos
            new_obj.estimation_type = "measurement"

        return new_obj

    def localize(self, objects, points2D, points3D, image=None, masks=None):
        """Localize all detected objects in 3D space.

        Args:
            objects: 2D object detection results
            points2D: 2D point cloud in image frame
            points3D: 3D point cloud in camera frame
            image: Optional image data
            masks: Optional [N, H, W] boolean instance masks aligned with objects
        Returns:
            list: List of DetectedObject instances
        """
        self.set_scene(objects, points2D, points3D, image, masks)
        self.id_dict = {}
        object_list = []

        for ind in range(len(objects)):
            new_obj = self.get_object_pos(ind)
            object_list.append(new_obj)

            if self.distance_estimator_save_data:
                self.data_saver([ind, new_obj.pos])

        return object_list
