#!/usr/bin/env python3
import json
import os
import uuid
from datetime import datetime


def _default_log_path() -> str:
    """A unique JSONL path inside the persisted bags volume.

    Logs land in a `object_detections/` subfolder of the bags directory so they
    survive container restarts (unlike the container-local home dir). Falls back
    to the workspace/home bags dir, then ~, when the volume isn't mounted.
    """
    bags_dir = os.environ.get("ROS_BAGS_DIR")
    if not bags_dir:
        for candidate in ("/a2_ros_ws/bags", "/a2_ros/bags"):
            if os.path.isdir(candidate):
                bags_dir = candidate
                break
    if not bags_dir:
        bags_dir = os.path.expanduser("~")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique = uuid.uuid4().hex[:6]
    return os.path.join(
        bags_dir, "object_detections", f"object_detections_{stamp}_{unique}.jsonl"
    )


class DetectionLogger:
    """Appends per-detection records to a JSONL file for post-mission processing.

    Two record types:
      - "metadata": written once when camera intrinsics are known.
      - "detection": one per localised object per frame.

    Post-processing reads this file, looks up T_map←camera(timestamp_ns) from
    the SLAM trajectory, transforms point_cam → point_map, then clusters.
    """

    def __init__(self, file_path: str = ""):
        if not file_path:
            file_path = _default_log_path()
        # Resolve ~ and make relative paths land in home, not the node's cwd.
        file_path = os.path.expanduser(file_path)
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.path.expanduser("~"), file_path)
        self._path = file_path
        base, _ = os.path.splitext(self._path)
        self._global_path = base + "_global.json"
        parent = os.path.dirname(self._path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._f = open(self._path, "a")
        self._metadata_written = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_metadata(self, K, width: int, height: int, camera_frame: str) -> None:
        """Write camera intrinsics once (idempotent)."""
        if self._metadata_written:
            return
        self._write({
            "type": "metadata",
            "camera_frame": camera_frame,
            "K": K.tolist(),
            "image_width": int(width),
            "image_height": int(height),
        })
        self._metadata_written = True

    def log_detection(
        self,
        stamp,
        class_id: str,
        confidence: float,
        point_cam,
        bbox,
        camera_frame: str,
        estimation_type: str,
        point_map=None,
        map_frame: str = None,
    ) -> None:
        """Append one detection record.

        Args:
            stamp: ROS stamp (has .sec / .nanosec).
            class_id: COCO class name string.
            confidence: YOLO confidence score [0, 1].
            point_cam: [X, Y, Z] in camera optical frame (Z forward).
            bbox: [xmin, ymin, xmax, ymax] in pixels.
            camera_frame: TF frame id of the camera (needed for post TF lookup).
            estimation_type: "measurement" | "estimation" | "none".
            point_map: optional [X, Y, Z] in the map frame for this single frame
                (the odometry "best guess" used live, with no cross-frame
                averaging). None when the map transform was unavailable.
            map_frame: TF frame id the point_map is expressed in.
        """
        # Drop detections with no valid pose (sentinel pos[2] == -1).
        if point_cam[2] < 0:
            return
        timestamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        record = {
            "type": "detection",
            "timestamp_ns": timestamp_ns,
            "camera_frame": camera_frame,
            "class_id": class_id,
            "confidence": float(confidence),
            "point_cam": [float(v) for v in point_cam],
            "bbox": [int(v) for v in bbox],
            "estimation_type": estimation_type,
        }
        if point_map is not None:
            record["point_map"] = [float(v) for v in point_map]
            record["map_frame"] = map_frame
        self._write(record)

    def write_global_objects(self, global_objects, stamp=None) -> None:
        """Overwrite the global objects file with the current best-guess map.

        Args:
            global_objects: list of GlobalObject instances.
            stamp: optional ROS stamp (has .sec / .nanosec).
        """
        import json as _json
        timestamp_ns = None
        if stamp is not None:
            timestamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        data = {
            "timestamp_ns": timestamp_ns,
            "objects": [
                {
                    "id": int(g.id),
                    "class_id": g.name,
                    "position": [float(v) for v in g.position],
                    "confidence": float(g.confidence),
                    "observation_count": int(g.count),
                }
                for g in global_objects
            ],
        }
        with open(self._global_path, "w") as f:
            _json.dump(data, f, indent=2)

    @property
    def global_path(self) -> str:
        return self._global_path

    def close(self) -> None:
        self._f.close()

    @property
    def path(self) -> str:
        return self._path

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write(self, record: dict) -> None:
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()
