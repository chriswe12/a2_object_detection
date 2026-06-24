#!/usr/bin/env python3
import json
import os
from datetime import datetime


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
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"~/object_detections_{ts}.jsonl"
        # Resolve ~ and make relative paths land in home, not the node's cwd.
        file_path = os.path.expanduser(file_path)
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.path.expanduser("~"), file_path)
        self._path = file_path
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
        """
        # Drop detections with no valid pose (sentinel pos[2] == -1).
        if point_cam[2] < 0:
            return
        timestamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        self._write({
            "type": "detection",
            "timestamp_ns": timestamp_ns,
            "camera_frame": camera_frame,
            "class_id": class_id,
            "confidence": float(confidence),
            "point_cam": [float(v) for v in point_cam],
            "bbox": [int(v) for v in bbox],
            "estimation_type": estimation_type,
        })

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
