#!/usr/bin/env python3
import numpy as np
import cv2
from rclpy.node import Node
from rclpy.logging import get_logger

AXIS_X = 0
AXIS_Y = 1


class PointProjector:
    def __init__(self, node):
        self.node = node
        self.logger = node.get_logger() if node else get_logger("point_projector")

        self.K = None
        self.dist_coeffs = None
        self.w = None
        self.h = None
        self.P = None

        # Standard ROS camera optical frame uses the Z-axis for forward depth.
        # (1-based index 3, which resolves to axis index 2 below)
        self.forward_axis = 3

        self.logger.info("PointProjector initialized successfully")

    def set_intrinsic_params(self, K, size, dist_coeffs=None):
        """Set camera intrinsic and distortion parameters."""
        self.K = K
        self.w = size[0]
        self.h = size[1]
        self.P = np.zeros((3, 4))
        self.P[:, :3] = self.K
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None and len(dist_coeffs) > 0 else None
        has_dist = self.dist_coeffs is not None and np.any(self.dist_coeffs != 0)
        self.logger.debug(f"Intrinsic parameters set (distortion={'yes' if has_dist else 'no'})")

    def project_points(self, points):
        """Project points from camera frame to image plane.

        Uses cv2.projectPoints when distortion coefficients are available so that
        projected LiDAR coordinates land in the same distorted pixel space as YOLO
        bounding boxes (which come from the raw camera image).
        """
        indices = np.arange(0, len(points))

        # Filter front hemisphere points (Z > 0 in optical frame)
        axis_idx = abs(self.forward_axis) - 1
        if self.forward_axis > 0:
            front_hemisphere = points[:, axis_idx] > 0
        else:
            front_hemisphere = points[:, axis_idx] < 0

        front_hemisphere_indices = np.nonzero(front_hemisphere)[0]
        indices = indices[front_hemisphere_indices]
        points = points[front_hemisphere_indices, :]

        if len(points) == 0:
            return np.empty((0, 2), dtype=np.float64), indices

        has_distortion = self.dist_coeffs is not None and np.any(self.dist_coeffs != 0)
        if has_distortion:
            # cv2.projectPoints expects (N,1,3) or (N,3); no rotation/translation
            # since points are already in the camera optical frame.
            pts, _ = cv2.projectPoints(
                points.astype(np.float64).reshape(-1, 1, 3),
                np.zeros(3),
                np.zeros(3),
                self.K,
                self.dist_coeffs,
            )
            xy = pts.reshape(-1, 2)
        else:
            # Pure pinhole projection
            homo_coor = np.ones(points.shape[0])
            XYZ = np.vstack((np.transpose(points), homo_coor))
            xy_h = self.P @ XYZ
            xy_h = xy_h / xy_h[2, None]
            xy = np.transpose(xy_h[:2, :])

        return xy, indices

    # def project_points_on_image(self, points):
    #     """Project 3D points onto image and filter to those within frame"""
    #     points_on_image, indices = self.project_points(points)
    #     points_on_image = np.uint32(np.squeeze(points_on_image))

    #     inside_frame_x = np.logical_and(
    #         (points_on_image[:, AXIS_X] >= 0), (points_on_image[:, AXIS_X] < self.w - 1)
    #     )
    #     inside_frame_y = np.logical_and(
    #         (points_on_image[:, AXIS_Y] >= 0), (points_on_image[:, AXIS_Y] < self.h - 1)
    #     )
    #     inside_frame_indices = np.nonzero(
    #         np.logical_and(inside_frame_x, inside_frame_y)
    #     )[0]

    #     indices = indices[inside_frame_indices]
    #     points_on_image = points_on_image[inside_frame_indices, :]
    #     return points_on_image, indices
    def project_points_on_image(self, points):
        """
        Project 3D points onto image and filter to those within frame

        Args:
            points: Nx3 array of 3D points
        Returns:
            points_on_image: Nx2 array of image coordinates
            indices: indices of valid points in original array
        """
        # Project points and get initial indices
        points_on_image, indices = self.project_points(points)

        # Handle empty input
        if points_on_image.size == 0:
            return np.array([]), np.array([])

        # Remove single-dimensional entries
        points_on_image = np.squeeze(points_on_image)

        # Handle single point case
        if len(points_on_image.shape) == 1:
            points_on_image = points_on_image.reshape(1, 2)

        # Filter on float coordinates first — clipping before this check would
        # silently pull out-of-frame points to the image border and let them pass.
        inside_frame_x = np.logical_and(
            points_on_image[:, AXIS_X] >= 0, points_on_image[:, AXIS_X] < self.w - 1
        )
        inside_frame_y = np.logical_and(
            points_on_image[:, AXIS_Y] >= 0, points_on_image[:, AXIS_Y] < self.h - 1
        )
        inside_frame_indices = np.nonzero(
            np.logical_and(inside_frame_x, inside_frame_y)
        )[0]

        valid_indices = indices[inside_frame_indices]
        valid_points = points_on_image[inside_frame_indices].astype(np.uint32)

        # Add debug logging
        self.logger.debug(
            f"Projected points shape: {valid_points.shape}, "
            f"Valid indices: {len(valid_indices)}/{len(points)}"
        )

        return valid_points, valid_indices
