"""Confidence-aware keypoint and skeleton rendering."""

import cv2
import numpy as np


class PoseMixin:
    """Internal implementation mixed into the public Visualizer API."""

    def draw_pose(self, frame, keypoints, color):
        """Draw visible points and limbs, honoring per-keypoint confidence."""
        points = self._numpy(keypoints)
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError("Keypoints must have shape (K, 2) or (K, 3)")
        h, w = frame.shape[:2]
        valid = np.isfinite(points).all(axis=1)
        valid &= (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)
        if points.shape[1] == 3:
            valid &= points[:, 2] >= self.keypoint_confidence
        else:
            valid &= np.any(points[:, :2] != 0, axis=1)
        edges = self.skeleton
        if edges is None:
            edges = self.COCO_SKELETON if len(points) == 17 else ()
        for a, b in edges:
            if max(a, b) >= len(points):
                raise ValueError("Skeleton index exceeds the number of keypoints")
            if valid[a] and valid[b]:
                cv2.line(frame, tuple(points[a, :2].astype(int)), tuple(points[b, :2].astype(int)),
                         color, self.line_width, cv2.LINE_AA)
        for point in points[valid, :2]:
            cv2.circle(frame, tuple(point.astype(int)), self.keypoint_radius, color, -1, cv2.LINE_AA)
        return frame

    COCO_SKELETON = (
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
        (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
        (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3),
        (2, 4), (3, 5), (4, 6),
    )
