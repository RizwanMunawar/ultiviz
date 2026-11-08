"""Depth heatmaps using vectorized OpenCV operations."""

import cv2
import numpy as np


class DepthMixin:
    """Render existing depth predictions without another inference dependency."""

    def _annotate_depth(self, annotated, depth):
        if not self.show_depth or self.mask_opacity == 0:
            return annotated
        values = np.asarray(self._numpy(depth.data), dtype=np.float32)
        if values.ndim == 3 and values.shape[0] == 1:
            values = values[0]
        if values.ndim != 2 or values.shape != annotated.shape[:2]:
            raise ValueError("Depth must be a 2D map matching the frame height and width")
        valid = (np.isfinite(values) & (values > 0)).astype(np.uint8)
        if not np.any(valid):
            return annotated
        normalized = cv2.normalize(values, None, 0, 255, cv2.NORM_MINMAX,
                                   dtype=cv2.CV_8U, mask=valid)
        heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        blended = cv2.addWeighted(annotated, 1 - self.mask_opacity,
                                  heatmap, self.mask_opacity, 0)
        cv2.copyTo(blended, valid, annotated)
        return annotated
