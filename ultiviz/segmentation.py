"""Instance mask compositing and semantic segmentation rendering."""

import cv2
import numpy as np


class SegmentationMixin:
    """Internal implementation mixed into the public Visualizer API."""

    def draw_masks(self, frame, masks, colors):
        """Blend masks once; later instances own overlapping pixels and borders.

        Each mask must match the frame resolution. Interiors, including holes,
        are preserved; no tint is applied to pixels outside the masks.
        """
        owner = np.full(frame.shape[:2], -1, dtype=np.int32)
        overlay = frame.copy()
        regions = []
        for index, (mask, color) in enumerate(zip(masks, colors)):
            binary = np.asarray(mask) > 0.5
            if binary.shape != frame.shape[:2]:
                raise ValueError("Masks must match the frame height and width")
            overlay[binary], owner[binary] = color, index
            regions.append((binary.astype(np.uint8), color))
        cv2.addWeighted(overlay, self.mask_opacity, frame, 1 - self.mask_opacity, 0, dst=frame)
        if self.boundary_thickness:
            for index, (binary, color) in enumerate(regions):
                eroded = cv2.erode(binary, np.ones((3, 3), np.uint8),
                                   iterations=self.boundary_thickness,
                                   borderType=cv2.BORDER_CONSTANT, borderValue=0)
                frame[(binary > eroded) & (owner == index)] = color
        return frame

    def _annotate_semantic(self, frame, annotated, semantic, names):
        """Render a native semantic class map and optional coverage chart."""
        class_map = self._numpy(semantic.data)
        if class_map.ndim == 3 and class_map.shape[0] == 1:
            class_map = class_map[0]
        if class_map.ndim != 2:
            raise ValueError("Expected a 2D semantic class map")
        if class_map.shape != frame.shape[:2]:
            class_map = cv2.resize(class_map.astype(np.float32), (frame.shape[1], frame.shape[0]),
                                   interpolation=cv2.INTER_NEAREST).astype(np.int32)
        counts, masks, palette = {}, [], []
        for cls in np.unique(class_map):
            cls = int(cls)
            name = names.get(cls, str(cls))
            if cls in self.ignored_semantic_ids or not self._visible(cls, name):
                continue
            mask = class_map == cls
            counts[cls] = int(mask.sum())
            masks.append(mask)
            palette.append(self.color(cls, name))
        if self.show_masks:
            self.draw_masks(annotated, masks, palette)
        if self.show_chart:
            self.draw_chart(annotated, counts, names, semantic=True)
        return annotated
