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
        if len(masks) != len(colors):
            raise ValueError("Expected one color per mask")
        regions = []
        height, width = frame.shape[:2]
        left, top, right, bottom = width, height, 0, 0
        for mask, color in zip(masks, colors):
            binary = (np.asarray(mask) > 0.5).astype(np.uint8)
            if binary.shape != (height, width):
                raise ValueError("Masks must match the frame height and width")
            x, y, w, h = cv2.boundingRect(binary)
            if not w or not h:
                continue
            # Retain only occupied regions, rather than one full frame per mask.
            regions.append((x, y, binary[y:y + h, x:x + w].copy(), color))
            left, top = min(left, x), min(top, y)
            right, bottom = max(right, x + w), max(bottom, y + h)
        if not regions or (not self.mask_opacity and not self.boundary_thickness):
            return frame
        roi = frame[top:bottom, left:right]
        overlay = roi.copy()
        owner = np.full(roi.shape[:2], -1, dtype=np.int32) if self.boundary_thickness else None
        for index, (x, y, binary, color) in enumerate(regions):
            h, w = binary.shape
            x, y = x - left, y - top
            active = binary != 0
            overlay[y:y + h, x:x + w][active] = color
            if owner is not None:
                owner[y:y + h, x:x + w][active] = index
        cv2.addWeighted(overlay, self.mask_opacity, roi, 1 - self.mask_opacity, 0, dst=roi)
        if owner is not None:
            kernel = np.ones((3, 3), np.uint8)
            for index, (x, y, binary, color) in enumerate(regions):
                h, w = binary.shape
                x, y = x - left, y - top
                eroded = cv2.erode(binary, kernel, iterations=self.boundary_thickness,
                                   borderType=cv2.BORDER_CONSTANT, borderValue=0)
                roi[y:y + h, x:x + w][
                    (binary > eroded) & (owner[y:y + h, x:x + w] == index)
                ] = color
        return frame

    def _annotate_semantic(self, frame, annotated, semantic, names):
        """Render a native semantic class map and optional coverage chart."""
        if not self.show_masks and not self.show_chart:
            return annotated
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
