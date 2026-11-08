"""Detection, oriented box, instance mask, and pose result coordination."""

from collections import Counter

import cv2
import numpy as np


class DetectionMixin:
    """Internal detection rendering implementation for Visualizer."""

    def _annotate_detections(self, frame, annotated, result, names):
        obb = getattr(result, "obb", None)
        boxes = obb if obb is not None else getattr(result, "boxes", None)
        counts = Counter()
        if boxes is not None and len(boxes):
            # Transfer detection metadata together, avoiding repeated device syncs.
            if hasattr(boxes, "cpu"):
                boxes = boxes.cpu()
            classes = self._numpy(boxes.cls).astype(int)
            confidences = self._numpy(boxes.conf)
            ids = self._numpy(boxes.id).astype(int) if boxes.id is not None else [None] * len(boxes)
            geometry = self._numpy(boxes.xyxyxyxy if obb is not None else boxes.xyxy)
            selected = [i for i, cls in enumerate(classes)
                        if self._visible(int(cls), names.get(int(cls), str(cls)), ids[i])]
            masks = getattr(result, "masks", None)
            if selected and self.show_masks and masks is not None:
                if len(masks) != len(boxes):
                    raise ValueError("Expected one mask per detected object")
                data = self._numpy(masks.data)
                selected_masks = []
                polygons = None
                for i in selected:
                    if data[i].shape == frame.shape[:2]:
                        selected_masks.append(data[i])
                    else:
                        # masks.xy already accounts for inference letterboxing.
                        binary = np.zeros(frame.shape[:2], dtype=np.uint8)
                        if polygons is None:
                            polygons = masks.xy
                        polygon = np.asarray(polygons[i], dtype=np.int32).reshape(-1, 1, 2)
                        if len(polygon) >= 3:
                            cv2.fillPoly(binary, [polygon], 1)
                        selected_masks.append(binary)
                self.draw_masks(annotated, selected_masks,
                                [self.color(classes[i], names.get(int(classes[i]), "")) for i in selected])
            keypoints = getattr(result, "keypoints", None)
            poses = self._numpy(keypoints.data) if self.show_pose and keypoints is not None else None
            for i in selected:
                cls = int(classes[i])
                name = str(names.get(cls, cls))
                color = self.color(cls, name)
                counts[cls] += 1
                if obb is not None:
                    polygon = geometry[i]
                else:
                    x1, y1, x2, y2 = geometry[i]
                    polygon = np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                center = tuple(np.rint(polygon.mean(axis=0)).astype(int))
                tid = None if ids[i] is None else int(ids[i])
                if self.show_trails and tid is not None:
                    self._draw_trail(annotated, tid, center, color)
                if self.show_centers:
                    cv2.circle(annotated, center, 3, color, -1, cv2.LINE_AA)
                if self.show_boxes:
                    self.draw_obb(annotated, polygon, color)
                if poses is not None:
                    self.draw_pose(annotated, poses[i], color)
                if self.show_labels:
                    label = name
                    if self.show_ids and tid is not None:
                        label += f" #{tid}"
                    if self.show_confidence:
                        label += f" {confidences[i]:.2f}"
                    self.draw_label(annotated, label, polygon.min(axis=0), color)
        return counts
