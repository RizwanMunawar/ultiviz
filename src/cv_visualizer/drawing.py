"""Rounded boxes, labels, regions, and counting lines."""

import cv2
import numpy as np


class DrawingMixin:
    """Internal implementation mixed into the public Visualizer API."""

    @staticmethod
    def _rounded_polygon(polygon, radius):
        polygon = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
        points = []
        for i, corner in enumerate(polygon):
            incoming, outgoing = polygon[i - 1] - corner, polygon[(i + 1) % len(polygon)] - corner
            a, b = np.linalg.norm(incoming), np.linalg.norm(outgoing)
            if min(a, b) < 1e-6 or radius <= 0:
                points.append(corner)
                continue
            trim = min(radius, a / 2, b / 2)
            start, end = corner + incoming * trim / a, corner + outgoing * trim / b
            for t in np.linspace(0, 1, 10):
                points.append((1 - t) ** 2 * start + 2 * (1 - t) * t * corner + t ** 2 * end)
        return np.rint(points).astype(np.int32).reshape(-1, 1, 2)

    def draw_label(self, frame, text, anchor, color, *, above=True):
        """Draw a clipped rounded label; anchor is its preferred left/bottom."""
        text = str(text)
        if not text:
            return frame
        height, width = frame.shape[:2]
        padding = min(self.padding, max(0, (min(height, width) - 1) // 4))
        scale = self.font_scale
        for _ in range(3):
            (tw, th), baseline = cv2.getTextSize(text, 0, scale, self.text_thickness)
            ratio = min(1, (width - 2 * padding) / max(1, tw),
                        (height - 2 * padding) / max(1, th + baseline))
            if ratio >= 1:
                break
            scale *= ratio * 0.95
        lw, lh = min(width, tw + 2 * padding), min(height, th + baseline + 2 * padding)
        x, y = map(int, anchor)
        top = y - lh if above and y >= lh else y
        left, top = int(np.clip(x, 0, width - lw)), int(np.clip(top, 0, height - lh))
        roi = frame[top:top + lh, left:left + lw]
        overlay = roi.copy()
        polygon = self._rounded_polygon([(0, 0), (lw - 1, 0), (lw - 1, lh - 1), (0, lh - 1)], self.corner_radius)
        cv2.fillPoly(overlay, [polygon], color, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, self.label_opacity, roi, 1 - self.label_opacity, 0, dst=roi)
        b, g, r = color
        ink = (17, 31, 60) if 0.299 * r + 0.587 * g + 0.114 * b > 150 else (255, 255, 255)
        cv2.putText(roi, text, (padding, padding + th), 0, scale, ink, self.text_thickness, cv2.LINE_AA)
        return frame

    def draw_box(self, frame, xyxy, color, label=None):
        """Draw an axis-aligned rounded box and optional label."""
        x1, y1, x2, y2 = map(float, xyxy)
        return self.draw_obb(frame, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)], color, label)

    def draw_obb(self, frame, polygon, color, label=None):
        polygon = np.asarray(polygon, dtype=float).reshape(-1, 2)
        cv2.polylines(frame, [self._rounded_polygon(polygon, self.corner_radius)],
                      True, color, self.line_width, cv2.LINE_AA)
        if label:
            self.draw_label(frame, label, polygon.min(axis=0), color)
        return frame

    def draw_region(self, frame, polygon, color, label=None, opacity=0.15):
        """Draw a counting/proximity zone using coordinates supplied by its app."""
        if not 0 <= opacity <= 1:
            raise ValueError("opacity must be between 0 and 1")
        points = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
        if len(points) < 3:
            raise ValueError("A region needs at least three points")
        overlay = frame.copy()
        cv2.fillPoly(overlay, [points], color)
        cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, dst=frame)
        cv2.polylines(frame, [points], True, color, self.line_width, cv2.LINE_AA)
        if label:
            self.draw_label(frame, label, points.reshape(-1, 2).min(axis=0), color)
        return frame

    def draw_line(self, frame, start, end, color, label=None):
        """Draw a counting line with endpoint markers and an optional counter."""
        start, end = tuple(map(int, start)), tuple(map(int, end))
        cv2.line(frame, start, end, color, self.line_width, cv2.LINE_AA)
        for point in (start, end):
            cv2.circle(frame, point, self.line_width + 3, color, -1, cv2.LINE_AA)
        if label:
            self.draw_label(frame, label, min((start, end), key=lambda p: p[1]), color)
        return frame
