"""Current-frame object distribution and semantic coverage charts."""

import cv2
import numpy as np


class ChartMixin:
    """Internal implementation mixed into the public Visualizer API."""

    def draw_chart(self, frame, counts, names, *, semantic=False):
        """Draw a compact donut and legend, scaling the card to the frame."""
        entries = sorted(((int(k), int(v)) for k, v in counts.items() if v > 0),
                         key=lambda item: (-item[1], item[0]))
        total = sum(v for _, v in entries)
        card = np.full((280, 480, 3), (52, 28, 8), dtype=np.uint8)
        title = "Class coverage" if semantic else f"Objects: {total}"
        cv2.putText(card, title, (18, 30), 0, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        angle = -90.0
        for cls, count in entries:
            end = angle + count / total * 360
            color = self.color(cls, names.get(cls, str(cls)))
            cv2.ellipse(card, (120, 155), (90, 90), 0, angle, end, color, -1, cv2.LINE_AA)
            angle = end
        cv2.circle(card, (120, 155), 48, (52, 28, 8), -1, cv2.LINE_AA)
        if not total:
            cv2.putText(card, "No objects", (68, 160), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        for row, (cls, count) in enumerate(entries[:7]):
            y = 65 + row * 29
            name = str(names.get(cls, cls))
            name = name[:18] + "..." if len(name) > 21 else name
            value = f"{100 * count / total:.1f}%" if semantic else str(count)
            cv2.circle(card, (230, y - 4), 5, self.color(cls, name), -1)
            cv2.putText(card, f"{name}: {value}", (244, y), 0, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
        if len(entries) > 7:
            cv2.putText(card, f"+{len(entries) - 7} more classes", (230, 271), 0, 0.4, (255, 255, 255), 1)
        h, w = frame.shape[:2]
        scale = min(1, max(1, w - 16) / 480, max(1, h - 16) / 280)
        card = cv2.resize(card, (max(1, round(480 * scale)), max(1, round(280 * scale))))
        ch, cw = card.shape[:2]
        x, y = max(0, w - cw - 8), min(8, h - ch)
        roi = frame[y:y + ch, x:x + cw]
        cv2.addWeighted(card, 0.94, roi, 0.06, 0, dst=roi)
        return frame
