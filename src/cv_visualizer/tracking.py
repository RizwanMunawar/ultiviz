"""Bounded track trails and sequence state."""

from collections import deque

import cv2
import numpy as np


class TrackingMixin:
    """Internal implementation mixed into the public Visualizer API."""

    def reset(self):
        """Clear visualization history before annotating a different sequence."""
        self._history = {}
        self._last_seen = {}
        self._frame_index = 0

    def _advance_tracks(self):
        self._frame_index += 1
        for tid in list(self._last_seen):
            if self._frame_index - self._last_seen[tid] > self.trail_max_gap + 1:
                del self._last_seen[tid]
                del self._history[tid]

    def _draw_trail(self, annotated, tid, center, color):
        history = self._history.setdefault(tid, deque(maxlen=self.trail_length))
        history.append(center)
        self._last_seen[tid] = self._frame_index
        if len(history) > 1:
            cv2.polylines(annotated, [np.asarray(history, dtype=np.int32)], False,
                          color, self.line_width, cv2.LINE_AA)
