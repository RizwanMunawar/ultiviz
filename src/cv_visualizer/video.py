"""Video capture, preview, writing, and resource cleanup."""

from pathlib import Path

import cv2
import numpy as np


class VideoMixin:
    """Internal implementation mixed into the public Visualizer API."""

    def run(self, source, output="output-visualization.mp4", *, show_window=True, max_preview_width=1280):
        """Process a video/camera, save optional MP4 output, and preview until Q.

        Pass an integer source for a camera. Paths resolve from the current
        working directory. output=None disables saving. Each run starts a new
        visualization/tracker sequence; process_frame() retains the sequence.
        """
        if max_preview_width < 1:
            raise ValueError("max_preview_width must be positive")
        output_path = Path(output) if output is not None else None
        if output_path is not None:
            protected = [Path(source)] if isinstance(source, (str, Path)) else []
            if self._model_path is not None:
                protected.append(Path(self._model_path))
            if any(output_path.resolve() == path.resolve() for path in protected):
                raise ValueError("Output must differ from source and model paths")
        model = self._load_model()
        self.reset()
        for tracker in getattr(getattr(model, "predictor", None), "trackers", ()):
            tracker.reset()
        cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
        writer = None
        try:
            success, frame = cap.read()
            if not success:
                raise RuntimeError(f"Cannot read video source: {source}")
            height, width = frame.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(fps) or fps <= 0:
                fps = 30.0
            if output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot create output video: {output_path}")
            while success:
                annotated = self.process_frame(frame)
                if writer is not None:
                    writer.write(annotated)
                if show_window:
                    scale = min(1.0, max_preview_width / width)
                    preview = cv2.resize(annotated, (max(1, round(width * scale)), max(1, round(height * scale))))
                    cv2.imshow("Visualizer | Q to quit", preview)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                success, frame = cap.read()
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if show_window:
                cv2.destroyAllWindows()
        return output_path
