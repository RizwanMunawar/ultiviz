"""Video capture, preview, writing, and resource cleanup."""

from pathlib import Path

import cv2
import numpy as np


class VideoMixin:
    """Internal implementation mixed into the public Visualizer API."""

    @staticmethod
    def _source_fps(source):
        """Read local metadata once; cameras/URLs and unknown FPS use 30."""
        if not isinstance(source, (str, Path)) or not Path(source).is_file():
            return 30.0
        capture = cv2.VideoCapture(str(source))
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            return float(fps) if np.isfinite(fps) and fps > 0 else 30.0
        finally:
            capture.release()

    def save_stream(self, results, output="annotated.mp4", *, fps=None, copy=True):
        """Render predictions into an MP4 without rerunning model inference.

        Uses the first result's local source FPS when available, otherwise 30.
        Only the current frame is retained. Returns the output Path and closes
        the prediction iterator and writer on success or failure.
        """
        iterator = iter(results)
        writer = None
        self.reset()
        try:
            output_path = Path(output)
            if output_path.suffix.lower() != ".mp4":
                raise ValueError("Stream output must use an .mp4 extension")
            if fps is not None and (not np.isfinite(fps) or fps <= 0):
                raise ValueError("fps must be finite and positive")
            for result in iterator:
                source = getattr(result, "path", None)
                protected = [path for path in (source, self._model_path)
                             if isinstance(path, (str, Path))]
                if any(output_path.resolve() == Path(path).resolve() for path in protected):
                    raise ValueError("Output must differ from source and model paths")
                frame = self.annotate(result, copy=copy)
                size = (frame.shape[1], frame.shape[0])
                if writer is None:
                    frame_size = size
                    rate = self._source_fps(source) if fps is None else fps
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                             rate, frame_size)
                    if not writer.isOpened():
                        raise RuntimeError(f"Cannot create output video: {output_path}")
                elif size != frame_size:
                    raise ValueError("Video frame dimensions changed during capture")
                writer.write(frame)
            if writer is None:
                raise ValueError("No prediction results to visualize")
        finally:
            try:
                if writer is not None:
                    writer.release()
            finally:
                close = getattr(iterator, "close", None)
                if callable(close):
                    close()
        return output_path

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
            self._validate_frame(frame)
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
                if frame.shape != (height, width, 3):
                    raise ValueError("Video frame dimensions changed during capture")
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
