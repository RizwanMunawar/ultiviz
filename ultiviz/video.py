"""Video capture, preview, writing, and resource cleanup."""

from collections.abc import Iterator
from itertools import chain
from pathlib import Path
import os
import sys

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

    def _check_output(self, output, source):
        for path in (source, self._model_path):
            if isinstance(path, (str, Path)) and output.resolve() == Path(path).resolve():
                raise ValueError("Output must differ from source and model paths")

    @staticmethod
    def _preview(frame, delay):
        if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise RuntimeError("Preview needs a desktop display; use show=False for headless export")
        height, width = frame.shape[:2]
        if width > 1280:
            frame = cv2.resize(frame, (1280, max(1, round(height * 1280 / width))))
        try:
            cv2.imshow("ultiviz", frame)
            return cv2.waitKey(delay) & 0xFF in (ord("q"), 27)
        except cv2.error as exc:
            raise RuntimeError("Cannot open preview; use show=False for headless export") from exc

    @staticmethod
    def _close_preview():
        try:
            cv2.destroyWindow("ultiviz")
        except cv2.error:
            pass  # No window exists if preview initialization failed.

    def present(self, results, *, output=None, save=True, show=True, fps=None, copy=True):
        """Save/display predictions immediately and return annotated image data."""
        if isinstance(results, Iterator):
            try:
                first = next(results, None)
                if first is None:
                    raise ValueError("No prediction results to visualize")
                source = getattr(first, "path", None)
                items = chain([first], results)
                if (isinstance(source, (str, Path)) and Path(source).suffix.lower()
                        in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}):
                    return self._present_images(items, output=output, save=save, show=show,
                                                copy=copy, collect=False)
                return self.save_stream(items, output or "annotated.mp4", fps=fps,
                                        copy=copy, show=show, save=save, return_frame=True)
            finally:
                close = getattr(results, "close", None)
                if callable(close):
                    close()
        items = results if isinstance(results, (list, tuple)) else [results]
        if not items:
            raise ValueError("No prediction results to visualize")
        source = getattr(items[0], "path", None)
        video_suffixes = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".ts", ".asf"}
        if isinstance(source, (str, Path)) and Path(source).suffix.lower() in video_suffixes:
            return self.save_stream(items, output or "annotated.mp4", fps=fps,
                                    copy=copy, show=show, save=save, return_frame=True)
        return self._present_images(items, output=output, save=save, show=show, copy=copy)

    def _present_images(self, items, *, output, save, show, copy, collect=True):
        path = Path(output or "annotated.jpg")
        if save and path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
            raise ValueError("Image output must use a supported image extension")
        frames = []
        last_frame = None
        try:
            for index, result in enumerate(items):
                frame = self.annotate(result, copy=copy)
                target = path if not index else path.with_name(f"{path.stem}_{index + 1}{path.suffix}")
                if save:
                    self._check_output(target, getattr(result, "path", None))
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(target), frame):
                        raise RuntimeError(f"Cannot save output image: {target}")
                last_frame = frame
                if collect:
                    frames.append(frame)
                if show and self._preview(frame, 0):
                    break
        finally:
            if show:
                self._close_preview()
        return (frames[0] if len(frames) == 1 else frames) if collect else last_frame

    def save_stream(self, results, output="annotated.mp4", *, fps=None, copy=True,
                    show=False, save=True, return_frame=False):
        """Consume predictions immediately; save/preview with bounded memory.

        Uses local source FPS or 30. Returns a Path by default, or the last
        annotated frame with return_frame=True. Always closes acquired resources.
        """
        iterator = iter(results)
        writer = None
        last_frame = None
        self.reset()
        try:
            output_path = Path(output)
            if save and output_path.suffix.lower() != ".mp4":
                raise ValueError("Stream output must use an .mp4 extension")
            if fps is not None and (not np.isfinite(fps) or fps <= 0):
                raise ValueError("fps must be finite and positive")
            for result in iterator:
                source = getattr(result, "path", None)
                if save:
                    self._check_output(output_path, source)
                frame = self.annotate(result, copy=copy)
                size = (frame.shape[1], frame.shape[0])
                if last_frame is None:
                    frame_size = size
                    rate = self._source_fps(source) if fps is None else fps
                    if save:
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 rate, frame_size)
                        if not writer.isOpened():
                            raise RuntimeError(f"Cannot create output video: {output_path}")
                elif size != frame_size:
                    raise ValueError("Video frame dimensions changed during capture")
                if writer is not None:
                    writer.write(frame)
                last_frame = frame
                if show and self._preview(frame, max(1, round(1000 / rate))):
                    break
            if last_frame is None:
                raise ValueError("No prediction results to visualize")
        finally:
            try:
                if writer is not None:
                    writer.release()
            finally:
                try:
                    if show:
                        self._close_preview()
                finally:
                    close = getattr(iterator, "close", None)
                    if callable(close):
                        close()
        return last_frame if return_frame else output_path

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
