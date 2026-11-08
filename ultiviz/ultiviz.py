"""Ultralytics visualization utilities and automatic result dispatch."""

from pathlib import Path
from collections.abc import Iterator

import numpy as np

from .charts import ChartMixin
from .colors import ColorMixin
from .drawing import DrawingMixin
from .detections import DetectionMixin
from .inference import InferenceMixin
from .pose import PoseMixin
from .segmentation import SegmentationMixin
from .tracking import TrackingMixin
from .video import VideoMixin


def _is_result(value):
    return isinstance(getattr(value, "orig_img", None), np.ndarray)


class _VisualizerType(type):
    """Allow direct result rendering while retaining the configurable class API."""

    def __call__(cls, model=None, **options):
        if _is_result(model) or isinstance(model, (list, tuple, Iterator)):
            copy = options.pop("copy", True)
            if options.pop("save", False):
                output = options.pop("output", "annotated.mp4")
                fps = options.pop("fps", None)
                renderer = super().__call__(**options)
                results = [model] if _is_result(model) else model
                return renderer.save_stream(results, output, fps=fps, copy=copy)
            renderer = super().__call__(**options)
            return renderer(model, copy=copy)
        return super().__call__(model, **options)


class Visualizer(
    ColorMixin, DrawingMixin, DetectionMixin, SegmentationMixin, PoseMixin, ChartMixin,
    TrackingMixin, InferenceMixin, VideoMixin, metaclass=_VisualizerType,
):
    """Render detection, segmentation, pose, OBB, classification and tracks.

    Visualizer(results) directly renders a prediction or prediction list;
    a single result returns an image, multiple results return a list of images,
    and an iterator returns a lazy iterator of images, for images and videos
    alike. Saving requires save=True. For repeated calls, create Visualizer()
    once and call it with each result.

    The model is optional. Use annotate() with an existing Ultralytics Results
    object, or pass model weights/a model object for process_frame() and run().
    Tracking applies to detect/segment/pose/OBB models. A custom pose topology
    can be supplied as skeleton=[(0, 1), ...]; COCO edges are automatic only
    for 17-keypoint poses. Other layouts draw points only by default.
    """

    def __init__(
        self, model=None, *, tracking=False, tracker="bytetrack.yaml",
        confidence=0.25, iou=0.45, image_size=1280, device=None,
        class_ids=None, class_colors=None, hidden_classes=(), hidden_track_ids=(),
        show_boxes=True, show_labels=True, show_confidence=False, show_ids=True,
        show_masks=True, show_pose=True, show_trails=False, show_centers=False,
        show_chart=False, mask_opacity=0.45, label_opacity=0.9,
        line_width=2, corner_radius=8, boundary_thickness=2,
        font_scale=0.7, text_thickness=1, padding=7,
        keypoint_confidence=0.25, keypoint_radius=4, skeleton=None,
        trail_length=45, trail_max_gap=30, ignored_semantic_ids=(255,),
    ):
        for name, value in (
            ("image_size", image_size), ("line_width", line_width),
            ("text_thickness", text_thickness), ("keypoint_radius", keypoint_radius),
            ("trail_length", trail_length), ("corner_radius", corner_radius),
            ("boundary_thickness", boundary_thickness), ("padding", padding),
            ("trail_max_gap", trail_max_gap),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
        if image_size < 1:
            raise ValueError("image_size must be positive")
        for name, value in (
            ("confidence", confidence), ("iou", iou),
            ("mask_opacity", mask_opacity), ("label_opacity", label_opacity),
            ("keypoint_confidence", keypoint_confidence),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if min(line_width, text_thickness, keypoint_radius, trail_length) < 1:
            raise ValueError("Drawing widths, point radius and trail length must be positive")
        if (min(corner_radius, boundary_thickness, padding, trail_max_gap) < 0
                or not np.isfinite(font_scale) or font_scale <= 0):
            raise ValueError("Drawing sizes must be nonnegative; font_scale must be positive")
        self.model = model
        self._model_path = (Path(model) if isinstance(model, (str, Path))
                            else getattr(model, "ckpt_path", None))
        self.tracking, self.tracker = tracking or show_trails, tracker
        self.confidence, self.iou = confidence, iou
        self.image_size, self.device = image_size, device
        self.class_ids = None if class_ids is None else set(class_ids)
        self.class_colors = dict(class_colors or {})
        self.hidden_classes = {self.normalize_name(name) for name in hidden_classes}
        self.hidden_track_ids = set(hidden_track_ids)
        self.show_boxes, self.show_labels = show_boxes, show_labels
        self.show_confidence, self.show_ids = show_confidence, show_ids
        self.show_masks, self.show_pose = show_masks, show_pose
        self.show_trails, self.show_centers = show_trails, show_centers
        self.show_chart = show_chart
        self.mask_opacity, self.label_opacity = mask_opacity, label_opacity
        self.line_width, self.corner_radius = line_width, corner_radius
        self.boundary_thickness = boundary_thickness
        self.font_scale, self.text_thickness, self.padding = font_scale, text_thickness, padding
        self.keypoint_confidence, self.keypoint_radius = keypoint_confidence, keypoint_radius
        self.skeleton = None if skeleton is None else tuple(tuple(edge) for edge in skeleton)
        if self.skeleton is not None and any(
            len(edge) != 2 or any(not isinstance(i, int) or i < 0 for i in edge)
            for edge in self.skeleton
        ):
            raise ValueError("Skeleton edges must contain two nonnegative integer indices")
        self.trail_length, self.trail_max_gap = trail_length, trail_max_gap
        self.ignored_semantic_ids = set(ignored_semantic_ids)
        self.reset()

    @staticmethod
    def _numpy(value):
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        return np.asarray(value)

    def _visible(self, cls, name, track_id=None):
        return ((self.class_ids is None or cls in self.class_ids)
                and self.normalize_name(name) not in self.hidden_classes
                and track_id not in self.hidden_track_ids)

    @staticmethod
    def _validate_frame(frame):
        if (not isinstance(frame, np.ndarray) or frame.ndim != 3
                or frame.shape[2] != 3 or frame.dtype != np.uint8
                or min(frame.shape[:2]) == 0):
            raise ValueError("Expected a nonempty uint8 BGR frame with shape (H, W, 3)")

    def __call__(self, results, *, copy=True):
        """Render a result, a prediction list, or a lazy prediction iterator.

        A singleton list returns one image; a longer list returns images in
        order. Empty lists raise ValueError. Iterators stay lazy even when
        they contain only one result, keeping video memory bounded.
        """
        if isinstance(results, (list, tuple)):
            if not results:
                raise ValueError("No prediction results to visualize")
            if len(results) == 1:
                return self.annotate(results[0], copy=copy)
            return [self.annotate(result, copy=copy) for result in results]
        if isinstance(results, Iterator):
            return self._render_stream(results, copy=copy)
        return self.annotate(results, copy=copy)

    def _render_stream(self, results, *, copy):
        try:
            for result in results:
                yield self.annotate(result, copy=copy)
        finally:
            close = getattr(results, "close", None)
            if callable(close):
                close()

    def annotate(self, frame, result=None, *, copy=True):
        """Auto-detect result type and return an annotated copy of a BGR frame.

        Pass a Results object alone to use result.orig_img, or (frame, result)
        for an explicit image. copy=False draws in place on a writable,
        C-contiguous frame. Call once per frame, in sequence, for trails.
        """
        if result is None:
            result = frame
            frame = getattr(result, "orig_img", None)
        self._validate_frame(frame)
        original_shape = getattr(result, "orig_shape", None)
        if original_shape is not None and tuple(original_shape) != frame.shape[:2]:
            raise ValueError("Frame dimensions must match the result's original image")
        if not copy and (not frame.flags.writeable or not frame.flags.c_contiguous):
            raise ValueError("copy=False requires a writable C-contiguous frame")
        annotated = frame.copy() if copy else frame
        self._advance_tracks()
        raw_names = result.names
        names = dict(raw_names) if isinstance(raw_names, dict) else dict(enumerate(raw_names))
        semantic = getattr(result, "semantic_mask", None)
        if semantic is not None:
            return self._annotate_semantic(frame, annotated, semantic, names)

        counts = self._annotate_detections(frame, annotated, result, names)
        probs = getattr(result, "probs", None)
        if probs is not None and self.show_labels:
            probabilities = self._numpy(probs.data)
            for rank, cls in enumerate(np.argsort(-probabilities)[:5]):
                name = names.get(cls, str(cls))
                self.draw_label(annotated, f"{name} {float(probabilities[cls]):.2f}",
                                (8, 8 + rank * 42), self.color(cls, name), above=False)
        elif self.show_chart:
            self.draw_chart(annotated, counts, names)
        return annotated

