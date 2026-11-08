"""Lazy optional Ultralytics inference and tracking integration."""

from pathlib import Path


class InferenceMixin:
    """Internal implementation mixed into the public Visualizer API."""

    def _load_model(self):
        if self.model is None:
            raise ValueError("Pass model weights or a YOLO model to run inference")
        if isinstance(self.model, (str, Path)):
            try:
                from ultralytics import YOLO
            except ModuleNotFoundError as exc:
                if exc.name != "ultralytics":
                    raise
                raise ImportError(
                    'YOLO inference requires the optional extra: '
                    'pip install "ultiviz[ultralytics]"'
                ) from exc
            self.model = YOLO(str(self.model))
        return self.model

    def process_frame(self, frame, *, copy=True):
        """Infer and render one BGR frame; copy=False allows in-place drawing."""
        self._validate_frame(frame)
        if not copy and (not frame.flags.writeable or not frame.flags.c_contiguous):
            raise ValueError("copy=False requires a writable C-contiguous frame")
        model = self._load_model()
        task = model.task
        if self.tracking and task not in {"detect", "segment", "pose", "obb"}:
            raise ValueError(f"Tracking is unsupported for task {task!r}")
        kwargs = dict(imgsz=self.image_size, verbose=False)
        if self.device is not None:
            kwargs["device"] = self.device
        if task in {"detect", "segment", "pose", "obb"}:
            kwargs.update(conf=self.confidence, iou=self.iou,
                          classes=None if self.class_ids is None else sorted(self.class_ids))
        if task == "segment":
            kwargs["retina_masks"] = True
        if self.tracking:
            result = model.track(frame, persist=True, tracker=self.tracker, **kwargs)[0]
        else:
            result = model.predict(frame, **kwargs)[0]
        return self.annotate(frame, result, copy=copy)
