"""Per-frame timing without device synchronization or retained frame history."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from time import perf_counter


@dataclass(frozen=True, slots=True)
class FrameTiming:
    """Milliseconds for one frame; unavailable model metrics remain None."""

    frame_index: int
    inference_ms: float | None
    visualization_ms: float
    preprocess_ms: float | None = None
    postprocess_ms: float | None = None
    model_call_ms: float | None = None

    @property
    def processing_ms(self):
        """Inference plus visualization; excludes preprocessing and output I/O."""
        return None if self.inference_ms is None else self.inference_ms + self.visualization_ms


class TimingMixin:
    """Measure rendering separately from model, file, preview, and log work."""

    def _render_timed(self, frame, result=None, *, copy=True, model_call_ms=None):
        if not self.timing:
            return self._annotate_frame(frame, result, copy=copy)
        prediction = frame if result is None else result
        start = perf_counter()
        annotated = self._annotate_frame(frame, result, copy=copy)
        visualization_ms = (perf_counter() - start) * 1000
        speed = getattr(prediction, "speed", None)
        speed = speed if isinstance(speed, dict) else {}

        def metric(name):
            value = speed.get(name)
            return float(value) if isinstance(value, Real) and isfinite(value) and value >= 0 else None

        record = FrameTiming(self._frame_index, metric("inference"), visualization_ms,
                             metric("preprocess"), metric("postprocess"), model_call_ms)
        self.last_timing = record
        if self.on_timing is not None:
            self.on_timing(record)
        else:
            inference = "n/a" if record.inference_ms is None else f"{record.inference_ms:.2f} ms"
            processing = "" if record.processing_ms is None else f" | processing: {record.processing_ms:.2f} ms"
            call = "" if model_call_ms is None else f" | model call: {model_call_ms:.2f} ms"
            print(f"Frame {record.frame_index} | inference: {inference} | "
                  f"visualization: {visualization_ms:.2f} ms{processing}{call}")
        return annotated
