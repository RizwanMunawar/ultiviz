"""Per-frame metrics, model-call boundaries, disabled overhead, and reporting."""

import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from ultraviz import FrameTiming, Visualizer


class TimingTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((48, 64, 3), np.uint8)
        self.result = SimpleNamespace(orig_img=self.frame, names={},
                                      speed={"inference": 5.0, "preprocess": 1.0, "postprocess": 2.0})

    def test_report_contains_model_and_visualization_time_for_each_frame(self):
        visualizer = Visualizer()
        output = io.StringIO()
        with contextlib.redirect_stdout(output), \
                patch("ultiviz.timing.perf_counter", side_effect=[1, 1.002, 2, 2.003]):
            visualizer(self.result)
            visualizer(self.result)
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("Frame 1 | inference: 5.00 ms | visualization: 2.00 ms", lines[0])
        self.assertIn("processing: 7.00 ms", lines[0])
        self.assertIn("Frame 2", lines[1])
        self.assertEqual(visualizer.last_timing.frame_index, 2)
        self.assertAlmostEqual(visualizer.last_timing.visualization_ms, 3)

    def test_callback_receives_structured_metrics_without_console_output(self):
        callback = Mock()
        visualizer = Visualizer(on_timing=callback)
        with patch("builtins.print") as console:
            visualizer(self.result)
        console.assert_not_called()
        record = callback.call_args.args[0]
        self.assertIsInstance(record, FrameTiming)
        self.assertIs(record, visualizer.last_timing)
        self.assertEqual((record.inference_ms, record.preprocess_ms, record.postprocess_ms), (5, 1, 2))
        self.assertIsNone(record.model_call_ms)
        self.assertAlmostEqual(record.processing_ms, 5 + record.visualization_ms)

    def test_model_call_is_timed_separately_from_rendering(self):
        model = Mock(task="detect")
        model.predict.return_value = [self.result]
        callback = Mock()
        visualizer = Visualizer(model, on_timing=callback)
        with patch("ultiviz.inference.perf_counter", side_effect=[1, 1.010]), \
                patch("ultiviz.timing.perf_counter", side_effect=[2, 2.003]):
            visualizer.process_frame(self.frame)
        record = visualizer.last_timing
        self.assertAlmostEqual(record.model_call_ms, 10)
        self.assertAlmostEqual(record.visualization_ms, 3)
        self.assertEqual(record.inference_ms, 5)
        model.predict.assert_called_once()

    def test_missing_and_invalid_model_metrics_are_not_reported_as_zero(self):
        visualizer = Visualizer(on_timing=Mock())
        for speed in (None, {}, {"inference": None}, {"inference": -1},
                      {"inference": float("nan")}, {"inference": float("inf")},
                      {"inference": "fast"}):
            self.result.speed = speed
            visualizer(self.result)
            self.assertIsNone(visualizer.last_timing.inference_ms)
            self.assertIsNone(visualizer.last_timing.processing_ms)
        self.result.speed = {"inference": 0}
        visualizer(self.result)
        self.assertEqual(visualizer.last_timing.inference_ms, 0)

    def test_disabled_timing_avoids_clocks_records_callbacks_and_console(self):
        callback = Mock()
        visualizer = Visualizer(timing=False, on_timing=callback)
        with patch("ultiviz.timing.perf_counter") as clock, patch("builtins.print") as console:
            output = visualizer(self.result)
        self.assertEqual(output.shape, self.frame.shape)
        clock.assert_not_called()
        console.assert_not_called()
        callback.assert_not_called()
        self.assertIsNone(visualizer.last_timing)

    def test_stream_reports_incrementally_without_retaining_history(self):
        records = []
        def results():
            yield self.result
            self.assertEqual(len(records), 1)
            yield self.result
        output = Visualizer(results(), save=False, show=False, on_timing=records.append)
        self.assertEqual(records, [])
        self.assertEqual(len(list(output)), 2)
        self.assertEqual([record.frame_index for record in records], [1, 2])

    def test_reset_clears_last_record_and_frame_count(self):
        visualizer = Visualizer(on_timing=Mock())
        visualizer(self.result)
        visualizer.reset()
        self.assertIsNone(visualizer.last_timing)
        visualizer(self.result)
        self.assertEqual(visualizer.last_timing.frame_index, 1)

    def test_invalid_callback_is_rejected_early(self):
        with self.assertRaisesRegex(ValueError, "on_timing"):
            Visualizer(on_timing="callback")


if __name__ == "__main__":
    unittest.main()
