"""Portable video lifecycle checks and a real, headless MP4 round trip."""

from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from ultiviz import Visualizer


class VideoTests(unittest.TestCase):
    def setUp(self):
        self.contexts = ExitStack()
        self.addCleanup(self.contexts.close)
        self.temp = TemporaryDirectory(prefix="ultiviz-tests-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.frame = np.full((48, 64, 3), 100, dtype=np.uint8)
        self.model = SimpleNamespace(task="detect")
        self.visualizer = Visualizer(self.model)
        self.visualizer.process_frame = Mock(side_effect=lambda frame: frame.copy())
        self.cap = Mock()
        self.cap.read.side_effect = [(True, self.frame), (False, None)]
        self.cap.get.return_value = 24.0
        self.writer = Mock()
        self.writer.isOpened.return_value = True
        self.capture_factory = self.contexts.enter_context(
            patch("ultiviz.video.cv2.VideoCapture", return_value=self.cap)
        )
        self.writer_factory = self.contexts.enter_context(
            patch("ultiviz.video.cv2.VideoWriter", return_value=self.writer)
        )
        self.imshow = self.contexts.enter_context(patch("ultiviz.video.cv2.imshow"))
        self.destroy_windows = self.contexts.enter_context(patch("ultiviz.video.cv2.destroyAllWindows"))

    def test_camera_index_and_output_path_with_spaces(self):
        output = self.directory / "nested folder" / "result video.mp4"
        saved = self.visualizer.run(0, output, show_window=False)
        self.assertEqual(saved, output)
        self.assertTrue(output.parent.is_dir())
        self.capture_factory.assert_called_once_with(0)
        self.writer_factory.assert_called_once_with(
            str(output), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (64, 48),
        )
        np.testing.assert_array_equal(self.writer.write.call_args.args[0], self.frame)
        self.cap.release.assert_called_once()
        self.writer.release.assert_called_once()
        self.imshow.assert_not_called()
        self.destroy_windows.assert_not_called()

    def test_path_source_and_disabled_saving(self):
        source = self.directory / "input video.mp4"
        self.assertIsNone(self.visualizer.run(source, None, show_window=False))
        self.capture_factory.assert_called_once_with(str(source))
        self.writer_factory.assert_not_called()
        self.cap.release.assert_called_once()

    def test_invalid_fps_falls_back_to_thirty(self):
        for fps in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(fps=fps):
                self.cap.get.return_value = fps
                self.cap.read.side_effect = [(True, self.frame), (False, None)]
                self.visualizer.run(0, self.directory / "result.mp4", show_window=False)
                self.assertEqual(self.writer_factory.call_args.args[2], 30.0)

    def test_unreadable_source_releases_capture_without_opening_output(self):
        self.cap.read.side_effect = [(False, None)]
        with self.assertRaisesRegex(RuntimeError, "Cannot read video source"):
            self.visualizer.run("missing.mp4", self.directory / "out.mp4", show_window=False)
        self.cap.release.assert_called_once()
        self.writer_factory.assert_not_called()

    def test_writer_failure_releases_resources(self):
        self.writer.isOpened.return_value = False
        with self.assertRaisesRegex(RuntimeError, "Cannot create output video"):
            self.visualizer.run(0, self.directory / "out.mp4", show_window=False)
        self.cap.release.assert_called_once()
        self.writer.release.assert_called_once()

    def test_inference_failure_releases_resources(self):
        self.visualizer.process_frame.side_effect = RuntimeError("inference failed")
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            self.visualizer.run(0, self.directory / "out.mp4", show_window=False)
        self.cap.release.assert_called_once()
        self.writer.release.assert_called_once()

    def test_run_resets_visualization_and_existing_model_trackers(self):
        tracker = Mock()
        self.model.predictor = SimpleNamespace(trackers=[tracker])
        self.visualizer._draw_trail(self.frame.copy(), 7, (20, 20), (255, 0, 0))
        self.visualizer.run(0, None, show_window=False)
        tracker.reset.assert_called_once()
        self.assertFalse(self.visualizer._history)
        self.assertEqual(self.visualizer._frame_index, 0)

    def test_equivalent_source_and_output_paths_are_rejected(self):
        source = self.directory / "source.mp4"
        equivalent = self.directory / "folder" / ".." / "source.mp4"
        with self.assertRaisesRegex(ValueError, "Output must differ"):
            self.visualizer.run(source, equivalent, show_window=False)
        self.capture_factory.assert_not_called()
        self.writer_factory.assert_not_called()

    def test_preview_quit_releases_resources(self):
        with patch("ultiviz.video.cv2.waitKey", return_value=ord("q")):
            self.visualizer.run(0, None, show_window=True)
        self.imshow.assert_called_once()
        self.cap.release.assert_called_once()
        self.destroy_windows.assert_called_once()


class VideoRoundTripTests(unittest.TestCase):
    def test_mp4_with_spaces_in_paths_round_trips_without_gui(self):
        frame = np.full((48, 64, 3), 100, dtype=np.uint8)
        result = SimpleNamespace(names={0: "object"})
        model = Mock(task="detect", ckpt_path=None, predictor=None)
        model.predict.return_value = [result]
        with TemporaryDirectory(prefix="ultiviz video ") as directory:
            source = Path(directory) / "input video.mp4"
            output = Path(directory) / "output folder" / "output video.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
            try:
                self.assertTrue(writer.isOpened(), "MP4 encoder must be available")
                for _ in range(3):
                    writer.write(frame)
            finally:
                writer.release()
            with patch("ultiviz.video.cv2.imshow") as imshow:
                saved = Visualizer(model).run(source, output, show_window=False)
                imshow.assert_not_called()
            self.assertEqual(saved, output)
            self.assertEqual(model.predict.call_count, 3)
            cap = cv2.VideoCapture(str(output))
            count = 0
            try:
                self.assertTrue(cap.isOpened())
                while True:
                    success, decoded = cap.read()
                    if not success:
                        break
                    self.assertEqual(decoded.shape, frame.shape)
                    count += 1
            finally:
                cap.release()
            self.assertEqual(count, 3)
            # Windows forbids removing open video files; this also checks cleanup.
            source.unlink()
            output.unlink()


if __name__ == "__main__":
    unittest.main()
