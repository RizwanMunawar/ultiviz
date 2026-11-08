"""Automatic video saving, bounded streaming, metadata, and cleanup."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from ultraviz import Visualizer


class SaveStreamTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="ultiviz-stream-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.frame = np.full((48, 64, 3), 100, np.uint8)
        self.output = self.directory / "output.mp4"

    def result(self, **kwargs):
        return SimpleNamespace(orig_img=self.frame, names={0: "object"}, **kwargs)

    def test_default_call_saves_playable_video_with_source_fps(self):
        source = self.directory / "source.mp4"
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 12, (64, 48))
        self.assertTrue(writer.isOpened())
        for _ in range(4):
            writer.write(self.frame)
        writer.release()
        closed = []
        def results():
            try:
                for _ in range(4):
                    yield self.result(path=str(source))
            finally:
                closed.append(True)
        previous = Path.cwd()
        try:
            os.chdir(self.directory)
            output = Visualizer(results(), show_chart=True)
            self.assertEqual(output, Path("annotated.mp4"))
            capture = cv2.VideoCapture(str(output))
            try:
                self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 12, places=1)
                count = 0
                while True:
                    success, frame = capture.read()
                    if not success:
                        break
                    self.assertEqual(frame.shape, self.frame.shape)
                    self.assertTrue(np.any(frame != 100))
                    count += 1
                self.assertEqual(count, 4)
            finally:
                capture.release()
        finally:
            os.chdir(previous)
        self.assertEqual(closed, [True])
        np.testing.assert_array_equal(self.frame, 100)
        source.unlink()  # Handles must be closed on Windows, too.
        (self.directory / "annotated.mp4").unlink()

    def test_explicit_output_and_fps_with_incremental_consumption(self):
        writer = Mock()
        writer.isOpened.return_value = True
        def results():
            yield self.result()
            self.assertEqual(writer.write.call_count, 1)
            yield self.result()
            self.assertEqual(writer.write.call_count, 2)
        output = self.directory / "nested folder" / "custom.mp4"
        with patch("ultiviz.video.cv2.VideoWriter", return_value=writer) as factory:
            actual = Visualizer(results(), output=output, fps=24)
        self.assertEqual(actual, output)
        self.assertEqual(factory.call_args.args[2:], (24, (64, 48)))
        self.assertTrue(output.parent.is_dir())
        writer.release.assert_called_once()

    def test_video_prediction_lists_are_saved_without_a_frame_loop(self):
        writer = Mock()
        writer.isOpened.return_value = True
        for count in (1, 3):
            writer.reset_mock()
            results = [self.result(path="source.mp4") for _ in range(count)]
            with patch("ultiviz.video.cv2.VideoWriter", return_value=writer):
                output = Visualizer(results, output=self.output)
            self.assertEqual(output, self.output)
            self.assertEqual(writer.write.call_count, count)
            writer.release.assert_called_once()

    def test_video_batch_can_opt_out_of_saving(self):
        with patch("ultiviz.video.cv2.VideoWriter") as writer:
            outputs = Visualizer([self.result(path="source.mp4")], save=False)
            self.assertIs(iter(outputs), outputs)
            np.testing.assert_array_equal(next(outputs), self.frame)
            with self.assertRaises(StopIteration):
                next(outputs)
        writer.assert_not_called()

    def test_unknown_source_fps_defaults_to_30(self):
        writer = Mock()
        writer.isOpened.return_value = True
        with patch("ultiviz.video.cv2.VideoWriter", return_value=writer) as factory:
            Visualizer(iter([self.result(path="camera")]), output=self.output)
        self.assertEqual(factory.call_args.args[2], 30)

    def test_invalid_source_fps_releases_metadata_capture(self):
        source = self.directory / "source.mp4"
        source.touch()
        for fps in (0, float("nan"), float("inf")):
            capture = Mock()
            capture.get.return_value = fps
            with patch("ultiviz.video.cv2.VideoCapture", return_value=capture):
                self.assertEqual(Visualizer._source_fps(source), 30)
            capture.release.assert_called_once()

    def test_empty_stream_does_not_open_writer(self):
        with patch("ultiviz.video.cv2.VideoWriter") as factory:
            with self.assertRaisesRegex(ValueError, "No prediction results"):
                Visualizer(iter([]), output=self.output)
        factory.assert_not_called()

    def test_source_output_collision_does_not_open_writer(self):
        with patch("ultiviz.video.cv2.VideoWriter") as factory:
            with self.assertRaisesRegex(ValueError, "Output must differ"):
                Visualizer(iter([self.result(path=str(self.output))]), output=self.output)
        factory.assert_not_called()

    def test_invalid_output_settings_close_iterator(self):
        for options in ({"fps": 0}, {"fps": float("nan")}, {"output": "output.avi"}):
            iterator = Mock(__iter__=lambda self: self, __next__=Mock(side_effect=StopIteration))
            with self.assertRaises(ValueError):
                Visualizer(iterator, **options)
            iterator.close.assert_called_once()

    def test_failures_release_writer_and_prediction_generator(self):
        for failure in ("writer", "inference", "dimensions", "render"):
            closed = []
            writer = Mock()
            writer.isOpened.return_value = failure != "writer"
            def results():
                try:
                    yield self.result()
                    if failure == "inference":
                        raise RuntimeError("inference failed")
                    if failure == "dimensions":
                        yield SimpleNamespace(orig_img=self.frame[:24], names={})
                    if failure == "render":
                        yield SimpleNamespace(orig_img=None, names={})
                finally:
                    closed.append(True)
            with self.subTest(failure=failure), \
                    patch("ultiviz.video.cv2.VideoWriter", return_value=writer):
                with self.assertRaises((ValueError, RuntimeError)):
                    Visualizer(results(), output=self.output)
            writer.release.assert_called_once()
            self.assertEqual(closed, [True])


if __name__ == "__main__":
    unittest.main()
