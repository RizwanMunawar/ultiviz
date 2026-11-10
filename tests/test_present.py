"""The default direct call saves, displays, and returns image data immediately."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from ultiviz import Visualizer


class PresentTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="ultiviz-present-")
        self.addCleanup(self.temp.cleanup)
        previous = Path.cwd()
        os.chdir(self.temp.name)
        self.addCleanup(os.chdir, previous)
        self.frame = np.full((48, 64, 3), 100, np.uint8)
        preview_patch = patch.object(Visualizer, "_preview", return_value=False)
        close_patch = patch.object(Visualizer, "_close_preview")
        self.preview = preview_patch.start()
        self.addCleanup(preview_patch.stop)
        self.close = close_patch.start()
        self.addCleanup(close_patch.stop)

    def result(self, value=100, path="image.jpg"):
        return SimpleNamespace(orig_img=np.full_like(self.frame, value), names={0: "object"}, path=path)

    def decode(self, path):
        capture = cv2.VideoCapture(str(path))
        frames = []
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    return frames
                frames.append(frame)
        finally:
            capture.release()

    def test_image_default_saves_displays_and_returns_array(self):
        result = self.result()
        output = Visualizer([result], show_chart=True)
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(cv2.imread("annotated.jpg").shape, output.shape)
        self.assertTrue(np.any(output != 100))
        np.testing.assert_array_equal(result.orig_img, 100)
        self.preview.assert_called_once()
        self.assertEqual(self.preview.call_args.args[1], 0)
        self.close.assert_called_once()

    def test_default_stream_is_consumed_saves_displays_and_returns_last_frame(self):
        closed = []
        def results():
            try:
                for value in (100, 120, 140):
                    yield self.result(value, "input.mp4")
            finally:
                closed.append(True)
        output = Visualizer(results())
        np.testing.assert_array_equal(output, 140)
        self.assertEqual(len(self.decode("annotated.mp4")), 3)
        self.assertEqual(self.preview.call_count, 3)
        self.close.assert_called_once()
        self.assertEqual(closed, [True])
        Path("annotated.mp4").unlink()

    def test_streamed_image_is_saved_as_an_image_and_returns_array(self):
        closed = []
        def results():
            try:
                yield self.result(100, "input.jpg")
            finally:
                closed.append(True)
        output = Visualizer(results())
        np.testing.assert_array_equal(output, 100)
        self.assertTrue(Path("annotated.jpg").is_file())
        self.assertFalse(Path("annotated.mp4").exists())
        self.preview.assert_called_once()
        self.assertEqual(self.preview.call_args.args[1], 0)
        self.assertEqual(closed, [True])

    def test_preview_failure_releases_writer_stream_and_window(self):
        self.preview.side_effect = RuntimeError("preview failed")
        closed = []
        def results():
            try:
                yield self.result(path="input.mp4")
                self.fail("Stream must stop when preview fails")
            finally:
                closed.append(True)
        writer = Mock()
        writer.isOpened.return_value = True
        with patch("ultiviz.video.cv2.VideoWriter", return_value=writer):
            with self.assertRaisesRegex(RuntimeError, "preview failed"):
                Visualizer(results())
        writer.release.assert_called_once()
        self.close.assert_called_once()
        self.assertEqual(closed, [True])

    def test_nonstreamed_video_has_same_save_display_and_return_behavior(self):
        output = Visualizer([self.result(100, "input.mp4"), self.result(120, "input.mp4")])
        np.testing.assert_array_equal(output, 120)
        self.assertEqual(len(self.decode("annotated.mp4")), 2)
        self.assertEqual(self.preview.call_count, 2)

    def test_image_batch_saves_separate_files_and_returns_frames(self):
        outputs = Visualizer([self.result(100), self.result(120)], output="custom.png")
        self.assertEqual(len(outputs), 2)
        np.testing.assert_array_equal(cv2.imread("custom.png"), outputs[0])
        np.testing.assert_array_equal(cv2.imread("custom_2.png"), outputs[1])
        self.assertEqual(self.preview.call_count, 2)

    def test_headless_export_skips_all_gui_calls(self):
        output = Visualizer(iter([self.result(path="input.mp4")]), show=False)
        np.testing.assert_array_equal(output, 100)
        self.assertTrue(Path("annotated.mp4").is_file())
        self.preview.assert_not_called()
        self.close.assert_not_called()

    def test_display_without_saving_still_consumes_stream(self):
        with patch("ultiviz.video.cv2.VideoWriter") as writer:
            output = Visualizer(iter([self.result(100, "input.mp4"), self.result(120, "input.mp4")]), save=False)
        writer.assert_not_called()
        np.testing.assert_array_equal(output, 120)
        self.assertEqual(self.preview.call_count, 2)

    def test_quit_stops_capture_and_finalizes_playable_partial_video(self):
        self.preview.side_effect = [False, True]
        closed = []
        def results():
            try:
                for value in (100, 120, 140, 160):
                    yield self.result(value, "input.mp4")
            finally:
                closed.append(True)
        output = Visualizer(results())
        np.testing.assert_array_equal(output, 120)
        self.assertEqual(len(self.decode("annotated.mp4")), 2)
        self.assertEqual(closed, [True])
        self.close.assert_called_once()

    def test_inference_failure_releases_writer_stream_and_preview(self):
        closed = []
        def results():
            try:
                yield self.result(path="input.mp4")
                raise RuntimeError("inference failed")
            finally:
                closed.append(True)
        writer = Mock()
        writer.isOpened.return_value = True
        with patch("ultiviz.video.cv2.VideoWriter", return_value=writer):
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                Visualizer(results())
        writer.release.assert_called_once()
        self.close.assert_called_once()
        self.assertEqual(closed, [True])

    def test_image_write_failure_is_reported_and_preview_is_cleaned_up(self):
        with patch("ultiviz.video.cv2.imwrite", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "Cannot save output image"):
                Visualizer(self.result())
        self.preview.assert_not_called()
        self.close.assert_called_once()

    def test_image_source_is_protected_from_overwrite(self):
        with patch("ultiviz.video.cv2.imwrite") as write:
            with self.assertRaisesRegex(ValueError, "Output must differ"):
                Visualizer(self.result(path="annotated.jpg"))
        write.assert_not_called()


class PreviewTests(unittest.TestCase):
    def test_preview_scales_frame_and_honors_quit(self):
        with patch.dict(os.environ, {"DISPLAY": ":test"}), \
                patch("ultiviz.video.cv2.imshow") as show, \
                patch("ultiviz.video.cv2.waitKey", return_value=ord("q")) as wait:
            stopped = Visualizer._preview(np.zeros((720, 1920, 3), np.uint8), 33)
        self.assertTrue(stopped)
        self.assertEqual(show.call_args.args[1].shape, (480, 1280, 3))
        wait.assert_called_once_with(33)

    def test_missing_linux_display_reports_headless_option_without_opening_gui(self):
        with patch("ultiviz.video.sys.platform", "linux"), \
                patch.dict(os.environ, {}, clear=True), \
                patch("ultiviz.video.cv2.imshow") as show:
            with self.assertRaisesRegex(RuntimeError, "show=False"):
                Visualizer._preview(np.zeros((48, 64, 3), np.uint8), 1)
        show.assert_not_called()

    def test_gui_failure_reports_headless_option(self):
        with patch.dict(os.environ, {"DISPLAY": ":test"}), \
                patch("ultiviz.video.cv2.imshow", side_effect=cv2.error("no GUI")):
            with self.assertRaisesRegex(RuntimeError, "show=False"):
                Visualizer._preview(np.zeros((48, 64, 3), np.uint8), 1)


if __name__ == "__main__":
    unittest.main()
