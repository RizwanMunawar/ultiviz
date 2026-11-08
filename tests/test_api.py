"""Simple API, invalid input, and optimized rendering regressions."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from ultiviz import Visualizer


class SimpleApiTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((48, 64, 3), 100, dtype=np.uint8)
        self.result = SimpleNamespace(orig_img=self.frame, names={0: "object"})

    def test_result_only_and_callable_match_explicit_frame(self):
        visualizer = Visualizer(show_chart=True)
        expected = visualizer.annotate(self.frame, self.result)
        np.testing.assert_array_equal(visualizer.annotate(self.result), expected)
        np.testing.assert_array_equal(visualizer(self.result), expected)
        np.testing.assert_array_equal(self.frame, 100)
        self.assertIsNot(visualizer(self.result), self.frame)

    def test_explicit_in_place_drawing_returns_same_frame(self):
        output = Visualizer(show_chart=True)(self.result, copy=False)
        self.assertIs(output, self.frame)
        self.assertTrue(np.any(self.frame != 100))

    def test_in_place_requires_writable_contiguous_frame(self):
        visualizer = Visualizer()
        frames = [self.frame[:, ::2], self.frame.copy()]
        frames[1].flags.writeable = False
        for frame in frames:
            with self.subTest(strides=frame.strides, writable=frame.flags.writeable):
                with self.assertRaisesRegex(ValueError, "writable C-contiguous"):
                    visualizer.annotate(frame, self.result, copy=False)
                self.assertTrue(visualizer.annotate(frame, self.result).flags.writeable)

    def test_invalid_frames_fail_before_inference(self):
        model = Mock(task="detect")
        for frame in (None, [], np.zeros((0, 2, 3), np.uint8),
                      np.zeros((2, 0, 3), np.uint8), np.zeros((2, 2)),
                      np.zeros((2, 2, 4), np.uint8)):
            with self.subTest(shape=getattr(frame, "shape", None)):
                with self.assertRaisesRegex(ValueError, "nonempty uint8 BGR"):
                    Visualizer(model).process_frame(frame)
        model.predict.assert_not_called()

    def test_trails_enable_tracking_and_process_frame_can_draw_in_place(self):
        model = Mock(task="detect")
        model.track.return_value = [self.result]
        visualizer = Visualizer(model, show_trails=True)
        self.assertTrue(visualizer.tracking)
        self.assertIs(visualizer.process_frame(self.frame, copy=False), self.frame)
        model.track.assert_called_once()
        model.predict.assert_not_called()

    def test_mismatched_result_resolution_is_rejected_without_drawing(self):
        self.result.orig_shape = (24, 32)
        with self.assertRaisesRegex(ValueError, "dimensions must match"):
            Visualizer(show_chart=True).annotate(self.result, copy=False)
        np.testing.assert_array_equal(self.frame, 100)

    def test_classification_renders_top_five_from_bulk_probabilities(self):
        self.result.probs = SimpleNamespace(data=np.array([.1, .8, .05, .05]))
        visualizer = Visualizer()
        visualizer.draw_label = Mock()
        visualizer(self.result)
        self.assertEqual(visualizer.draw_label.call_count, 4)
        self.assertEqual(visualizer.draw_label.call_args_list[0].args[1], "1 0.80")

    def test_video_resolution_change_releases_capture_and_writer(self):
        cap = Mock()
        cap.get.return_value = 30
        cap.read.side_effect = [(True, self.frame), (True, self.frame[:24])]
        writer = Mock()
        writer.isOpened.return_value = True
        visualizer = Visualizer(Mock(task="detect", ckpt_path=None, predictor=None))
        visualizer.process_frame = Mock(side_effect=lambda frame: frame)
        with TemporaryDirectory() as directory, \
                patch("ultiviz.video.cv2.VideoCapture", return_value=cap), \
                patch("ultiviz.video.cv2.VideoWriter", return_value=writer):
            with self.assertRaisesRegex(ValueError, "dimensions changed"):
                visualizer.run(0, Path(directory) / "out.mp4", show_window=False)
        writer.write.assert_called_once()
        writer.release.assert_called_once()
        cap.release.assert_called_once()

    def test_invalid_drawing_and_inference_sizes_fail_at_construction(self):
        for options in ({"line_width": 1.5}, {"trail_length": 2.5},
                        {"padding": True}, {"boundary_thickness": 0.5},
                        {"image_size": 0}, {"image_size": -1},
                        {"font_scale": float("nan")}, {"font_scale": float("inf")}):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    Visualizer(**options)

    def test_mask_colors_and_shapes_are_validated_before_modifying_frame(self):
        visualizer = Visualizer()
        for masks, colors in (([np.ones((48, 64))], []),
                              ([np.ones((48, 64)), np.ones((1, 1))], [(0, 0, 0)] * 2)):
            with self.assertRaises(ValueError):
                visualizer.draw_masks(self.frame, masks, colors)
            np.testing.assert_array_equal(self.frame, 100)
        self.assertIs(visualizer.draw_masks(self.frame, [], []), self.frame)

    def test_cropped_mask_rendering_matches_full_frame_reference(self):
        rng = np.random.default_rng(4)
        masks = np.zeros((4, 48, 64), np.float32)
        masks[0, :35, :40] = 1
        masks[1, 15:, 20:] = 1
        masks[2, 20:30, 25:35] = 1
        masks[:, 24:27, 29:32] = 0  # overlapping masks with a shared hole
        colors = [(200, 0, 0), (0, 200, 0), (0, 0, 200), (200, 200, 0)]
        for opacity in (0, .45, 1):
            for thickness in (0, 1, 3):
                with self.subTest(opacity=opacity, thickness=thickness):
                    frame = rng.integers(0, 256, self.frame.shape, dtype=np.uint8)
                    overlay, expected = frame.copy(), frame.copy()
                    owner = np.full(frame.shape[:2], -1)
                    for index, (mask, color) in enumerate(zip(masks, colors)):
                        active = mask > .5
                        overlay[active], owner[active] = color, index
                    cv2.addWeighted(overlay, opacity, expected, 1 - opacity, 0, dst=expected)
                    if thickness:
                        for index, (mask, color) in enumerate(zip(masks, colors)):
                            binary = (mask > .5).astype(np.uint8)
                            eroded = cv2.erode(binary, np.ones((3, 3), np.uint8), iterations=thickness,
                                               borderType=cv2.BORDER_CONSTANT, borderValue=0)
                            expected[(binary > eroded) & (owner == index)] = color
                    actual = Visualizer(mask_opacity=opacity, boundary_thickness=thickness).draw_masks(
                        frame.copy(), masks, colors)
                    np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
