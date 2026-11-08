"""Depth rendering with invalid data, known endpoints, and real Results."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from ultiviz import Visualizer


class DepthTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((24, 32, 3), 100, np.uint8)

    def result(self, depth):
        return SimpleNamespace(orig_img=self.frame, depth=SimpleNamespace(data=depth))

    def test_depth_endpoints_have_distinct_colors_and_leave_inputs_intact(self):
        depth = np.full((24, 32), 1, np.float32)
        depth[:, 16:] = 10
        original = depth.copy()
        output = Visualizer(mask_opacity=1, timing=False).annotate(self.result(depth))
        palette = cv2.applyColorMap(np.array([[0, 255]], np.uint8), cv2.COLORMAP_TURBO)
        np.testing.assert_array_equal(output[10, 5], palette[0, 0])
        np.testing.assert_array_equal(output[10, 25], palette[0, 1])
        np.testing.assert_array_equal(depth, original)
        np.testing.assert_array_equal(self.frame, 100)

    def test_invalid_depth_pixels_preserve_original_image(self):
        depth = np.full((24, 32), 2, np.float32)
        depth[0, :5] = [0, -1, np.nan, np.inf, -np.inf]
        output = Visualizer(mask_opacity=1, timing=False).annotate(self.result(depth))
        np.testing.assert_array_equal(output[0, :5], self.frame[0, :5])
        self.assertTrue(np.any(output[10, 10] != self.frame[10, 10]))

    def test_all_invalid_depth_is_an_unchanged_copy(self):
        depth = np.full((24, 32), np.nan, np.float32)
        output = Visualizer(timing=False).annotate(self.result(depth))
        self.assertIsNot(output, self.frame)
        np.testing.assert_array_equal(output, self.frame)

    def test_single_channel_constant_depth_map_renders_without_division_by_zero(self):
        depth = np.full((1, 24, 32), 2, np.float32)
        output = Visualizer(timing=False).annotate(self.result(depth))
        self.assertEqual(output.dtype, np.uint8)
        self.assertTrue(np.any(output != self.frame))
        np.testing.assert_array_equal(output[0, 0], output[-1, -1])

    def test_hidden_depth_avoids_data_transfer(self):
        depth = Mock()
        result = SimpleNamespace(orig_img=self.frame, depth=depth)
        output = Visualizer(show_depth=False, timing=False).annotate(result)
        np.testing.assert_array_equal(output, self.frame)
        depth.cpu.assert_not_called()

    def test_malformed_depth_shapes_raise_clear_error(self):
        for shape in ((1,), (2, 24, 32), (12, 16)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "Depth must be a 2D map"):
                    Visualizer(timing=False).annotate(self.result(np.ones(shape, np.float32)))


if __name__ == "__main__":
    unittest.main()
