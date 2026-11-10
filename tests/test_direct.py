"""Direct YOLO prediction rendering, imports, streaming, and image saving."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from ultiviz import Visualizer


class DirectRenderingTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((48, 64, 3), 100, dtype=np.uint8)
        self.result = SimpleNamespace(orig_img=self.frame, names={0: "object"})

    def test_result_and_singleton_predictions_return_image(self):
        expected = Visualizer(show_chart=True)(self.result)
        for results in (self.result, [self.result], (self.result,)):
            with self.subTest(kind=type(results)):
                output = Visualizer(results, save=False, show=False, show_chart=True)
                self.assertIsInstance(output, np.ndarray)
                np.testing.assert_array_equal(output, expected)
                self.assertIsNot(output, self.frame)
        np.testing.assert_array_equal(self.frame, 100)

    def test_multiple_predictions_keep_order_and_resolution(self):
        second = SimpleNamespace(orig_img=np.zeros((24, 32, 3), np.uint8), names={})
        outputs = Visualizer([self.result, second], save=False, show=False)
        self.assertIsInstance(outputs, list)
        self.assertEqual([image.shape for image in outputs], [(48, 64, 3), (24, 32, 3)])
        np.testing.assert_array_equal(outputs[0], 100)
        np.testing.assert_array_equal(outputs[1], 0)

    def test_empty_prediction_lists_raise_clear_error(self):
        for results in ([], ()):
            with self.assertRaisesRegex(ValueError, "No prediction results"):
                Visualizer(results)

    def test_direct_in_place_option_and_reusable_list_api(self):
        self.assertIs(Visualizer([self.result], copy=False, save=False, show=False), self.frame)
        renderer = Visualizer()
        self.assertIs(renderer([self.result], copy=False), self.frame)
        self.assertEqual(len(renderer([self.result, self.result])), 2)

    def test_video_stream_is_lazy_and_closes_upstream_on_early_stop(self):
        events = []
        def predictions():
            try:
                for i in range(3):
                    events.append(i)
                    yield self.result
            finally:
                events.append("closed")
        outputs = Visualizer(predictions(), save=False, show=False)
        self.assertEqual(events, [])
        np.testing.assert_array_equal(next(outputs), self.frame)
        self.assertEqual(events, [0])
        outputs.close()
        self.assertEqual(events, [0, "closed"])

    def test_stream_render_failure_closes_upstream(self):
        closed = []
        def predictions():
            try:
                yield SimpleNamespace(orig_img=None, names={})
            finally:
                closed.append(True)
        outputs = Visualizer(predictions(), save=False, show=False)
        with self.assertRaisesRegex(ValueError, "nonempty uint8 BGR"):
            next(outputs)
        self.assertEqual(closed, [True])

    def test_empty_and_single_result_iterators_always_return_iterators(self):
        self.assertEqual(list(Visualizer(iter([]), save=False, show=False)), [])
        outputs = Visualizer(iter([self.result]), save=False, show=False)
        self.assertIs(iter(outputs), outputs)
        np.testing.assert_array_equal(next(outputs), self.frame)
        with self.assertRaises(StopIteration):
            next(outputs)

    def test_trails_persist_across_direct_stream_frames(self):
        class Boxes:
            cls = np.array([0])
            conf = np.array([.9])
            id = np.array([1])
            def __init__(self, x):
                self.xyxy = np.array([[x, 10, x + 10, 30]])
            def __len__(self):
                return 1
        results = (SimpleNamespace(orig_img=self.frame, names={0: "object"}, boxes=Boxes(x))
                   for x in (5, 25))
        first, second = Visualizer(results, save=False, show=False, show_trails=True, show_boxes=False, show_labels=False)
        np.testing.assert_array_equal(first, self.frame)
        self.assertTrue(np.any(second[20, 20] != self.frame[20, 20]))

    def test_model_arguments_still_construct_reusable_visualizer(self):
        for model in ("weights.pt", Path("weights.pt"), Mock(task="detect")):
            renderer = Visualizer(model)
            self.assertIsInstance(renderer, Visualizer)
            self.assertIs(renderer.model, model)

    def test_direct_output_can_be_saved_by_opencv(self):
        output = Visualizer([self.result], save=False, show=False, show_chart=True)
        with TemporaryDirectory() as directory:
            filename = str(Path(directory) / "annotated.png")
            self.assertTrue(cv2.imwrite(filename, output))
            np.testing.assert_array_equal(cv2.imread(filename), output)


if __name__ == "__main__":
    unittest.main()
