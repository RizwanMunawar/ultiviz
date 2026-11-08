"""Rendering regression checks using real Ultralytics Results, without weights."""

import unittest

import numpy as np
from types import SimpleNamespace

try:
    from ultralytics.engine.results import Results
except ModuleNotFoundError as exc:
    if exc.name != "ultralytics":
        raise
    Results = None

from ultiviz import Visualizer


@unittest.skipIf(Results is None, "Install ultiviz[ultralytics] for integration tests")
class VisualizerTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((96, 160, 3), 100, dtype=np.uint8)
        self.names = {0: "object", 1: "other"}

    def result(self, **kwargs):
        return Results(self.frame, "test.jpg", self.names, **kwargs)

    def test_direct_rendering_accepts_real_prediction_types(self):
        boxes = np.array([[10, 10, 100, 90, .9, 0]], dtype=np.float32)
        masks = np.zeros((1, 96, 160), dtype=np.float32)
        masks[:, 20:80, 20:90] = 1
        points = np.array([[[30, 30, .9], [70, 60, .9]]], dtype=np.float32)
        obb = np.array([[80, 48, 50, 30, .5, .9, 0]], dtype=np.float32)
        predictions = [self.result(boxes=boxes), self.result(boxes=boxes, masks=masks),
                       self.result(boxes=boxes, keypoints=points), self.result(obb=obb),
                       self.result(probs=np.array([.9, .1], dtype=np.float32))]
        for result in predictions:
            with self.subTest(result=result):
                expected = Visualizer().annotate(result)
                np.testing.assert_array_equal(Visualizer([result]), expected)
                self.assertTrue(np.any(expected != self.frame))
        np.testing.assert_array_equal(self.frame, 100)

    def test_detection_filtering_does_not_modify_input(self):
        boxes = np.array([[10, 10, 50, 70, 9, .9, 0],
                          [80, 10, 140, 70, 10, .8, 1]], dtype=np.float32)
        visualizer = Visualizer(hidden_track_ids={9}, show_labels=False)
        output = visualizer.annotate(self.frame, self.result(boxes=boxes))
        np.testing.assert_array_equal(self.frame, 100)
        np.testing.assert_array_equal(output[:, :60], self.frame[:, :60])
        self.assertTrue(np.any(output[:, 70:] != self.frame[:, 70:]))

    def test_masks_blend_once_preserve_holes_and_outside(self):
        masks = np.zeros((2, 96, 160), dtype=np.uint8)
        masks[0, 10:70, 10:90] = 1
        masks[1, 30:80, 40:120] = 1
        masks[:, 45:50, 55:60] = 0
        visualizer = Visualizer(mask_opacity=.5, boundary_thickness=0)
        output = visualizer.draw_masks(self.frame.copy(), masks, [(200, 0, 0), (0, 200, 0)])
        np.testing.assert_array_equal(output[35, 45], [50, 150, 50])
        np.testing.assert_array_equal(output[20, 20], [150, 50, 50])
        np.testing.assert_array_equal(output[46, 56], [100, 100, 100])
        np.testing.assert_array_equal(output[0, 0], [100, 100, 100])

    def test_segmentation_filter_applies_to_masks(self):
        masks = np.ones((1, 96, 160), dtype=np.float32)
        boxes = np.array([[10, 10, 50, 70, .9, 0]], dtype=np.float32)
        visualizer = Visualizer(hidden_classes={"object"})
        output = visualizer.annotate(self.frame, self.result(boxes=boxes, masks=masks))
        np.testing.assert_array_equal(output, self.frame)

    def test_semantic_ignore_and_nearest_resize(self):
        class_map = np.full((48, 80), 255, dtype=np.int32)
        class_map[:, 40:] = 1
        visualizer = Visualizer(boundary_thickness=0, mask_opacity=1,
                                class_colors={1: (0, 200, 0)})
        output = visualizer.annotate(self.frame, SimpleNamespace(names=self.names, semantic_mask=SimpleNamespace(data=class_map)))
        np.testing.assert_array_equal(output[:, :80], self.frame[:, :80])
        np.testing.assert_array_equal(output[40, 120], [0, 200, 0])

    def test_pose_confidence_suppresses_point_and_connected_limb(self):
        visualizer = Visualizer(skeleton=[(0, 1)], keypoint_confidence=.5)
        points = np.array([[20, 30, .9], [80, 30, .1]])
        output = visualizer.draw_pose(self.frame.copy(), points, (0, 255, 0))
        self.assertTrue(np.any(output[30, 20] != self.frame[30, 20]))
        np.testing.assert_array_equal(output[30, 50:90], self.frame[30, 50:90])

    def test_obb_without_tracking_id_is_rendered(self):
        obb = np.array([[80, 48, 50, 30, .5, .9, 0]], dtype=np.float32)
        output = Visualizer().annotate(self.frame, self.result(obb=obb))
        self.assertTrue(np.any(output != self.frame))

    def test_pose_results_dispatch(self):
        boxes = np.array([[10, 10, 100, 90, .9, 0]], dtype=np.float32)
        points = np.array([[[30, 30, .9], [70, 60, .9]]], dtype=np.float32)
        visualizer = Visualizer(show_boxes=False, show_labels=False, skeleton=[(0, 1)])
        output = visualizer.annotate(self.frame, self.result(boxes=boxes, keypoints=points))
        self.assertTrue(np.any(output[30, 30] != self.frame[30, 30]))

    def test_trails_are_bounded_expire_and_reset(self):
        visualizer = Visualizer(show_trails=True, trail_length=2, trail_max_gap=1)
        boxes = np.array([[10, 10, 50, 70, 9, .9, 0]], dtype=np.float32)
        for _ in range(4):
            visualizer.annotate(self.frame, self.result(boxes=boxes))
        self.assertEqual(len(visualizer._history[9]), 2)
        for _ in range(3):
            visualizer.annotate(self.frame, self.result())
        self.assertFalse(visualizer._history)
        visualizer.reset()
        self.assertEqual(visualizer._frame_index, 0)

    def test_empty_result_and_small_frame_chart(self):
        visualizer = Visualizer(show_chart=True)
        for size in (1, 8, 96):
            frame = np.zeros((size, size, 3), dtype=np.uint8)
            result = Results(frame, "empty.jpg", self.names)
            self.assertEqual(visualizer.annotate(frame, result).shape, frame.shape)

    def test_output_protects_weights_after_model_is_loaded(self):
        visualizer = Visualizer("models/person-hg-yolo26n.pt")
        # Simulate lazy loading replacing the path with a model object.
        visualizer.model = object()
        with self.assertRaisesRegex(ValueError, "Output must differ"):
            visualizer.run("input.mp4", "models/person-hg-yolo26n.pt", show_window=False)


if __name__ == "__main__":
    unittest.main()
