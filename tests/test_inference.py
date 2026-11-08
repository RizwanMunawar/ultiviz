"""Inference dispatch without downloading weights or importing Ultralytics."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from ultiviz import Visualizer


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((48, 64, 3), dtype=np.uint8)
        self.result = SimpleNamespace(names={0: "object"})

    def model(self, task):
        model = Mock(task=task)
        model.predict.return_value = [self.result]
        model.track.return_value = [self.result]
        return model

    def test_detection_forwards_inference_settings(self):
        model = self.model("detect")
        visualizer = Visualizer(model, confidence=.35, iou=.6, image_size=640,
                                device="cpu", class_ids={2, 0})
        annotated = visualizer.process_frame(self.frame)
        self.assertEqual(annotated.shape, self.frame.shape)
        self.assertIsNot(annotated, self.frame)
        model.predict.assert_called_once_with(
            self.frame, imgsz=640, verbose=False, device="cpu", conf=.35,
            iou=.6, classes=[0, 2],
        )
        model.track.assert_not_called()

    def test_detection_tasks_share_settings(self):
        for task in ("detect", "segment", "pose", "obb"):
            with self.subTest(task=task):
                model = self.model(task)
                Visualizer(model).process_frame(self.frame)
                kwargs = model.predict.call_args.kwargs
                self.assertEqual(kwargs["conf"], .25)
                self.assertEqual(kwargs["iou"], .45)
                self.assertIsNone(kwargs["classes"])
                self.assertNotIn("device", kwargs)
                self.assertEqual(kwargs.get("retina_masks", False), task == "segment")

    def test_tracking_persists_between_frames(self):
        for task in ("detect", "segment", "pose", "obb"):
            with self.subTest(task=task):
                model = self.model(task)
                visualizer = Visualizer(model, tracking=True, tracker="botsort.yaml")
                visualizer.process_frame(self.frame)
                visualizer.process_frame(self.frame)
                self.assertEqual(model.track.call_count, 2)
                for call in model.track.call_args_list:
                    self.assertTrue(call.kwargs["persist"])
                    self.assertEqual(call.kwargs["tracker"], "botsort.yaml")
                model.predict.assert_not_called()

    def test_classification_omits_detection_arguments(self):
        model = self.model("classify")
        Visualizer(model, device="cpu").process_frame(self.frame)
        model.predict.assert_called_once_with(
            self.frame, imgsz=1280, verbose=False, device="cpu",
        )

    def test_unsupported_tracking_fails_before_model_call(self):
        model = self.model("classify")
        with self.assertRaisesRegex(ValueError, "Tracking is unsupported"):
            Visualizer(model, tracking=True).process_frame(self.frame)
        model.predict.assert_not_called()
        model.track.assert_not_called()

    def test_dense_prediction_tasks_omit_detection_only_options(self):
        for task in ("semantic", "depth"):
            with self.subTest(task=task):
                model = self.model(task)
                Visualizer(model, timing=False).process_frame(self.frame)
                model.predict.assert_called_once_with(self.frame, imgsz=1280, verbose=False)
                model.track.assert_not_called()

    def test_inference_requires_model(self):
        with self.assertRaisesRegex(ValueError, "Pass model weights"):
            Visualizer().process_frame(self.frame)


if __name__ == "__main__":
    unittest.main()
