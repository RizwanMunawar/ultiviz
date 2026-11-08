"""Package and CLI behavior without loading models or opening videos."""

import contextlib
import io
import subprocess
import sys
import unittest
from unittest.mock import Mock

from cv_visualizer.cli import main


class PackageTests(unittest.TestCase):
    def test_cli_forwards_options_and_camera_index(self):
        factory = Mock()
        factory.return_value.run.return_value = "output.mp4"
        with contextlib.redirect_stdout(io.StringIO()):
            main(["--model", "pose.pt", "--source", "0", "--trails", "--chart",
                  "--headless", "--hide-boxes", "--classes", "0", "2"],
                 visualizer_class=factory)
        options = factory.call_args.kwargs
        self.assertTrue(options["tracking"])
        self.assertTrue(options["show_trails"])
        self.assertTrue(options["show_chart"])
        self.assertFalse(options["show_boxes"])
        self.assertEqual(options["class_ids"], [0, 2])
        factory.return_value.run.assert_called_once_with(
            0, "output-visualization.mp4", show_window=False,
        )

    def test_cli_requires_model_and_source_before_inference(self):
        factory = Mock()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main([], visualizer_class=factory)
        self.assertEqual(error.exception.code, 2)
        factory.assert_not_called()

    def test_base_import_does_not_load_ultralytics(self):
        subprocess.run(
            [sys.executable, "-c", "import sys; from cv_visualizer import Visualizer; "
             "assert 'ultralytics' not in sys.modules; Visualizer()"],
            check=True, capture_output=True, text=True,
        )


if __name__ == "__main__":
    unittest.main()
