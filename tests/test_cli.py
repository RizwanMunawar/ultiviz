"""Package and CLI behavior without loading models or opening videos."""

import contextlib
import io
from importlib.metadata import distribution
import subprocess
import sys
import unittest
from unittest.mock import Mock

from ultiviz import __version__
from ultiviz.cli import main


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

    def test_installed_distribution_version_and_entry_point(self):
        package = distribution("ultiviz")
        self.assertEqual(package.version, __version__)
        entry_points = [entry for entry in package.entry_points
                        if entry.group == "console_scripts" and entry.name == "ultiviz"]
        self.assertEqual(len(entry_points), 1)
        self.assertIs(entry_points[0].load(), main)

    def test_module_help_exits_without_model_or_source(self):
        result = subprocess.run(
            [sys.executable, "-m", "ultiviz", "--help"],
            check=True, capture_output=True, text=True,
        )
        self.assertIn("--model", result.stdout)
        self.assertIn("--source", result.stdout)
        self.assertNotIn("Traceback", result.stderr)

    def test_filesystem_source_is_preserved(self):
        factory = Mock()
        factory.return_value.run.return_value = "output.mp4"
        with contextlib.redirect_stdout(io.StringIO()):
            main(["--model", "weights/model.pt", "--source", "videos/input video.mp4",
                  "--output", "output folder/result.mp4", "--headless"],
                 visualizer_class=factory)
        factory.return_value.run.assert_called_once_with(
            "videos/input video.mp4", "output folder/result.mp4", show_window=False,
        )

    def test_base_import_does_not_load_ultralytics(self):
        subprocess.run(
            [sys.executable, "-c", "import sys; from ultiviz import Visualizer; "
             "assert 'ultralytics' not in sys.modules; Visualizer()"],
            check=True, capture_output=True, text=True,
        )


if __name__ == "__main__":
    unittest.main()
