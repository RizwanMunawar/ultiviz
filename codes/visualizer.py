"""Compatibility launcher; reusable code lives in src/cv_visualizer.

Install with: python3 -m pip install -e ".[ultralytics]"
Then import: from cv_visualizer import Visualizer
"""

import sys
from pathlib import Path

try:
    from cv_visualizer import Visualizer
except ModuleNotFoundError as exc:
    if exc.name != "cv_visualizer":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from cv_visualizer import Visualizer


if __name__ == "__main__":
    Visualizer.main()
