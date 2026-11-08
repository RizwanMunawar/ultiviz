"""Compatibility launcher; reusable code lives in src/ultiviz.

Install with: python3 -m pip install -e ".[ultralytics]"
Then import: from ultiviz import Visualizer
"""

import sys
from pathlib import Path

try:
    from ultiviz import Visualizer
except ModuleNotFoundError as exc:
    if exc.name != "ultiviz":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from ultiviz import Visualizer


if __name__ == "__main__":
    Visualizer.main()
