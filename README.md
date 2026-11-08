# ultiviz

A reusable Python package for drawing detection boxes, instance and semantic
segmentation masks, pose skeletons, oriented bounding boxes (OBB), classification
labels, tracking trails, and class distribution charts with OpenCV.

The public API is `from ultiviz import Visualizer`. Rendering is organized
into separate modules while preserving the constructor and drawing methods from
the original single-file implementation.

## Install locally

Requires Python 3.10 or newer. Run from this project's root:

```bash
python3 -m pip install -e .
```

For built-in YOLO inference and tracking:

```bash
python3 -m pip install -e ".[ultralytics]"
```

The base package needs only NumPy and OpenCV. Ultralytics is imported lazily
when loading model weights; drawing helpers do not require it.

## Annotate existing results

```python
from ultralytics import YOLO
from ultiviz import Visualizer

model = YOLO("models/fruit-yolo26n-seg.pt")
visualizer = Visualizer(show_chart=True, show_boxes=False)
result = model.predict("image.jpg", retina_masks=True)[0]
annotated = visualizer.annotate(result.orig_img, result)
```

`annotate()` detects the result type and returns a copy of the frame. Use weights
for the appropriate task: detection, instance segmentation, semantic
segmentation, pose, OBB, or classification. Semantic rendering requires an
Ultralytics version/model exposing `result.semantic_mask` as a class-ID map.
For instance masks, `retina_masks=True` preserves full-resolution masks and holes.

## Run inference and video visualization

```python
from ultiviz import Visualizer

visualizer = Visualizer(
    "models/person-hg-yolo26n.pt",
    tracking=True,
    show_trails=True,
    show_confidence=True,
)
visualizer.run(
    "videos/construction-workers-demo.mov",
    "output.mp4",
    show_window=False,
)
```

Use `process_frame(frame)` for an existing video loop. `run()` supports video
paths and integer camera indices; set `output=None` to disable saving. Paths are
relative to your working directory. Models and videos are supplied by the caller
and are excluded from package distributions.

Tracking works with detection, instance segmentation, pose, and OBB. When calling
`annotate()` directly, results must already contain tracker IDs for trails to
appear. Call it once per frame, in order, and use `reset()` before switching
sequences. `run()` resets visualization state and the model's trackers itself.

## Command line

```bash
ultiviz --model models/person-hg-yolo26n.pt \
  --source videos/construction-workers-demo.mov \
  --output output.mp4 --track --trails --headless

python3 -m ultiviz --help
```

The CLI requires explicit `--model` and `--source` arguments. Other options
include `--chart`, `--hide-boxes`, `--hide-labels`, `--show-confidence`,
`--classes 0 1`, `--conf 0.35`, `--imgsz 640`, and `--device cpu`.

The original launcher still works from a source checkout without installing:

```bash
python3 codes/visualizer.py --model models/fruit-yolo26n-seg.pt \
  --source videos/fruits.mp4 --chart --hide-boxes
```

## Drawing and styling

```python
from ultiviz import Visualizer

visualizer = Visualizer(
    class_colors={0: (255, 0, 0)},
    hidden_classes={"background"},
    hidden_track_ids={9},
    mask_opacity=0.45,
    line_width=2,
    corner_radius=8,
    skeleton=[(0, 1), (1, 2)],
)
```

All colors are BGR. Custom skeleton edges use zero-based indices. COCO skeleton
edges are automatic for 17-keypoint poses; other layouts draw points unless a
skeleton is supplied. Keypoints below `keypoint_confidence` are omitted.

Public drawing methods modify their frame in place and return it:

| Method | Purpose |
| --- | --- |
| `draw_box(frame, xyxy, color, label=None)` | Rounded axis-aligned box |
| `draw_obb(frame, polygon, color, label=None)` | Rounded oriented box |
| `draw_label(frame, text, anchor, color, above=True)` | Rounded label (`above` is keyword-only) |
| `draw_masks(frame, masks, colors)` | Full-resolution mask blending and boundaries |
| `draw_pose(frame, keypoints, color)` | Points and skeleton edges |
| `draw_region(frame, polygon, color, label=None, opacity=0.15)` | Zone overlay |
| `draw_line(frame, start, end, color, label=None)` | Counting line and markers |
| `draw_chart(frame, counts, names, semantic=False)` | Donut and class legend (`semantic` is keyword-only) |

Charts show current-frame object counts, or pixel percentages among visible
semantic classes. They do not accumulate unique tracks. Counting and proximity
rules stay in the calling application; their results can be displayed with the
region, line, and label helpers.

## Package layout

```text
pyproject.toml
src/ultiviz/
    __init__.py       # Public API and version
    __main__.py       # python -m ultiviz
    visualizer.py     # Configuration and result dispatch
    detections.py     # Detection, OBB, mask, and pose coordination
    drawing.py        # Rounded geometry, labels, boxes, regions, lines
    segmentation.py   # Mask compositing and semantic rendering
    pose.py           # Skeletons and keypoints
    charts.py         # Donut charts and legends
    colors.py         # Palette and normalized class names
    tracking.py       # Trail history and expiry
    inference.py      # Optional YOLO integration
    video.py          # Capture, writing, preview, cleanup
    cli.py            # Command-line argument handling
tests/
codes/visualizer.py   # Compatibility launcher
```

Implementation mixins are internal; applications should import `Visualizer`
from the package root.

## Supported platforms and tests

GitHub Actions runs the base package tests on macOS, Linux, and Windows with
Python 3.10, 3.11, 3.12, and 3.13. Separate Python 3.12 jobs on each OS install
Ultralytics and exercise rendering with real Results objects. Tests use synthetic
frames, temporary video files, and model doubles; they do not download model
weights, use a camera, or open GUI windows.

Base tests cover inference dispatch, tracking arguments, camera/path handling,
video resource cleanup on success and failure, FPS fallback, CLI entry points,
and a real MP4 encode/process/decode round trip. Path tests include spaces, and
video files are explicitly removed after processing to catch leaked handles on
Windows. Integration tests are skipped when Ultralytics is not installed.
The build job validates the wheel and source distribution and tests the installed
wheel separately.

Run the base suite after installing the package:

```bash
python3 -m pip install .
python3 -m unittest discover -s tests -v
```

## Test and build

```bash
python3 -m pip install -e ".[dev]"
python3 -m unittest discover -s tests -v
python3 -m build
python3 -m twine check dist/*
```

Builds produce a wheel and source distribution under `dist/`. Only the package,
tests, and project metadata belong in these distributions; local model weights,
videos, and the unrelated example scripts are excluded.

## Publishing

`ultiviz` is the distribution, import, and CLI name. PyPI name availability has
not been checked. Before a public release, add your author details and chosen
license, and update both the project version and `ultiviz.__version__` together.

To upload the reviewed distributions to TestPyPI, then PyPI:

```bash
python3 -m twine upload --repository testpypi dist/*
python3 -m twine upload dist/*
```

These commands require your registry credentials. No upload is performed by
building or installing the package.
