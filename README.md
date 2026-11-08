# ultiviz

Simple visualization utilities for **Ultralytics YOLO results**: bounding boxes,
segmentation masks, pose keypoints and skeletons, oriented bounding boxes (OBB),
classification labels, and tracking trails. Visualize existing predictions or
pass a model to process a video in one call.

## Install

Requires Python 3.10 or newer. From the repository root:

```bash
python3 -m pip install -e ".[ultralytics]"
```

For drawing only, `python3 -m pip install -e .` installs NumPy and OpenCV without
Ultralytics. Model weights and input media are supplied by you; replace the
example paths below with your own.

## Visualize an image

```python
import cv2
from ultralytics import YOLO
from ultraviz import Visualizer

model = YOLO("model.pt")
results = model.predict(source="path/to/image.jpg", retina_masks=True)
output = Visualizer(results)
if not cv2.imwrite("annotated.jpg", output):
    raise OSError("Could not save annotated.jpg")
```

Both `from ultraviz import Visualizer` and `from ultiviz import Visualizer`
work; the distribution and command-line name remain `ultiviz`.
`cv2.imwrite` needs an output filename and the annotated image.

Use detection weights for bounding boxes, segmentation weights for masks, pose
weights for skeletons, or OBB weights for rotated boxes. The renderer selects the
right visualization automatically. `retina_masks=True` keeps instance masks at
the original image resolution. Drawing preserves the input image by default.

| Input to `Visualizer(...)` | Output |
| --- | --- |
| A single Results object | One BGR image array |
| A one-item prediction list or tuple | One BGR image array |
| Multiple predictions in a list or tuple | A list of BGR image arrays in input order |
| A prediction iterator (`stream=True`) | A lazy iterator of BGR image arrays |

An empty list raises `ValueError`; an empty stream yields no frames. For multiple
images, save each returned image separately. Model paths and model objects still
create a reusable visualizer for inference, as shown below.

## Visualize existing video predictions

Use `stream=True` for videos so predictions and annotations are processed one
frame at a time:

```python
results = model.predict(source="path/to/video.mp4", stream=True, retina_masks=True)
outputs = Visualizer(results)
try:
    for output in outputs:
        # Display output or write it with your cv2.VideoWriter here.
        pass
finally:
    outputs.close()  # Also closes the prediction generator if you stop early.
```

Frames are rendered on demand through one shared renderer; inference is not
repeated. A video contains multiple frames, so use `cv2.VideoWriter` to save a
video or the one-call video API below.

## Visualize a video

```python
from ultiviz import Visualizer

Visualizer("model.pt").run("input.mp4", "output.mp4", show_window=False)
```

Enable tracking trails with one option:

```python
Visualizer("model.pt", show_trails=True).run(
    "input.mp4", "tracked.mp4", show_window=False,
)
```

Pass `0` instead of a video path for a webcam. Set `show_window=True` for a live
preview (press Q or Escape to stop), or `output=None` to disable saving. Use
`process_frame(frame)` to run inference and visualization in your own video loop.

## Customize the visualization

```python
visualizer = Visualizer(
    show_boxes=False,       # Show segmentation masks without boxes
    show_confidence=True,
    mask_opacity=0.5,
    class_colors={0: (0, 255, 0)},  # BGR colors
)
annotated = visualizer(results)
```

Other options include `show_labels`, `show_masks`, `show_pose`, `show_centers`,
`show_chart`, `class_ids`, `hidden_classes`, `hidden_track_ids`, and `line_width`.
For custom poses, pass zero-based skeleton edges such as `skeleton=[(0, 1)]`;
17-keypoint poses use the COCO skeleton automatically.

| Drawing utility | Use |
| --- | --- |
| `draw_box(frame, xyxy, color, label=None)` | Bounding box |
| `draw_obb(frame, polygon, color, label=None)` | Oriented bounding box |
| `draw_masks(frame, masks, colors)` | Segmentation masks and boundaries |
| `draw_pose(frame, keypoints, color)` | Keypoints and skeleton |
| `draw_label(frame, text, anchor, color)` | Text label |
| `draw_region(frame, polygon, color)` | Region overlay |
| `draw_line(frame, start, end, color)` | Line and endpoint markers |
| `draw_chart(frame, counts, names)` | Class counts and legend |

Call these methods on a `Visualizer()` instance. Drawing utilities modify and
return the supplied BGR frame; `draw_masks` expects full-resolution masks.

Semantic segmentation rendering requires results exposing `semantic_mask` as a
class-ID map. Tracking supports detection, instance segmentation, pose, and OBB.
When annotating existing tracked results, enable `show_trails=True` and supply
results containing tracker IDs once per frame, in order. Call `reset()` between
sequences; `run()` resets visualization state and model trackers automatically.
Charts show current-frame counts or visible semantic-class pixel percentages.

## Keep visualization fast

Create one `Visualizer` and reuse it across frames. For an existing YOLO loop,
pass each prediction directly without running inference twice:

```python
visualizer = Visualizer()
for result in model.predict("input.mp4", stream=True, retina_masks=True):
    annotated = visualizer(results)
    # Display, save, or pass annotated to your application here.
```

Use `Visualizer(results, copy=False)`, `visualizer(result, copy=False)`,
or `process_frame(frame, copy=False)` to
avoid copying the input image when you can modify it in place. In-place drawing
requires a writable, C-contiguous BGR image. The default preserves your input.
Mask overlays retain cropped regions, and rounded geometry uses vectorized
calculations. Disable optional features such as charts and trails when unused.
For `run()`, `show_window=False` avoids preview work. Model inference speed
still depends on your weights, device, and `image_size` (default: 1280).

## Command line

```bash
ultiviz --model model.pt --source input.mp4 --output output.mp4 --headless
ultiviz --model model.pt --source input.mp4 --track --trails --headless
python3 -m ultiviz --help
```

Use `--source 0` for a webcam. Options include `--hide-boxes`, `--hide-labels`,
`--show-confidence`, `--chart`, `--classes 0 1`, `--conf 0.35`, `--imgsz 640`, and
`--device cpu`.

## Development

The public API is `from ultiviz import Visualizer`; its implementation lives in
`ultiviz/ultiviz.py`, with drawing, inference, and video helpers alongside it.

```bash
python3 -m pip install -e ".[dev]"
python3 -m unittest discover -s tests -v
python3 -m build
python3 -m twine check dist/*
```

CI tests macOS, Linux, and Windows, including installed-package and wheel checks.
Tests use synthetic data without downloading weights, accessing cameras, or
opening preview windows.
