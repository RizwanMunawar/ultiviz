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
from ultralytics import YOLO
from ultraviz import Visualizer

model = YOLO("model.pt")
results = model.predict(source="path/to/image.jpg", retina_masks=True)
output = Visualizer(results)
```

Both `from ultraviz import Visualizer` and `from ultiviz import Visualizer`
work; the distribution and command-line name remain `ultiviz`.
This call saves `annotated.jpg`, displays it, and returns the annotated BGR
image array. Press any key to close the image preview.

Use detection weights for bounding boxes, segmentation weights for masks, pose
weights for skeletons, or OBB weights for rotated boxes. The renderer selects the
right visualization automatically. `retina_masks=True` keeps instance masks at
the original image resolution. Drawing preserves the input image by default.

`Visualizer(results)` saves and displays predictions immediately. The same call
works with image and video sources; no frame loop, `cv2.imwrite`, or video writer
is needed in your code. Existing output files with the selected names are
replaced; input files are protected from being overwritten.

## Visualize a video stream

```python
from ultralytics import YOLO
from ultraviz import Visualizer

model = YOLO("model.pt")
results = model.predict(source="path/to/video.mp4", stream=True, retina_masks=True)
output = Visualizer(results)
```

The call consumes the prediction stream immediately, saves `annotated.mp4`,
and displays the video. `output` is the last annotated frame as a BGR image
array, matching the image return format. The saved file contains all processed
frames without audio. Press Q or Escape to stop early; the partial video is
finalized and resources are released.

Frames are processed one at a time without repeating inference or storing the
whole video in memory. Video prediction lists also work; `stream=True` is
recommended for long videos. Source FPS is read from local video metadata or
falls back to 30 FPS.

Optional settings use the same call:

```python
output = Visualizer(results, output="custom.mp4", fps=25, show=False)
```

Use a fresh prediction stream for each call. `show=False` saves without opening
a window and is required for servers, headless OpenCV, or environments without
a desktop display. Use `save=False` to display without saving. Image output paths
can use `.jpg`, `.png`, or another supported image extension; video output uses
`.mp4`. The default image filename is `annotated.jpg`, and batch images use
`annotated_2.jpg`, `annotated_3.jpg`, and so on. Image batches return a list of
annotated arrays; streamed images return the last annotated image.

For tracking trails, use tracked predictions:

```python
results = model.track(source="path/to/video.mp4", stream=True, retina_masks=True)
output = Visualizer(results, show_trails=True)
```

For video inference managed entirely by ultiviz, you can also use
`Visualizer("model.pt").run("input.mp4", "output.mp4", show_window=False)`.

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

`Visualizer(results)` immediately saves and displays the predictions. For a
headless export, set `show=False` to skip preview work. Prefer `stream=True` for
long videos to keep memory use bounded.

For pure rendering without saving or display, use
`Visualizer(results, save=False, show=False)`. This returns an annotated array
for one prediction, a list for multiple predictions, or a lazy frame iterator
for streamed input. Exhaust or close a lazy iterator to release its input.
Alternatively, create `visualizer = Visualizer()` once and call
`visualizer(result)` in your own processing loop.

Use `Visualizer(results, copy=False)`, `visualizer(result, copy=False)`,
or `process_frame(frame, copy=False)` to avoid copying the input image when you
can modify it in place. In-place drawing requires a writable, C-contiguous BGR
image. The default preserves your input.
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
