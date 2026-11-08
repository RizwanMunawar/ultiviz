# ultiviz

Simple visualization utilities for **Ultralytics YOLO results**: bounding boxes,
segmentation masks, pose keypoints and skeletons, oriented bounding boxes (OBB),
classification labels, depth heatmaps, and tracking trails. Visualize existing predictions or
pass a model to process a video in one call.

## Install

Requires Python 3.10 or newer. From the repository root:

```bash
python3 -m pip install -e ".[ultralytics]"
```

For drawing only, `python3 -m pip install -e .` installs NumPy and OpenCV without
Ultralytics. Model weights and input media are supplied by you; replace the
example paths below with your own.

## Task examples

All examples use the same `output = Visualizer(results)` call. Images are
saved to `annotated.jpg`; videos are saved to `annotated.mp4`. Both are displayed
automatically, and per-frame inference/visualization times are reported.
Image previews close on any key; video previews stop on Q or Escape. Use
`Visualizer(results, show=False)` when no desktop display is available.

Replace source paths with your own files. Ultralytics downloads the selected
pretrained weights on first use if they are not already available locally.
Semantic segmentation and depth require an Ultralytics version supporting these
YOLO26 tasks (examples verified with version 8.4.190).

### Object detection

```python
from ultralytics import YOLO
from ultraviz import Visualizer

model = YOLO("yolo26n.pt")
results = model.predict(source="path/to/image.jpg")
output = Visualizer(results)
```

### Instance segmentation

```python
from ultralytics import YOLO
from ultraviz import Visualizer

model = YOLO("yolo26n-seg.pt")
results = model.predict(source="path/to/image.jpg", retina_masks=True)
output = Visualizer(results)
```

### Pose estimation

```python
from ultralytics import YOLO
from ultraviz import Visualizer

model = YOLO("yolo26n-pose.pt")
results = model.predict(source="path/to/image.jpg")
output = Visualizer(results)
```

### Oriented bounding boxes (OBB)

```python
from ultralytics import YOLO
from ultraviz import Visualizer

model = YOLO("yolo26n-obb.pt")
results = model.predict(source="path/to/image.jpg")
output = Visualizer(results)
```

### Semantic segmentation

```python
from ultralytics import YOLO
from ultraviz import Visualizer

model = YOLO("yolo26n-sem.pt")
results = model.predict(source="path/to/image.jpg")
output = Visualizer(results)
```

### Depth estimation

```python
from ultralytics import YOLO
from ultraviz import Visualizer

model = YOLO("yolo26n-depth.pt")
results = model.predict(source="path/to/image.jpg")
output = Visualizer(results)
```

### Video inference

```python
from ultralytics import YOLO
from ultraviz import Visualizer

model = YOLO("yolo26n.pt")
results = model.predict(source="path/to/video.mp4", stream=True, retina_masks=True)
output = Visualizer(results)
```

This call processes the video immediately, saves `annotated.mp4`, displays the
annotations, and reports inference and visualization time for each frame.
No frame loop or video writer is needed. Press Q or Escape to stop playback;
`output` is the last annotated BGR frame, and the saved video contains all
processed frames without audio.

The same video pattern works for instance segmentation, pose, OBB, semantic
segmentation, and depth: replace `yolo26n.pt` with the corresponding weights from
the examples above. Keep `retina_masks=True` for full-resolution instance masks.
`stream=True` processes one frame at a time instead of loading the whole video
into memory. For headless use, call `Visualizer(results, show=False)` to save
without opening a preview window.

### Tracking

```python
from ultralytics import YOLO
from ultraviz import Visualizer

model = YOLO("yolo26n.pt")
results = model.track(source="path/to/video.mp4", stream=True, tracker="bytetrack.yaml")
output = Visualizer(results, show_trails=True)
```

For model names and task details, see the official
[YOLO26 overview](https://docs.ultralytics.com/models/yolo26/),
[semantic segmentation guide](https://docs.ultralytics.com/tasks/semantic/), and
[depth estimation guide](https://docs.ultralytics.com/tasks/depth/).

## Outputs and options

For instance segmentation, `retina_masks=True` preserves full-resolution masks;
use `show_boxes=False` to display masks alone. Depth colors are normalized per
frame; use `results[0].depth.data` for original model values. Invalid depth pixels
remain unchanged. `show_depth=False` hides depth, and `mask_opacity=1.0` displays
a fully colored map.

For video inference without tracking, use
`model.predict(source="path/to/video.mp4", stream=True)`. Tracking works with
detection, instance segmentation, pose, and OBB models.

`output` is an annotated BGR image array for one image or the last processed
frame for a video. The saved video contains all processed frames without audio.
Image batches return a list of arrays and save additional images as
`annotated_2.jpg`, `annotated_3.jpg`, and so on. Streamed images return the last
annotated image. Video processing retains only the current frames in memory;
use `stream=True` for long inputs.

Both `from ultraviz import Visualizer` and `from ultiviz import Visualizer` work;
the distribution and command-line name remain `ultiviz`. Existing output files
with the selected names are replaced; input files are protected from overwrite.

Optional output settings keep the same call:

```python
output = Visualizer(results, output="custom.mp4", fps=25, show=False)
```

Use a fresh prediction stream for each call. `show=False` saves without a preview;
`save=False` displays without saving. Use an image extension such as `.jpg` or
`.png` for image output and `.mp4` for video output. Video FPS is read from local
source metadata or defaults to 30; `fps=` overrides it. For inference managed by
ultiviz, use `Visualizer("yolo26n.pt").run("input.mp4", "output.mp4", show_window=False)`.

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
`show_chart`, `show_depth`, `class_ids`, `hidden_classes`, `hidden_track_ids`, and `line_width`.
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

## Per-frame speed

Timing is reported automatically for each rendered frame, for images and videos:

```text
Frame 1 | inference: 5.20 ms | visualization: 1.40 ms | processing: 6.60 ms
```

These are example values, not a speed guarantee. Inference, preprocessing, and
postprocessing metrics come from Ultralytics `result.speed`. Visualization time
is measured around frame copying and drawing only. `processing` adds inference
and visualization; it excludes preprocessing, postprocessing, saving, preview,
and console output. Missing inference metrics show `n/a`, never a fabricated zero.
When using `process_frame()`, `model call` additionally measures host wall time
around the model call (including its preprocessing/postprocessing and overhead).
It is separate from backend inference time and adds no extra GPU synchronization.

The existing one-call usage stays the same:

```python
output = Visualizer(results)
```

Use `timing=False` to disable timing work and console output. To consume metrics
without per-frame console writes, pass a callback instead:

```python
def report_speed(record):
    print(record.frame_index, record.inference_ms, record.visualization_ms)

output = Visualizer(results, on_timing=report_speed)
```

A reusable `Visualizer()` also exposes `last_timing`, a `FrameTiming` record with
`frame_index`, `inference_ms`, `visualization_ms`, `processing_ms`,
`preprocess_ms`, `postprocess_ms`, and `model_call_ms`. Only the latest record is
retained; `reset()` clears it. Callbacks run synchronously, so keep them short.

## Keep visualization fast

`Visualizer(results)` immediately saves and displays the predictions. For a
headless export, set `show=False` to skip preview work. Use `timing=False`
when console reporting is unnecessary. Prefer `stream=True` for
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

## License

Licensed under the GNU Affero General Public License version 3.0
(`AGPL-3.0-only`). See [LICENSE](LICENSE) for the complete license text.
