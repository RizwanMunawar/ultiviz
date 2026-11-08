# ultiviz

Visualize **Ultralytics YOLO results** with one call: bounding boxes, instance and
semantic masks, pose skeletons, oriented boxes, depth heatmaps, and tracking
trails. `Visualizer(results)` automatically saves and displays annotations.

[Explore YOLO26 models](https://docs.ultralytics.com/models/yolo26/) ·
[Learn the Ultralytics prediction API](https://docs.ultralytics.com/modes/predict/)

## Install

Requires Python 3.10 or newer. From the repository root:

```bash
python3 -m pip install -e ".[ultralytics]"
```

Both `from ultraviz import Visualizer` and `from ultiviz import Visualizer` work.
Replace the example source paths with your own file paths. Ultralytics downloads model
weights on first use. Semantic and depth examples need a recent Ultralytics
version supporting those tasks; rendering is tested with 8.4.190.

## Examples

Images save to `annotated.jpg`; videos save to `annotated.mp4`. Press any key to
close an image preview, or Q/Escape to stop a video. Use `show=False` for headless
export. Each call needs fresh predictions when using a stream.

### Object detection

```python
from ultralytics import YOLO
from ultraviz import Visualizer

model = YOLO("yolo26n.pt")
results = model.predict(source="path/to/image.jpg")
output = Visualizer(results)
```

[Explore object detection models and usage →](https://docs.ultralytics.com/tasks/detect/)

### Instance segmentation

```python
from ultralytics import YOLO
from ultraviz import Visualizer

model = YOLO("yolo26n-seg.pt")
results = model.predict(source="path/to/image.jpg", retina_masks=True)
output = Visualizer(results)
```

[Explore instance segmentation and mask results →](https://docs.ultralytics.com/tasks/segment/)

### Pose estimation

```python
from ultralytics import YOLO
from ultraviz import Visualizer

model = YOLO("yolo26n-pose.pt")
results = model.predict(source="path/to/image.jpg")
output = Visualizer(results)
```

[Explore pose models and keypoint results →](https://docs.ultralytics.com/tasks/pose/)

### Oriented bounding boxes (OBB)

```python
from ultralytics import YOLO
from ultraviz import Visualizer

model = YOLO("yolo26n-obb.pt")
results = model.predict(source="path/to/image.jpg")
output = Visualizer(results)
```

[Explore oriented bounding box detection →](https://docs.ultralytics.com/tasks/obb/)

### Semantic segmentation

```python
from ultralytics import YOLO
from ultraviz import Visualizer

model = YOLO("yolo26n-sem.pt")
results = model.predict(source="path/to/image.jpg")
output = Visualizer(results)
```

[Explore semantic segmentation models →](https://docs.ultralytics.com/tasks/semantic/)

### Depth estimation

```python
from ultralytics import YOLO
from ultraviz import Visualizer

model = YOLO("yolo26n-depth.pt")
results = model.predict(source="path/to/image.jpg")
output = Visualizer(results)
```

[Explore depth estimation and depth maps →](https://docs.ultralytics.com/tasks/depth/)

### Video inference

```python
from ultralytics import YOLO
from ultraviz import Visualizer

model = YOLO("yolo26n.pt")
results = model.predict(source="path/to/video.mp4", stream=True, retina_masks=True)
output = Visualizer(results)
```

The same video pattern works with the segmentation, pose, OBB, semantic, and
depth weights above. `stream=True` keeps memory use bounded; no frame loop or
video writer is needed.

[Explore video sources and prediction settings →](https://docs.ultralytics.com/modes/predict/)

### Tracking

```python
from ultralytics import YOLO
from ultraviz import Visualizer

model = YOLO("yolo26n.pt")
results = model.track(source="path/to/video.mp4", stream=True, tracker="bytetrack.yaml")
output = Visualizer(results, show_trails=True)
```

[Explore trackers and tracking settings →](https://docs.ultralytics.com/modes/track/)

## Output and styling

`output` is an annotated BGR array for an image, or the last annotated frame for
a video. The saved video contains all processed frames without audio. Image
batches return a list of arrays and save numbered files. Existing output files
with the selected names are replaced; input files are protected from overwrite.

```python
output = Visualizer(
    results,
    output="custom.mp4",  # Use .jpg or .png for images
    show=False,          # Save without opening a preview window
    show_confidence=True,
    mask_opacity=0.5,
)
```

| Option | Use |
| --- | --- |
| `show=False` | Headless export |
| `save=False` | Preview without saving |
| `show_boxes=False` | Masks or poses without boxes |
| `show_trails=True` | Trails for predictions from `model.track()` |
| `class_colors={0: (0, 255, 0)}` | Custom BGR class colors |
| `mask_opacity=1.0` | Fully colored masks or depth maps |
| `timing=False` | Disable timing work and console reports |
| `copy=False` | Draw in place on a writable, contiguous BGR frame |

Video FPS comes from local metadata or defaults to 30; use `fps=25` to override
it. Depth colors are normalized per frame, so use `results[0].depth.data` for
original depth values. Invalid depth pixels remain unchanged.

For drawing only, `Visualizer(results, save=False, show=False)` returns arrays
or a lazy iterator for streamed input; exhaust or close that iterator when done.
You can also reuse `visualizer = Visualizer()` and call `visualizer(result)`.
The base install (`pip install -e .`) needs only NumPy and OpenCV.

## Per-frame speed

Each rendered frame reports model inference and visualization time:

```text
Frame 1 | inference: 5.20 ms | visualization: 1.40 ms | processing: 6.60 ms
```

Example values only. Inference comes from `result.speed`; visualization measures
copying and drawing. Their sum excludes preprocessing, postprocessing, saving,
and preview. Unavailable inference time displays `n/a`.
Use `on_timing=callback` for structured `FrameTiming` records instead of console
reports; reusable visualizers expose the latest record as `last_timing`.

[Learn about Ultralytics prediction results and speed metrics →](https://docs.ultralytics.com/modes/predict/#working-with-results)

## Command line

```bash
ultiviz --model yolo26n.pt --source input.mp4 --output annotated.mp4 --headless
python3 -m ultiviz --help
```

## License

[GNU Affero General Public License v3.0](LICENSE) (`AGPL-3.0`).
