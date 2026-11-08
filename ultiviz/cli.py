"""Command-line entry point for installed packages and python -m."""

import argparse

from .ultiviz import Visualizer


def main(argv=None, *, visualizer_class=Visualizer):
    parser = argparse.ArgumentParser(description="Visualize detection, segmentation, pose, OBB, and tracking results.", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default="output-visualization.mp4")
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--trails", action="store_true", help="Also enables tracking")
    parser.add_argument("--chart", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--hide-boxes", action="store_true")
    parser.add_argument("--hide-labels", action="store_true")
    parser.add_argument("--show-confidence", action="store_true")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default=None)
    parser.add_argument("--classes", type=int, nargs="+")
    args = parser.parse_args(argv)
    visualizer = visualizer_class(args.model, tracking=args.track or args.trails, tracker=args.tracker,
                     confidence=args.conf, image_size=args.imgsz, device=args.device,
                     class_ids=args.classes, show_trails=args.trails, show_chart=args.chart,
                     show_boxes=not args.hide_boxes, show_labels=not args.hide_labels,
                     show_confidence=args.show_confidence)
    source = int(args.source) if args.source.isdecimal() else args.source
    output = visualizer.run(source, args.output, show_window=not args.headless)
    print(f"Saved: {output}")
