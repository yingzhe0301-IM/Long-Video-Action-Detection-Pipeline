# fixed_experiments.py

import argparse
from typing import List

from common_experiments import (
    run_single_fixed_experiment,
    run_fixed_frame_skip_scan
)
from experiment_config import ExperimentConfig


def generate_custom_skips(min_skip: int = 1, max_skip: int = 64) -> List[int]:
    return list(range(min_skip, max_skip + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed frame skip experiments (single run or scan)")
    # MODIFIED: Added 'catch' to choices
    parser.add_argument("--action_type", type=str, default="pumping", choices=["pumping", "haul", "setting", "catch"],
                        help="Type of action to detect")
    parser.add_argument("--detection_mode", type=str, default="interval", choices=["interval", "peak"],
                        help="Detection mode for 'pumping' or 'setting' type")
    parser.add_argument("--video_root", type=str, default="test_video", help="Base directory for videos")
    parser.add_argument("--plot_folder", type=str, default="plot", help="Folder for output plots")
    parser.add_argument("--model_weight", type=str, default=None,
                        help="Model weight filename. Defaults to best.pt for custom actions, or yolov8n.pt for 'setting'")
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing")
    parser.add_argument("--window_size", type=int, default=100, help="Window size for averaging")

    parser.add_argument("--target_class_id", type=int, default=None,
                        help="Specify a single class ID to detect (e.g., 7 for 'Human').")

    parser.add_argument("--save_frames", action="store_true", help="Save annotated frames")
    parser.add_argument("--display", action="store_true", help="Display results during processing")
    parser.add_argument("--multiprocess", action="store_true", help="Enable multiprocessing mode")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of worker processes")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--single", action="store_true", help="Run a single experiment with specified frame_skip")
    mode_group.add_argument("--scan", action="store_true", help="Run scan over frame_skip values")
    parser.add_argument("--frame_skip", type=int, default=5, help="(Single mode) Number of frames to skip")
    parser.add_argument("--min_skip", type=int, default=1, help="(Scan mode) Minimum frame_skip value")
    parser.add_argument("--max_skip", type=int, default=64, help="(Scan mode) Maximum frame_skip value")
    parser.add_argument("--auto_custom", action="store_true", help="Automatically generate custom frame skip values")
    parser.add_argument("--custom_skips", type=str, default=None,
                        help="Custom comma-separated list of frame skip values")
    args = parser.parse_args()

    model_weight = args.model_weight

    if model_weight is None:
        if args.action_type == 'setting':
            model_weight = 'yolov8n.pt'
        else:
            model_weight = 'best.pt'

    if 'yolo' not in model_weight:
        if not model_weight.startswith('model_weights/'):
            model_weight = f'model_weights/{model_weight}'

    if args.target_class_id is not None:
        print(f"Targeting single class detection for ID: {args.target_class_id}")

    if args.num_workers < 1:
        args.num_workers = 1
    if args.multiprocess:
        print(f"Multiprocessing mode ({args.num_workers} workers)")
    else:
        print("Single-Process mode")

    config = ExperimentConfig(
        action_type=args.action_type,
        detection_mode=args.detection_mode,
        video_root=args.video_root,
        plot_folder=args.plot_folder,
        model_weight=model_weight,
        confidence=args.confidence,
        batch_size=args.batch_size,
        window_size=args.window_size,
        save_frames=args.save_frames,
        display=args.display,
        use_multiprocessing=args.multiprocess,
        num_workers=args.num_workers,
        frame_skip=args.frame_skip,
        min_skip=args.min_skip,
        max_skip=args.max_skip,
        target_class_id=args.target_class_id,
    )

    if args.single:
        run_single_fixed_experiment(config)
    else:
        custom_skips = None
        if args.auto_custom:
            custom_skips = generate_custom_skips(config.min_skip, config.max_skip)
        elif args.custom_skips:
            custom_skips = [int(x.strip()) for x in args.custom_skips.split(',')]
        config.custom_skips = custom_skips
        run_fixed_frame_skip_scan(config)


if __name__ == "__main__":
    main()