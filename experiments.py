#!/usr/bin/env python3
"""Command-line entry point for action detection experiments."""

import argparse
import time
from pathlib import Path
from typing import List, Optional

from haul.inference.inference_utils import clear_memory, get_device, load_yolo_model
from haul.config.experiment_config import DEFAULT_DETECTION_MODE, DEFAULT_EXPERIMENT_OPTIONS, ExperimentConfig
from haul.batch_runner import process_all_videos, process_single_video


def run_single_experiment(config: ExperimentConfig, video_file: Optional[str] = None) -> Optional[float]:
    print(f"\n{'=' * 50}")
    print(f"Action Type: {config.action_type}")
    print(f"Detection Mode: {config.detection_mode}")
    print(f"Frame Skip: {config.frame_skip}")
    print(f"Model: {config.model_weight}")
    print(f"Device: {config.device}")
    print(f"Async Prefetch: {config.use_prefetch}")
    print(f"{'=' * 50}\n")

    if video_file:
        video_path = Path(video_file)
        if not video_path.exists():
            video_path = Path(config.video_root) / config.action_type / video_file
            if not video_path.exists():
                print(f"Error: Video file not found: {video_file}")
                return 0.0

        print(f"Processing single video: {video_path.name}")
        model = load_yolo_model(config.model_weight, device=config.device, confidence=config.confidence)
        result = process_single_video(video_path, config, model)

        if result.get("skipped"):
            print("Processing skipped")
            return 0.0

        print(f"Plot saved to: {result.get('plot_path', 'unknown')}")
        evaluation = result.get("evaluation", {})
        if not evaluation.get("skipped"):
            print(f"Detected: {evaluation.get('detected_actions', 0)}, "
                  f"Ground Truth: {evaluation.get('gt_actions', 0)}, "
                  f"Success: {evaluation.get('success', False)}")
            return 100.0 if evaluation.get("success") else 0.0
        return 0.0

    return float(process_all_videos(config))


def run_scan_experiment(config: ExperimentConfig, skip_values: List[int]) -> None:
    print(f"\n{'=' * 50}")
    print(f"FRAME SKIP SCAN: {config.action_type}")
    print(f"Skip Values: {skip_values}")
    print(f"Async Prefetch: {config.use_prefetch}")
    print(f"{'=' * 50}\n")

    model = load_yolo_model(config.model_weight, device=config.device, confidence=config.confidence)
    results = []
    total_start = time.time()

    for idx, skip in enumerate(skip_values, 1):
        print(f"\n[{idx}/{len(skip_values)}] Testing frame_skip = {skip}")
        config.frame_skip = skip
        start = time.time()
        stats = process_all_videos(config, model=model, return_details=True)
        runtime = time.time() - start
        results.append({
            "frame_skip": skip,
            "accuracy": stats.get("accuracy", 0.0),
            "success_count": stats.get("success_count", 0),
            "total_videos": stats.get("total_videos", 0),
            "runtime": runtime,
            "use_prefetch": config.use_prefetch,
        })
        clear_memory(config.device)

    # Summary
    print(f"\n{'=' * 50}\nSCAN RESULTS\n{'=' * 50}")
    print(f"{'Skip':<10}{'Videos':<12}{'Accuracy':<12}{'Runtime':<12}")
    best = max(results, key=lambda x: x["accuracy"])
    for r in results:
        accuracy_str = f"{r['accuracy']:.2f}%"
        runtime_str = f"{r['runtime']:.2f}s"
        videos_str = f"{r.get('success_count', 0)}/{r.get('total_videos', 0)}"
        print(f"{r['frame_skip']:<10}{videos_str:<12}{accuracy_str:<12}{runtime_str:<12}")
    print(f"\nBest: frame_skip={best['frame_skip']} ({best['accuracy']:.2f}%)")
    print(f"Total time: {time.time() - total_start:.2f}s")

    if config.plot_folder:
        import pandas as pd
        results_dir = Path(config.plot_folder) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        ordered_cols = ["frame_skip", "success_count", "total_videos", "accuracy", "runtime", "use_prefetch"]
        pd.DataFrame(results)[ordered_cols].to_csv(results_dir / f"{config.action_type}_scan.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run action detection experiments")
    parser.add_argument("--action_type", type=str, required=True,
                        choices=["haul"])

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--single", action="store_true")
    mode.add_argument("--scan", action="store_true")

    parser.add_argument("--detection_mode", type=str, choices=["peak"])
    parser.add_argument("--frame_skip", type=int, default=DEFAULT_EXPERIMENT_OPTIONS["frame_skip"])
    parser.add_argument("--min_skip", type=int, default=DEFAULT_EXPERIMENT_OPTIONS["min_skip"])
    parser.add_argument("--max_skip", type=int, default=DEFAULT_EXPERIMENT_OPTIONS["max_skip"])
    parser.add_argument("--custom_skips", type=str)
    parser.add_argument("--video_root", type=str, default=DEFAULT_EXPERIMENT_OPTIONS["video_root"])
    parser.add_argument("--model_weight", type=str)
    parser.add_argument("--plot_folder", type=str, default=DEFAULT_EXPERIMENT_OPTIONS["plot_folder"])
    parser.add_argument("--confidence", type=float, default=DEFAULT_EXPERIMENT_OPTIONS["confidence"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_EXPERIMENT_OPTIONS["batch_size"])
    parser.add_argument("--window_size", type=int, default=DEFAULT_EXPERIMENT_OPTIONS["window_size"])
    parser.add_argument("--save_frames", action="store_true")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--video", type=str)
    parser.add_argument("--no-prefetch", action="store_true", help="Disable async prefetching")
    parser.add_argument("--prefetch_batches", type=int, default=DEFAULT_EXPERIMENT_OPTIONS["prefetch_batches"])
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    if args.video and not args.single:
        raise SystemExit("--video requires --single")

    # Resolve model weight: prefer explicit path, then action-specific weight under model_weights/, otherwise best.pt
    model_weight = args.model_weight
    if not model_weight:
        candidate = Path("model_weights") / f"{args.action_type}.pt"
        model_weight = str(candidate) if candidate.exists() else "best.pt"
    elif not Path(model_weight).exists():
        candidate = Path("model_weights") / model_weight
        if candidate.exists():
            model_weight = str(candidate)

    return ExperimentConfig(
        action_type=args.action_type,
        detection_mode=args.detection_mode or DEFAULT_DETECTION_MODE,
        video_root=args.video_root,
        plot_folder=args.plot_folder,
        model_weight=model_weight,
        confidence=args.confidence,
        batch_size=args.batch_size,
        window_size=args.window_size,
        save_frames=args.save_frames,
        display=args.display,
        frame_skip=args.frame_skip,
        use_prefetch=not args.no_prefetch,
        prefetch_batches=args.prefetch_batches,
        device=get_device(),
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)

    if args.single:
        run_single_experiment(config, args.video)
    else:
        skips = [int(x) for x in args.custom_skips.split(",")] if args.custom_skips else list(
            range(args.min_skip, args.max_skip + 1))
        run_scan_experiment(config, skips)

    print("\nExperiment completed!")


if __name__ == "__main__":
    main()
