#!/usr/bin/env python3
"""Command-line entry point for action detection experiments."""

import argparse
import time
from pathlib import Path
from typing import List, Optional

from haul.inference.inference_utils import clear_memory, get_device, load_yolo_model
from haul.inference.pipeline_profiler import PipelineProfiler, SyncProfiler
from haul.config.experiment_config import (
    DEFAULT_CONFIG,
    DEFAULT_ADAPTIVE_CONFIG,
    ExperimentConfig,
    AdaptiveSkipConfig,
)
from haul.batch_runner import process_all_videos, process_single_video


def run_single_experiment(config: ExperimentConfig, video_file: Optional[str] = None) -> Optional[float]:
    print(f"\n{'=' * 50}")
    print(f"Action Type: {config.action_type}")
    print(f"Detection Mode: {config.detection_mode}")

    adaptive_cfg = config.adaptive_config
    if adaptive_cfg and adaptive_cfg.enabled:
        print(f"Adaptive Mode: enabled")
        print(f"  Initial Skip: {adaptive_cfg.initial_skip}")
        print(f"  Consecutive Negative Threshold: {adaptive_cfg.consecutive_negative_threshold}")
        print(f"  Max Skip: {adaptive_cfg.max_skip}")
    else:
        print(f"Frame Skip: {config.frame_skip}")
        print(f"Async Prefetch: {config.use_prefetch}")
        print(f"Decode All Frames: {config.decode_all}")

    print(f"Model: {config.model_weight}")
    print(f"Device: {config.device}")
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


def run_scan_experiment(
    config: ExperimentConfig,
    skip_values: List[int],
    print_prefetch_summary: bool = False,
) -> None:
    print(f"\n{'=' * 50}")
    print(f"FRAME SKIP SCAN: {config.action_type}")
    print(f"Skip Values: {skip_values}")
    print(f"Async Prefetch: {config.use_prefetch}")
    print(f"Decode All Frames: {config.decode_all}")
    print(f"{'=' * 50}\n")

    model = load_yolo_model(config.model_weight, device=config.device, confidence=config.confidence)
    results = []
    total_start = time.time()
    original_prefetch_log_stdout = config.prefetch_log_stdout
    config.prefetch_log_stdout = False

    for idx, skip in enumerate(skip_values, 1):
        print(f"[{idx}/{len(skip_values)}] frame_skip={skip}", end=" ... ")
        config.frame_skip = skip
        start = time.time()
        prefetch_summaries: List[dict] = []

        def _collect_prefetch_summary(summary: dict) -> None:
            prefetch_summaries.append(summary)

        stats = process_all_videos(
            config,
            model=model,
            return_details=True,
            verbose=False,
            prefetch_summary_collector=_collect_prefetch_summary,
        )
        runtime = time.time() - start
        print(f"Accuracy: {stats.get('accuracy', 0.0):.2f}% | Runtime: {runtime:.2f}s")

        summary_stats = {}
        if config.enable_profiler:
            if config.use_prefetch:
                summary_stats = PipelineProfiler.aggregate(prefetch_summaries)
                if print_prefetch_summary and prefetch_summaries:
                    PipelineProfiler.print_aggregate(prefetch_summaries, skip)
            else:
                summary_stats = SyncProfiler.aggregate(prefetch_summaries)
                if print_prefetch_summary and prefetch_summaries:
                    SyncProfiler.print_aggregate(prefetch_summaries, skip)

        results.append({
            "frame_skip": skip,
            "accuracy": stats.get("accuracy", 0.0),
            "success_count": stats.get("success_count", 0),
            "total_videos": stats.get("total_videos", 0),
            "runtime": runtime,
            "use_prefetch": config.use_prefetch,
            **summary_stats,
        })
        clear_memory(config.device)

    config.prefetch_log_stdout = original_prefetch_log_stdout

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
        ordered_cols = [
            "frame_skip",
            "success_count",
            "total_videos",
            "accuracy",
            "runtime",
            "use_prefetch",
        ]
        if config.enable_profiler and config.use_prefetch:
            ordered_cols.extend([
                "prefetch_videos",
                "prefetch_total_batches",
                "prefetch_avg_wait_ms",
                "prefetch_avg_infer_ms",
                "prefetch_avg_prepare_ms",
                "prefetch_wait_ratio",
                "prefetch_steady_batches",
                "prefetch_steady_avg_wait_ms",
                "prefetch_steady_avg_infer_ms",
                "prefetch_steady_avg_prepare_ms",
                "prefetch_steady_wait_ratio",
            ])
        elif config.enable_profiler and not config.use_prefetch:
            ordered_cols.extend([
                "sync_videos",
                "sync_total_batches",
                "sync_total_prepare_ms",
                "sync_total_infer_ms",
                "sync_avg_prepare_ms",
                "sync_avg_infer_ms",
                "sync_wait_ratio",
                "sync_ideal_serial_ms",
                "sync_ideal_overlap_ms",
                "sync_ideal_saved_ms",
                "sync_steady_batches",
                "sync_steady_total_prepare_ms",
                "sync_steady_total_infer_ms",
                "sync_steady_avg_prepare_ms",
                "sync_steady_avg_infer_ms",
                "sync_steady_wait_ratio",
            ])
        prefetch_tag = "prefetch" if config.use_prefetch else "no_prefetch"
        profiler_tag = "_profiler" if config.enable_profiler else ""
        decode_tag = "decode_all" if config.decode_all else "decode_on_demand"
        df = pd.DataFrame(results)
        if "runtime" in df.columns:
            df["runtime"] = df["runtime"].map(lambda x: f"{x:.2f}")
        df[ordered_cols].to_csv(
            results_dir / f"{config.action_type}_scan_{decode_tag}_{prefetch_tag}{profiler_tag}.csv",
            index=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run action detection experiments")
    parser.add_argument("--action_type", type=str, required=True,
                        choices=["haul"])

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--single", action="store_true")
    mode.add_argument("--scan", action="store_true")

    parser.add_argument("--detection_mode", type=str, choices=["peak"])
    parser.add_argument("--frame_skip", type=int, default=DEFAULT_CONFIG.frame_skip)
    parser.add_argument("--min_skip", type=int, default=DEFAULT_CONFIG.min_skip)
    parser.add_argument("--max_skip", type=int, default=DEFAULT_CONFIG.max_skip)
    parser.add_argument("--custom_skips", type=str)
    parser.add_argument("--video_root", type=str, default=DEFAULT_CONFIG.video_root)
    parser.add_argument("--model_weight", type=str)
    parser.add_argument("--plot_folder", type=str, default=DEFAULT_CONFIG.plot_folder)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIG.confidence)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument("--window_size", type=int, default=DEFAULT_CONFIG.window_size)
    parser.add_argument("--save_frames", action="store_true")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--video", type=str)
    parser.add_argument("--no-prefetch", action="store_true", help="Disable async prefetching")
    parser.add_argument("--decode-all", action="store_true",
                        help="Decode every frame with cap.read() before skipping inference")
    parser.add_argument("--prefetch_batches", type=int, default=DEFAULT_CONFIG.prefetch_batches)
    parser.add_argument("--print_prefetch_scan_summary", action="store_true",
                        help="Print per-skip prefetch summaries during scan")

    # Adaptive frame skip arguments
    parser.add_argument("--adaptive", action="store_true", help="Enable adaptive frame skip mode")
    parser.add_argument("--initial_skip", type=int, default=DEFAULT_ADAPTIVE_CONFIG.initial_skip,
                        help="Initial frame skip for adaptive mode")
    parser.add_argument("--consecutive_negative_threshold", type=int,
                        default=DEFAULT_ADAPTIVE_CONFIG.consecutive_negative_threshold,
                        help="Number of consecutive negatives before doubling skip")
    parser.add_argument("--adaptive_max_skip", type=int, default=DEFAULT_ADAPTIVE_CONFIG.max_skip,
                        help="Maximum frame skip for adaptive mode")

    # Profiler
    parser.add_argument("--profiler", action="store_true",
                        help="Enable pipeline profiler for timing analysis")

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

    # Build adaptive config if enabled
    adaptive_config = None
    if args.adaptive:
        adaptive_config = AdaptiveSkipConfig(
            enabled=True,
            initial_skip=args.initial_skip,
            consecutive_negative_threshold=args.consecutive_negative_threshold,
            max_skip=args.adaptive_max_skip,
        )

    return ExperimentConfig(
        action_type=args.action_type,
        detection_mode=args.detection_mode or DEFAULT_CONFIG.detection_mode,
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
        decode_all=args.decode_all,
        prefetch_batches=args.prefetch_batches,
        prefetch_log_stdout=True,
        enable_profiler=args.profiler,
        device=get_device(),
        adaptive_config=adaptive_config,
    )


def run_adaptive_scan_experiment(config: ExperimentConfig) -> None:
    """Sweep over adaptive hyperparameter combinations."""
    # Default scan ranges
    initial_skips = [1, 2, 4, 8]
    thresholds = [3, 5, 7, 10, 15]
    max_skips = [64, 128, 256]

    print(f"\n{'=' * 50}")
    print(f"ADAPTIVE FRAME SKIP SCAN: {config.action_type}")
    print(f"Initial Skips: {initial_skips}")
    print(f"Thresholds: {thresholds}")
    print(f"Max Skips: {max_skips}")
    print(f"{'=' * 50}\n")

    model = load_yolo_model(config.model_weight, device=config.device, confidence=config.confidence)
    results = []
    total_start = time.time()

    total_combinations = len(initial_skips) * len(thresholds) * len(max_skips)
    idx = 0

    for initial_skip in initial_skips:
        for threshold in thresholds:
            for max_skip in max_skips:
                idx += 1
                print(
                    f"[{idx}/{total_combinations}] initial_skip={initial_skip}, neg_threshold={threshold}, max_skip={max_skip}",
                    end=" ... ")

                # Update config with current hyperparameters
                config.adaptive_config = AdaptiveSkipConfig(
                    enabled=True,
                    initial_skip=initial_skip,
                    consecutive_negative_threshold=threshold,
                    max_skip=max_skip,
                )

                start = time.time()
                stats = process_all_videos(config, model=model, return_details=True, verbose=False)
                runtime = time.time() - start

                inference_ratio = stats.get("inference_ratio", 1.0)
                saved_cost = 1.0 - inference_ratio
                print(f"Accuracy: {stats.get('accuracy', 0.0):.2f}% | "
                      f"Saved: {saved_cost * 100:.1f}% | Runtime: {runtime:.2f}s")

                # Format error details as string: "video1(det/gt),video2(det/gt),..."
                error_details = stats.get("error_details", [])
                error_str = ",".join(
                    f"{e['video']}({e['detected']}/{e['gt']})" for e in error_details
                ) if error_details else ""

                results.append({
                    "initial_skip": initial_skip,
                    "threshold": threshold,
                    "max_skip": max_skip,
                    "accuracy": stats.get("accuracy", 0.0),
                    "success_count": stats.get("success_count", 0),
                    "total_videos": stats.get("total_videos", 0),
                    "detection_attempts": stats.get("detection_attempts", 0),
                    "total_frames": stats.get("total_frames", 0),
                    "inference_ratio": inference_ratio,
                    "saved_cost": saved_cost,
                    "runtime": runtime,
                    "errors": error_str,
                })
                clear_memory(config.device)

    # Summary
    print(f"\n{'=' * 90}\nADAPTIVE SCAN RESULTS\n{'=' * 90}")
    print(f"{'initial_skip':<14}{'neg_threshold':<15}{'max_skip':<10}{'Videos':<10}"
          f"{'Accuracy':<12}{'Saved Cost':<12}{'Runtime':<10}")
    best = max(results, key=lambda x: x["accuracy"])
    for r in results:
        accuracy_str = f"{r['accuracy']:.2f}%"
        saved_str = f"{r['saved_cost'] * 100:.1f}%"
        runtime_str = f"{r['runtime']:.2f}s"
        videos_str = f"{r.get('success_count', 0)}/{r.get('total_videos', 0)}"
        print(f"{r['initial_skip']:<14}{r['threshold']:<15}{r['max_skip']:<10}"
              f"{videos_str:<10}{accuracy_str:<12}{saved_str:<12}{runtime_str:<10}")

    print(f"\nBest: initial_skip={best['initial_skip']}, threshold={best['threshold']}, "
          f"max_skip={best['max_skip']} (Accuracy: {best['accuracy']:.2f}%, Saved: {best['saved_cost'] * 100:.1f}%)")
    print(f"Total time: {time.time() - total_start:.2f}s")

    if config.plot_folder:
        import pandas as pd
        results_dir = Path(config.plot_folder) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        ordered_cols = ["initial_skip", "threshold", "max_skip", "success_count",
                        "total_videos", "accuracy", "detection_attempts", "total_frames",
                        "inference_ratio", "saved_cost", "runtime", "errors"]
        pd.DataFrame(results)[ordered_cols].to_csv(
            results_dir / f"{config.action_type}_adaptive_scan.csv", index=False
        )


def main() -> None:
    args = parse_args()
    config = build_config(args)

    if args.single:
        run_single_experiment(config, args.video)
    elif args.adaptive and args.scan:
        run_adaptive_scan_experiment(config)
    elif args.scan:
        skips = [int(x) for x in args.custom_skips.split(",")] if args.custom_skips else list(
            range(args.min_skip, args.max_skip + 1))
        run_scan_experiment(config, skips, print_prefetch_summary=args.print_prefetch_scan_summary)
    else:
        # --adaptive --single (already handled by run_single_experiment via config)
        run_single_experiment(config, args.video)

    print("\nExperiment completed!")


if __name__ == "__main__":
    main()
