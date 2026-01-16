"""Batch orchestration helpers for action detection experiments."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config.experiment_config import ExperimentConfig
from .inference.inference_utils import get_device, load_yolo_model
from .inference.video_inference import process_video
from .post_inference.plotting import plot_results
from .post_inference.post_inference import analyze_detection_data, calculate_accuracy


def get_video_files(video_root: Path) -> List[Path]:
    """Return sorted list of .mp4 files under the given root."""
    return sorted(video_root.rglob("*.mp4"))


def run_detection(video_path: Path, config: ExperimentConfig, model: Optional[Any] = None) -> Dict[str, Any]:
    """Run YOLO detection on a single video."""
    device = config.device or get_device()
    model = model or load_yolo_model(config.model_weight, device=device, confidence=config.confidence)

    output_dir = Path(config.plot_folder) / f"output_frames_{video_path.stem}"

    detections = process_video(
        model, video_path, str(output_dir),
        batch_size=config.batch_size,
        frame_skip=config.frame_skip or 1,
        conf=config.confidence,
        device=device,
        display=config.display,
        save_annotated_frames=config.save_frames,
        prefetch_batches=config.prefetch_batches,
        use_async=config.use_prefetch,
    )

    return {
        "video_name": video_path.name,
        "video_stem": video_path.stem,
        "model_weight": config.model_weight,
        "detections_per_frame": detections,
    }


def analyze_actions(detection_data: Dict[str, Any], config: ExperimentConfig) -> Dict[str, Any]:
    """Analyze detection data to identify actions."""
    result = analyze_detection_data(config.action_type, detection_data, config)
    result["video_name"] = detection_data["video_name"]
    result["video_stem"] = detection_data["video_stem"]

    if not result.get("skipped"):
        result["plot_path"] = plot_results(result, Path(config.plot_folder), config.model_weight, detection_data["video_name"])

    return result


def process_single_video(video_path: Path, config: ExperimentConfig, model: Optional[Any] = None) -> Dict[str, Any]:
    """Process a single video: detection + action analysis."""
    detection_data = run_detection(video_path, config, model)
    return analyze_actions(detection_data, config)


def process_all_videos(
    config: ExperimentConfig,
    model: Optional[Any] = None,
    return_details: bool = False,
) -> Dict[str, float] | float:
    """Process all videos and return accuracy or full stats."""
    start_time = time.time()
    video_root = Path(config.video_root) / config.action_type
    video_files = get_video_files(video_root)

    if not video_files:
        print(f"No videos found in {video_root}")
        return 0.0

    Path(config.plot_folder).joinpath("latest").mkdir(parents=True, exist_ok=True)

    device = config.device or get_device()
    model = model or load_yolo_model(config.model_weight, device=device, confidence=config.confidence)

    results = []
    for idx, video_path in enumerate(video_files, 1):
        result = process_single_video(video_path, config, model)
        results.append(result)

        # Format status inline
        ev = result.get("evaluation", {})
        if result.get("skipped"):
            status = "SKIPPED"
        elif ev.get("skipped"):
            status = "processed (no ground truth)"
        else:
            status = f"DETECTED={ev['detected_actions']} | GT={ev['gt_actions']} | {'SUCCESS' if ev['success'] else 'FAIL'}"
        print(f"[{idx}/{len(video_files)}] [{video_path.name}] {status}")

    stats = calculate_accuracy(results)
    elapsed = time.time() - start_time

    if stats["total_videos"]:
        print(f"\nResults: {stats['success_count']}/{stats['total_videos']} videos correct")
        print(f"Accuracy: {stats['accuracy']:.2f}%")
    else:
        print("\nNo videos with valid ground truth found")
    print(f"Total time: {elapsed:.2f} seconds")

    return stats if return_details else stats["accuracy"]
