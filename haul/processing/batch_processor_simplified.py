"""Batch processing helpers for action detection experiments."""

import time
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..detection.detection_utils import get_device, load_yolo_model
from ..config.experiment_config import ExperimentConfig
from ..actions.action_registry import process_action
from ..actions.action_evaluator import calculate_accuracy
from .video_processor_unified import process_video


def get_video_files(video_root: Path) -> List[Path]:
    return sorted(video_root.rglob("*.mp4"))


def plot_results(result: Dict[str, Any], plot_folder: Path, model_weight: str, video_name: str) -> str:
    """Generate and save detection plot."""
    sub_folder = plot_folder / "latest"
    sub_folder.mkdir(parents=True, exist_ok=True)

    mode = result.get("detection_mode", "peak")
    weight_name = Path(model_weight).stem
    plot_file = sub_folder / f"{result.get('video_stem', video_name)}_{weight_name}_{mode}.png"

    fig, ax = plt.subplots(figsize=(10, 5))
    centers = result.get("centers", [])
    averages = result.get("avg_detections", [])
    ax.plot(centers, averages, label="Avg detections")

    if mode == "interval":
        for start, end in result.get("final_actions", []):
            if centers and averages:
                si = min(bisect_left(centers, start), len(centers) - 1)
                ei = max(bisect_right(centers, end) - 1, si)
                while si > 0 and averages[si - 1] > 0: si -= 1
                while ei + 1 < len(averages) and averages[ei + 1] > 0: ei += 1
                start, end = centers[si], centers[ei]
            ax.axvspan(start, end, color="#f58518", alpha=0.2)
    else:
        for idx in result.get("final_actions", []):
            if 0 <= idx < len(centers):
                ax.scatter(centers[idx], averages[idx], color="#d62728")

    ax.set_title(f"{result.get('action_type', 'Action').title()} Detection - {video_name} ({mode})")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Average detections")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_file, dpi=100)
    plt.close(fig)
    return str(plot_file)


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
    result = process_action(config.action_type, detection_data, config.to_action_config())
    result["video_name"] = detection_data["video_name"]
    result["video_stem"] = detection_data["video_stem"]

    if not result.get("skipped"):
        result["plot_path"] = plot_results(result, Path(config.plot_folder), config.model_weight, detection_data["video_name"])

    return result


def process_single_video(video_path: Path, config: ExperimentConfig, model: Optional[Any] = None) -> Dict[str, Any]:
    """Process a single video: detection + action analysis."""
    detection_data = run_detection(video_path, config, model)
    return analyze_actions(detection_data, config)


def process_all_videos(config: ExperimentConfig, model: Optional[Any] = None) -> float:
    """Process all videos and return accuracy."""
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

    return stats["accuracy"]