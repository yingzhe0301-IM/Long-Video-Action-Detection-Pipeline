"""Post-inference helpers for peak-only action detection."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config.experiment_config import ExperimentConfig
from ..config.unified_config import scale_peak_params
from .peak_detection import detect_peaks


def _slide_window_average(data: List[int], window_size: int) -> Tuple[List[float], List[float]]:
    n = len(data)
    if n < window_size:
        return [], []

    data_arr = np.asarray(data, dtype=float)
    kernel = np.full(window_size, 1.0 / window_size, dtype=float)
    averages = np.convolve(data_arr, kernel, mode="valid")
    centers = np.arange(window_size // 2, window_size // 2 + averages.size, dtype=float)

    return centers.tolist(), averages.tolist()


def compute_signal(detections: List[int], window_size: int, frame_skip: int) -> Tuple[List[float], List[float]]:
    window = max(1, int(window_size / max(frame_skip or 1, 1)))
    binary_detections = [1 if count > 0 else 0 for count in detections]
    return _slide_window_average(binary_detections, window)


def _empty_result(action_type: str) -> Dict[str, Any]:
    return {
        "skipped": True,
        "centers": [],
        "avg_detections": [],
        "final_actions": [],
        "action_type": action_type,
        "detection_mode": "",
        "evaluation": {"skipped": True},
    }


def _parse_ground_truth(video_stem: str, action_type: str) -> Optional[int]:
    if action_type != "haul":
        return None

    parts = video_stem.split("_")
    suffix = parts[-1] if parts else ""
    if not suffix.isdigit():
        return None

    return len(suffix) if len(suffix) % 2 == 0 else None


def evaluate_action(video_stem: str, detected_actions: List[int], action_type: str) -> Dict[str, Any]:
    ground_truth = _parse_ground_truth(video_stem, action_type)
    if ground_truth is None:
        return {
            "skipped": True,
            "detected_actions": len(detected_actions),
            "gt_actions": 0,
            "success": False,
        }

    return {
        "skipped": False,
        "detected_actions": len(detected_actions),
        "gt_actions": ground_truth,
        "success": len(detected_actions) == ground_truth,
    }


def calculate_accuracy(results: List[Dict[str, Any]]) -> Dict[str, float]:
    total = 0
    success = 0

    for result in results:
        evaluation = result.get("evaluation", {})
        if evaluation.get("skipped"):
            continue
        total += 1
        if evaluation.get("success"):
            success += 1

    accuracy = (success / total) * 100 if total else 0.0
    return {"total_videos": total, "success_count": success, "accuracy": accuracy}


def analyze_detection_data(
    action_type: str,
    detection_data: Dict[str, Any],
    config: ExperimentConfig,
) -> Dict[str, Any]:
    if action_type != "haul":
        raise ValueError(f"Unsupported action type: {action_type}")

    detections = detection_data["detections_per_frame"]
    frame_skip = int(config.frame_skip or 1)
    centers, avg = compute_signal(detections, config.window_size, frame_skip)
    if not avg:
        return _empty_result(action_type)

    detection_params = scale_peak_params(frame_skip)
    actions = detect_peaks(centers, avg, detection_params)
    evaluation = evaluate_action(detection_data["video_stem"], actions, action_type)

    return {
        "centers": centers,
        "avg_detections": avg,
        "final_actions": actions,
        "evaluation": evaluation,
        "action_type": action_type,
        "detection_mode": config.detection_mode,
        "skipped": False,
    }


__all__ = [
    "analyze_detection_data",
    "calculate_accuracy",
    "compute_signal",
    "evaluate_action",
]
