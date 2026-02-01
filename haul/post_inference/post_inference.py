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


def compute_signal_sparse(
    detections: List[Tuple[int, int]],
    window_size: int,
    total_frames: int,
) -> Tuple[List[float], List[float]]:
    """
    Compute signal from sparse detections in actual frame space.

    Uses step interpolation: each detection value fills until the next sample point.

    Args:
        detections: List of (frame_idx, detection_count) pairs, sorted by frame_idx.
        window_size: Sliding window size (unscaled, in actual frames).
        total_frames: Total number of frames in the video.

    Returns:
        Tuple of (centers, averages) in actual frame indices.
    """
    if not detections or total_frames == 0:
        return [], []

    # Step interpolation: create dense binary signal
    dense_signal = np.zeros(total_frames, dtype=float)

    for i, (frame_idx, det_count) in enumerate(detections):
        # Determine the range this sample covers
        if i + 1 < len(detections):
            next_idx = detections[i + 1][0]
        else:
            next_idx = total_frames

        # Fill with binary value (1 if detection, 0 otherwise)
        binary_value = 1.0 if det_count > 0 else 0.0
        dense_signal[frame_idx:next_idx] = binary_value

    # Apply sliding window average
    n = len(dense_signal)
    if n < window_size:
        return [], []

    kernel = np.full(window_size, 1.0 / window_size, dtype=float)
    averages = np.convolve(dense_signal, kernel, mode="valid")

    # Centers are in actual frame indices
    centers = np.arange(window_size // 2, window_size // 2 + averages.size, dtype=float)

    return centers.tolist(), averages.tolist()


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


def _is_sparse_detections(detections) -> bool:
    """Check if detections are in sparse format (list of tuples)."""
    if not detections:
        return False
    return isinstance(detections[0], (tuple, list)) and len(detections[0]) == 2


def analyze_detection_data(
    action_type: str,
    detection_data: Dict[str, Any],
    config: ExperimentConfig,
) -> Dict[str, Any]:
    if action_type != "haul":
        raise ValueError(f"Unsupported action type: {action_type}")

    detections = detection_data["detections_per_frame"]
    total_frames = detection_data.get("total_frames")

    # Detect adaptive mode by checking if detections is List[Tuple]
    is_adaptive = _is_sparse_detections(detections)

    if is_adaptive:
        # Adaptive mode: use sparse signal computation with unscaled parameters
        if total_frames is None:
            return _empty_result(action_type)
        centers, avg = compute_signal_sparse(detections, config.window_size, total_frames)
        # No scaling for adaptive mode (equivalent to frame_skip=1)
        detection_params = scale_peak_params(1)
    else:
        # Fixed frame skip mode: existing logic
        frame_skip = int(config.frame_skip or 1)
        centers, avg = compute_signal(detections, config.window_size, frame_skip)
        detection_params = scale_peak_params(frame_skip)

    if not avg:
        return _empty_result(action_type)

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
    "compute_signal_sparse",
    "evaluate_action",
]
