"""Route raw detections through the appropriate action-specific logic."""

from typing import Any, Callable, Dict, List, Optional, Tuple

from ..detection.detection_utils import slide_window_average
from .action_detectors import (
    detect_action_intervals_basic,
    detect_action_peaks,
    detect_setting_intervals,
)
from .action_evaluator import evaluate_action
from ..config.unified_config import default_detection_mode, make_action_config


def _compute_signal(detections: List[int], config: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    window = max(1, int(config["window_size"] / max(config["frame_skip"], 1)))
    return slide_window_average(detections, window)


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


def _build_result(
    centers: List[float],
    avg_detections: List[float],
    final_actions: List,
    evaluation: Dict[str, Any],
    action_type: str,
    detection_mode: str,
) -> Dict[str, Any]:
    return {
        "centers": centers,
        "avg_detections": avg_detections,
        "final_actions": final_actions,
        "evaluation": evaluation,
        "action_type": action_type,
        "detection_mode": detection_mode,
        "skipped": False,
    }


def _process_action_generic(
    detection_data: Dict[str, Any],
    config: Dict[str, Any],
    *,
    action_type: str,
    detection_mode: str,
    detector: Callable[[List[float], List[float], Dict[str, Any]], Any],
) -> Dict[str, Any]:
    detections = detection_data["detections_per_frame"]
    centers, avg = _compute_signal(detections, config)
    if not avg:
        return _empty_result(action_type)

    actions = detector(centers, avg, config["detection"])
    evaluation = evaluate_action(detection_data["video_stem"], actions, action_type)
    mode = config.get("detection_mode", detection_mode)
    return _build_result(centers, avg, actions, evaluation, action_type, mode)


_PROCESSORS: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = {
    "pumping": lambda data, cfg: _process_action_generic(
        data, cfg, action_type="pumping", detection_mode="interval", detector=detect_action_intervals_basic
    ),
    "setting": lambda data, cfg: _process_action_generic(
        data, cfg, action_type="setting", detection_mode="interval", detector=detect_setting_intervals
    ),
    "haul": lambda data, cfg: _process_action_generic(
        data, cfg, action_type="haul", detection_mode="peak", detector=detect_action_peaks
    ),
    "catch": lambda data, cfg: _process_action_generic(
        data, cfg, action_type="catch", detection_mode="interval", detector=detect_action_intervals_basic
    ),
}


def process_action(
    action_type: str,
    detection_data: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if action_type not in _PROCESSORS:
        raise ValueError(f"Unknown action type: {action_type}")

    config = config or make_action_config(action_type)
    return _PROCESSORS[action_type](detection_data, config)


__all__ = [
    "process_action",
    "default_detection_mode",
]
