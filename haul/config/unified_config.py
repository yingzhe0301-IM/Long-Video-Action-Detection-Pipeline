"""Simplified configuration helpers for action detection."""

from copy import deepcopy
from typing import Any, Dict

# Baseline detection parameters (frame_skip = 1)
BASE_DETECTION: Dict[str, Any] = {
    "peak_prominence_factor": 0.05,
    "fraction_of_max": 0.25,
    "adaptive_factor": 2.0,
    "valley_ratio": 0.7,
    "coalesce_time_thr": 1200.0,

    "start_threshold": 0.5,
    "end_threshold": 0.0,
    "end_duration_threshold": 400,
    "min_duration": 1000,
    "max_val_threshold": 3.0,
    "max_interval_gap": 250,
    "interval_threshold": 0.25,

}

# Parameters that scale with frame_skip
INT_FRAMES = {
    "end_duration_threshold",
    "min_duration",
    "max_interval_gap",
}
FLOAT_FRAMES = {"coalesce_time_thr"}

DEFAULT_ACTIONS: Dict[str, Dict[str, Any]] = {
    "pumping": {"detection_mode": "interval", "model_weight": "pumping.pt"},
    "setting": {"detection_mode": "interval", "model_weight": "yolov8n.pt"},
    "haul": {"detection_mode": "peak", "model_weight": "best.pt"},
    "catch": {"detection_mode": "interval", "model_weight": "best.pt"},
}

DEFAULT_WINDOW_SIZE = 100


def _scaled_detection(frame_skip: int, overrides: Dict[str, Any]) -> Dict[str, Any]:
    params = deepcopy(BASE_DETECTION)
    params.update({k: v for k, v in overrides.items() if k in BASE_DETECTION})
    skip = max(int(frame_skip or 1), 1)

    for name in FLOAT_FRAMES:
        params[name] = params[name] / skip

    for name in INT_FRAMES:
        params[name] = int(params[name] / skip)

    return params


def make_action_config(action_type: str, **overrides: Any) -> Dict[str, Any]:
    if action_type not in DEFAULT_ACTIONS:
        raise ValueError(f"Unknown action type: {action_type}")

    base = DEFAULT_ACTIONS[action_type]
    overrides = dict(overrides)

    frame_skip = int(overrides.pop("frame_skip", 1) or 1)
    window_size = overrides.pop("window_size", DEFAULT_WINDOW_SIZE)
    detection_mode = overrides.pop("detection_mode", base["detection_mode"])
    model_weight = overrides.pop("model_weight", base["model_weight"])

    detection_overrides = {k: overrides.pop(k) for k in list(overrides.keys()) if k in BASE_DETECTION}

    if overrides:
        unknown = ", ".join(sorted(overrides))
        raise ValueError(f"Unknown config overrides: {unknown}")

    return {
        "action_type": action_type,
        "detection_mode": detection_mode,
        "model_weight": model_weight,
        "frame_skip": frame_skip,
        "window_size": window_size,
        "detection": _scaled_detection(frame_skip, detection_overrides),
    }


def config_from_args(args) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}

    for name in ("frame_skip", "window_size", "detection_mode", "model_weight"):
        if hasattr(args, name):
            value = getattr(args, name)
            if value not in (None, ""):
                overrides[name] = value

    for name in BASE_DETECTION:
        if hasattr(args, name):
            value = getattr(args, name)
            if value is not None:
                overrides[name] = value

    return make_action_config(args.action_type, **overrides)


def default_detection_mode(action_type: str) -> str:
    if action_type not in DEFAULT_ACTIONS:
        raise ValueError(f"Unknown action type: {action_type}")
    return DEFAULT_ACTIONS[action_type]["detection_mode"]
