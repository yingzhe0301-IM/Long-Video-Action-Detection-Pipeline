"""Configuration helpers for experiments and detectors."""

from .experiment_config import ExperimentConfig
from .unified_config import (
    BASE_DETECTION,
    DEFAULT_ACTIONS,
    DEFAULT_WINDOW_SIZE,
    config_from_args,
    default_detection_mode,
    make_action_config,
)

__all__ = [
    "ExperimentConfig",
    "BASE_DETECTION",
    "DEFAULT_ACTIONS",
    "DEFAULT_WINDOW_SIZE",
    "config_from_args",
    "default_detection_mode",
    "make_action_config",
]
