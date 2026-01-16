"""Configuration helpers for experiments and detectors."""

from .experiment_config import (
    DEFAULT_ACTION_TYPE,
    DEFAULT_DETECTION_MODE,
    DEFAULT_EXPERIMENT_OPTIONS,
    DEFAULT_MODEL_WEIGHT,
    DEFAULT_WINDOW_SIZE,
    ExperimentConfig,
)
from .unified_config import PEAK_DETECTION_PARAMS, scale_peak_params

__all__ = [
    "DEFAULT_ACTION_TYPE",
    "DEFAULT_DETECTION_MODE",
    "DEFAULT_EXPERIMENT_OPTIONS",
    "DEFAULT_MODEL_WEIGHT",
    "DEFAULT_WINDOW_SIZE",
    "ExperimentConfig",
    "PEAK_DETECTION_PARAMS",
    "scale_peak_params",
]
