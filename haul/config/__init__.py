"""Configuration helpers for experiments and detectors."""

from .experiment_config import (
    DEFAULT_CONFIG,
    DEFAULT_ADAPTIVE_CONFIG,
    ExperimentConfig,
    AdaptiveSkipConfig,
)
from .unified_config import PEAK_DETECTION_PARAMS, scale_peak_params

__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_ADAPTIVE_CONFIG",
    "ExperimentConfig",
    "AdaptiveSkipConfig",
    "PEAK_DETECTION_PARAMS",
    "scale_peak_params",
]
