from dataclasses import dataclass, field
from typing import Optional

import torch

DEFAULT_ACTION_TYPE = "haul"
DEFAULT_DETECTION_MODE = "peak"
DEFAULT_MODEL_WEIGHT = "haul.pt"
DEFAULT_WINDOW_SIZE = 100

DEFAULT_EXPERIMENT_OPTIONS = {
    "frame_skip": 5,
    "min_skip": 1,
    "max_skip": 64,
    "video_root": "video/test_video",
    "plot_folder": "plot",
    "confidence": 0.5,
    "batch_size": 64,
    "window_size": DEFAULT_WINDOW_SIZE,
    "prefetch_batches": 3,
    "prefetch_log_stdout": True,
}

DEFAULT_ADAPTIVE_OPTIONS = {
    "initial_skip": 2,
    "consecutive_negative_threshold": 5,
    "max_skip": 256,
}


@dataclass
class AdaptiveSkipConfig:
    """Configuration for adaptive frame skip strategy."""
    enabled: bool = False
    initial_skip: int = DEFAULT_ADAPTIVE_OPTIONS["initial_skip"]
    consecutive_negative_threshold: int = DEFAULT_ADAPTIVE_OPTIONS["consecutive_negative_threshold"]
    max_skip: int = DEFAULT_ADAPTIVE_OPTIONS["max_skip"]


@dataclass
class ExperimentConfig:
    """Container for experiment settings."""
    action_type: str
    detection_mode: str
    video_root: str
    plot_folder: str
    model_weight: str
    confidence: float
    batch_size: int
    window_size: int
    save_frames: bool
    display: bool
    frame_skip: Optional[int] = None
    use_prefetch: bool = True
    decode_all: bool = False
    prefetch_batches: int = 3
    prefetch_log_stdout: bool = DEFAULT_EXPERIMENT_OPTIONS["prefetch_log_stdout"]
    enable_profiler: bool = False
    device: Optional[torch.device] = None
    adaptive_config: Optional[AdaptiveSkipConfig] = None
