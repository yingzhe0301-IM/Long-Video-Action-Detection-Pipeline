from dataclasses import dataclass
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
    "video_root": "video/selected_test_video",
    "plot_folder": "plot",
    "confidence": 0.5,
    "batch_size": 64,
    "window_size": DEFAULT_WINDOW_SIZE,
    "prefetch_batches": 3,
}


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
    prefetch_batches: int = 3
    device: Optional[torch.device] = None
