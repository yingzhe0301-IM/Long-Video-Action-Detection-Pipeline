from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class AdaptiveSkipConfig:
    """Configuration for adaptive frame skip strategy."""
    enabled: bool = False
    initial_skip: int = 2
    consecutive_negative_threshold: int = 5
    max_skip: int = 256


@dataclass
class ExperimentConfig:
    """Container for experiment settings."""
    action_type: str = "haul"
    detection_mode: str = "peak"
    video_root: str = "video/test_video"
    plot_folder: str = "plot"
    model_weight: str = "haul.pt"
    confidence: float = 0.5
    batch_size: int = 64
    window_size: int = 100
    save_frames: bool = False
    display: bool = False
    frame_skip: Optional[int] = 5
    min_skip: int = 1
    max_skip: int = 64
    use_prefetch: bool = True
    decode_all: bool = False
    prefetch_batches: int = 4
    prefetch_log_stdout: bool = True
    enable_profiler: bool = False
    device: Optional[torch.device] = None
    adaptive_config: Optional[AdaptiveSkipConfig] = None


DEFAULT_CONFIG = ExperimentConfig()
DEFAULT_ADAPTIVE_CONFIG = AdaptiveSkipConfig()
