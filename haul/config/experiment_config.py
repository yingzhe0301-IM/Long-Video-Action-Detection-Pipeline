from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from .unified_config import make_action_config


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
    max_interval_gap: Optional[int] = None
    use_prefetch: bool = True
    prefetch_batches: int = 3
    device: Optional[torch.device] = None

    def to_action_config(self) -> Dict[str, Any]:
        overrides = {"window_size": self.window_size}
        if self.frame_skip is not None:
            overrides["frame_skip"] = self.frame_skip
        if self.detection_mode:
            overrides["detection_mode"] = self.detection_mode
        if self.model_weight:
            overrides["model_weight"] = self.model_weight
        if self.max_interval_gap is not None:
            overrides["max_interval_gap"] = self.max_interval_gap
        return make_action_config(self.action_type, **overrides)


DEFAULT_EXPERIMENT_OPTIONS = {
    "frame_skip": 5,
    "min_skip": 1,
    "max_skip": 64,
    "video_root": "video/selected_test_video",
    "plot_folder": "plot",
    "confidence": 0.5,
    "batch_size": 64,
    "window_size": 100,
    "max_interval_gap": None,
    "prefetch_batches": 3,
}
