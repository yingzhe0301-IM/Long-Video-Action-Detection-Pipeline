
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import torch


@dataclass
class ExperimentConfig:
    """
    A single object to hold all experiment configurations.
    This avoids passing numerous arguments between functions.
    """
    # Core settings from argparse
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
    use_multiprocessing: bool
    num_workers: int

    # NEW: Add a field for the target class ID (e.g., 0 for 'person')
    target_class_id: Optional[int] = None

    # Experiment mode-specific settings
    frame_skip: Optional[int] = None
    custom_skips: Optional[List[int]] = None
    min_skip: int = 1
    max_skip: int = 64

    # Environment-related properties (to be populated later)
    device: Optional[torch.device] = None
    video_path: Optional[Path] = None
    plot_path: Optional[Path] = None

    def __post_init__(self):
        """Perform initial setup after the object is created."""
        # This logic is now handled in fixed_experiments.py to allow defaults
        pass
