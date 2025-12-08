# detection_utils.py - Simplified version

import gc
from typing import List, Tuple, Optional, Union

import numpy as np
import torch
from ultralytics import YOLO


def get_device() -> torch.device:
    """Return the best available torch device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def load_yolo_model(model_weight: str, device: Optional[torch.device] = None, confidence: float = 0.5) -> YOLO:
    """Load YOLO model"""
    if device is None:
        device = get_device()
    model = YOLO(model_weight).to(device)
    return model


def supports_half_precision(device: Optional[torch.device] = None) -> bool:
    """Return True when inference can run in half precision."""
    if device is not None:
        return device.type == "cuda"
    return torch.cuda.is_available()


def clear_memory(device: Optional[torch.device]) -> None:
    """Release Python and torch caches between runs to curb RAM growth."""
    gc.collect()
    if device:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()


def slide_window_average(data: List[Union[int, float]], window_size: int) -> Tuple[List[float], List[float]]:
    """Apply sliding window average to data"""
    n = len(data)
    if n < window_size:
        return [], []

    data_arr = np.asarray(data, dtype=float)
    kernel = np.full(window_size, 1.0 / window_size, dtype=float)
    averages = np.convolve(data_arr, kernel, mode="valid")
    centers = np.arange(window_size // 2, window_size // 2 + averages.size, dtype=float)

    return centers.tolist(), averages.tolist()
