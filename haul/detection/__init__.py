"""Low-level detection and signal processing tools."""

from .detection_utils import get_device, load_yolo_model, slide_window_average
from .signal_processing_utils import (
    detect_action_intervals,
    detect_action_intervals_with_end_condition,
    filter_low_max_value_intervals,
    filter_short_intervals,
    group_peaks_adaptive,
    coalesce_close_peaks,
    split_cluster_by_valley,
)

__all__ = [
    "get_device",
    "load_yolo_model",
    "slide_window_average",
    "detect_action_intervals",
    "detect_action_intervals_with_end_condition",
    "filter_low_max_value_intervals",
    "filter_short_intervals",
    "group_peaks_adaptive",
    "coalesce_close_peaks",
    "split_cluster_by_valley",
]
