"""Detection utilities for each action type."""

from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import find_peaks

from ..detection.signal_processing_utils import (
    detect_action_intervals,
    detect_action_intervals_with_end_condition,
    filter_low_max_value_intervals,
    filter_short_intervals,
    group_peaks_adaptive,
    coalesce_close_peaks,
)


def detect_action_peaks(centers: List[float], avg_detections: List[float], config: Dict) -> List[int]:
    if not avg_detections:
        return []

    max_detection = np.max(avg_detections)
    prominence = config["peak_prominence_factor"] * max_detection
    min_height = config["fraction_of_max"] * max_detection
    peaks, _ = find_peaks(avg_detections, prominence=prominence, height=min_height)

    if peaks.size == 0:
        return []

    representative = group_peaks_adaptive(peaks, centers, avg_detections, config)
    return coalesce_close_peaks(representative, centers, avg_detections, config["coalesce_time_thr"])


def detect_action_intervals_basic(centers: List[float], avg_detections: List[float], config: Dict) -> List[Tuple[int, int]]:
    return detect_action_intervals(
        centers,
        avg_detections,
        threshold=config["interval_threshold"],
        max_gap=config.get("max_interval_gap", 0),
    )


def detect_setting_intervals(centers: List[float], avg_detections: List[float], config: Dict) -> List[Tuple[int, int]]:
    intervals = detect_action_intervals_with_end_condition(
        centers,
        avg_detections,
        start_threshold=config["start_threshold"],
        end_threshold=config["end_threshold"],
        min_end_duration=config["end_duration_threshold"],
    )

    duration_filtered = filter_short_intervals(intervals, config["min_duration"])
    return filter_low_max_value_intervals(
        duration_filtered,
        centers,
        avg_detections,
        config["max_val_threshold"],
    )
