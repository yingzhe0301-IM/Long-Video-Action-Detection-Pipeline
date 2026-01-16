"""Peak-only detection algorithms for the haul pipeline."""

from typing import Dict, List

import numpy as np
from scipy.signal import find_peaks


def _coalesce_close_peaks(
    peaks: List[int],
    centers: List[float],
    avg_detections: List[float],
    coalesce_time_thr: float,
) -> List[int]:
    if len(peaks) <= 1:
        return peaks

    peaks_sorted = sorted(peaks, key=lambda i: centers[i])
    final_peaks = []
    current_peak = peaks_sorted[0]

    for i in range(1, len(peaks_sorted)):
        next_peak = peaks_sorted[i]
        dist = centers[next_peak] - centers[current_peak]

        if dist <= coalesce_time_thr:
            if avg_detections[next_peak] > avg_detections[current_peak]:
                current_peak = next_peak
        else:
            final_peaks.append(current_peak)
            current_peak = next_peak

    final_peaks.append(current_peak)
    return final_peaks


def detect_peaks(centers: List[float], avg_detections: List[float], config: Dict[str, float]) -> List[int]:
    if not avg_detections:
        return []

    max_detection = np.max(avg_detections)
    if max_detection <= 0:
        return []

    prominence = config["peak_prominence_factor"] * max_detection
    min_height = config["fraction_of_max"] * max_detection
    peaks, _ = find_peaks(avg_detections, prominence=prominence, height=min_height)

    if peaks.size == 0:
        return []

    return _coalesce_close_peaks(peaks.tolist(), centers, avg_detections, config["coalesce_time_thr"])


__all__ = ["detect_peaks"]
