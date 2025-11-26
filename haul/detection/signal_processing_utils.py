# signal_processing_utils.py
"""
General-purpose signal processing utilities
These are building blocks that can be used by any detector
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


# ========== Interval Detection ==========

def detect_action_intervals(
        centers: List[float],
        avg_detections: List[float],
        threshold: float = 0.0,
        max_gap: int = 0,
) -> List[Tuple[int, int]]:
    """
    Detect continuous intervals where signal > threshold, allowing short gaps.

    Args:
        centers: Frame indices aligned with `avg_detections`.
        avg_detections: Sliding window averages for the signal.
        threshold: Minimum value that counts as active.
        max_gap: Maximum number of frames the signal may dip below the
            threshold without breaking the interval. Use 0 to disable gap
            bridging.

    Returns:
        List of (start_frame, end_frame) tuples.
    """
    if not centers or not avg_detections:
        return []

    detected_intervals = []
    in_interval = False
    start_frame = 0
    last_active_idx: Optional[int] = None
    max_gap = max(int(max_gap or 0), 0)

    for i, value in enumerate(avg_detections):
        frame = int(centers[i])
        is_above_threshold = value > threshold

        if is_above_threshold:
            last_active_idx = i
            if not in_interval:
                in_interval = True
                start_frame = frame
        elif in_interval:
            if max_gap > 0 and last_active_idx is not None:
                gap = frame - int(centers[last_active_idx])
                if gap <= max_gap:
                    continue

            end_idx = last_active_idx if last_active_idx is not None else max(i - 1, 0)
            end_frame = int(centers[end_idx])
            detected_intervals.append((start_frame, end_frame))
            in_interval = False
            last_active_idx = None

    if in_interval:
        end_idx = last_active_idx if last_active_idx is not None else len(centers) - 1
        end_frame = int(centers[end_idx])
        detected_intervals.append((start_frame, end_frame))

    return detected_intervals


def detect_action_intervals_with_end_condition(
        centers: List[float],
        avg_detections: List[float],
        start_threshold: float,
        end_threshold: float,
        min_end_duration: int
) -> List[Tuple[int, int]]:
    """
    Advanced interval detection with separate start/end conditions

    Used by: setting detector
    """
    if not centers or not avg_detections:
        return []

    intervals = []
    in_interval = False
    start_frame_candidate = 0
    potential_end_index = -1

    i = 0
    while i < len(avg_detections):
        value = avg_detections[i]

        if not in_interval:
            if value > start_threshold:
                in_interval = True
                start_frame_candidate = int(centers[i])
                potential_end_index = -1
        else:
            if value <= end_threshold:
                if potential_end_index == -1:
                    potential_end_index = i

                start_check_frame = centers[potential_end_index]
                current_check_frame = centers[i]

                if (current_check_frame - start_check_frame) >= min_end_duration:
                    end_frame = int(centers[potential_end_index])
                    intervals.append((start_frame_candidate, end_frame))
                    in_interval = False
                    potential_end_index = -1
            else:
                potential_end_index = -1
        i += 1

    if in_interval:
        end_frame = int(centers[-1])
        intervals.append((start_frame_candidate, end_frame))

    return intervals


# ========== Filtering Functions ==========

def filter_short_intervals(
        intervals: List[Tuple[int, int]],
        min_duration: int
) -> List[Tuple[int, int]]:
    """Filter out intervals shorter than min_duration"""
    if not intervals:
        return []

    return [
        (start, end) for start, end in intervals
        if (end - start) >= min_duration
    ]


def filter_low_max_value_intervals(
        intervals: List[Tuple[int, int]],
        centers: List[float],
        avg_detections: List[float],
        max_val_threshold: float
) -> List[Tuple[int, int]]:
    """Filter intervals where peak value is below threshold"""
    if not intervals:
        return []

    valid_intervals = []
    centers_arr = np.array(centers)

    for start_frame, end_frame in intervals:
        start_idx = np.searchsorted(centers_arr, start_frame, side='left')
        end_idx = np.searchsorted(centers_arr, end_frame, side='right')

        if start_idx < end_idx:
            max_val_in_interval = np.max(avg_detections[start_idx:end_idx])

            if max_val_in_interval >= max_val_threshold:
                valid_intervals.append((start_frame, end_frame))

    return valid_intervals


# ========== Peak Clustering ==========

def split_cluster_by_valley(
        cluster_peaks: List[int],
        centers: List[float],
        avg_detections: List[float],
        config: Dict[str, float]
) -> List[List[int]]:
    """
    Split peak cluster by valleys between peaks

    Used by: peak clustering algorithms
    """
    if len(cluster_peaks) <= 1:
        return [cluster_peaks]

    valley_ratio = config.get('valley_ratio', 0.7)
    sub_clusters = []
    current_sub = [cluster_peaks[0]]

    for i in range(len(cluster_peaks) - 1):
        p1 = cluster_peaks[i]
        p2 = cluster_peaks[i + 1]
        amp1 = avg_detections[p1]
        amp2 = avg_detections[p2]
        left_idx = min(p1, p2)
        right_idx = max(p1, p2)
        local_min = np.min(avg_detections[left_idx:right_idx + 1])
        threshold_val = valley_ratio * min(amp1, amp2)

        if local_min < threshold_val:
            if cluster_peaks[i] not in current_sub:
                current_sub.append(cluster_peaks[i])
            sub_clusters.append(current_sub)
            current_sub = [p2]
        else:
            if p2 not in current_sub:
                current_sub.append(p2)

    if len(current_sub) > 0:
        sub_clusters.append(current_sub)

    return sub_clusters


def group_peaks_adaptive(
        peaks: np.ndarray,
        centers: List[float],
        avg_detections: List[float],
        config: Optional[Dict[str, float]] = None
) -> List[int]:
    """
    Group peaks using adaptive distance threshold

    Used by: pumping detector (peak mode)
    Args:
        config: Dictionary with 'adaptive_factor' and 'valley_ratio'
    """
    if len(peaks) <= 1:
        return peaks.tolist()

    config = config or {}
    adaptive_factor = config.get('adaptive_factor', 2.0)

    sorted_peaks = sorted(peaks, key=lambda i: centers[i])

    # Calculate adaptive threshold
    distances = []
    for i in range(1, len(sorted_peaks)):
        prev_center = centers[sorted_peaks[i - 1]]
        curr_center = centers[sorted_peaks[i]]
        distances.append(curr_center - prev_center)

    median_dist = np.median(distances) if len(distances) > 0 else 0
    threshold = adaptive_factor * median_dist

    # Group peaks
    clusters = []
    current_cluster = [sorted_peaks[0]]

    for i in range(1, len(sorted_peaks)):
        prev_center = centers[sorted_peaks[i - 1]]
        curr_center = centers[sorted_peaks[i]]
        if (curr_center - prev_center) <= threshold:
            current_cluster.append(sorted_peaks[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [sorted_peaks[i]]

    if current_cluster:
        clusters.append(current_cluster)

    # Process clusters
    final_clusters = []
    for cluster_peaks in clusters:
        if len(cluster_peaks) == 1:
            final_clusters.append(cluster_peaks)
        else:
            sub_clusters = split_cluster_by_valley(cluster_peaks, centers, avg_detections, config)
            final_clusters.extend(sub_clusters)

    # Select representative peaks
    representative_peaks = []
    for c in final_clusters:
        best_idx = max(c, key=lambda i: avg_detections[i])
        representative_peaks.append(best_idx)

    return representative_peaks


def coalesce_close_peaks(
        rep_peaks: List[int],
        centers: List[float],
        avg_detections: List[float],
        coalesce_time_thr: float
) -> List[int]:
    """
    Merge peaks that are too close together

    Used by: pumping detector (peak mode)
    """
    if len(rep_peaks) <= 1:
        return rep_peaks

    rep_peaks_sorted = sorted(rep_peaks, key=lambda i: centers[i])
    final_actions = []
    current_peak = rep_peaks_sorted[0]

    for i in range(1, len(rep_peaks_sorted)):
        next_peak = rep_peaks_sorted[i]
        dist = centers[next_peak] - centers[current_peak]

        if dist <= coalesce_time_thr:
            if avg_detections[next_peak] > avg_detections[current_peak]:
                current_peak = next_peak
        else:
            final_actions.append(current_peak)
            current_peak = next_peak

    final_actions.append(current_peak)
    return final_actions
