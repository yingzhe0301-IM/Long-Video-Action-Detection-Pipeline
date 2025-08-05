# action_detection.py

import numpy as np
from typing import List, Tuple
from config import action_config


def detect_action_intervals(
        centers: List[float],
        avg_detections: List[float],
        threshold: float = 0.0
) -> List[Tuple[int, int]]:
    """
    Detects all continuous action intervals where avg_detections > threshold.
    """
    if not centers or not avg_detections:
        return []

    detected_intervals = []
    in_interval = False
    start_frame = 0

    for i, value in enumerate(avg_detections):
        is_above_threshold = value > threshold

        if is_above_threshold and not in_interval:
            in_interval = True
            start_frame = int(centers[i])
        elif not is_above_threshold and in_interval:
            in_interval = False
            end_frame = int(centers[i - 1])
            detected_intervals.append((start_frame, end_frame))

    if in_interval:
        end_frame = int(centers[-1])
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
    NEW: Detects action intervals using separate start/end thresholds and an end duration condition.
    - An interval STARTS when avg_detections > start_threshold.
    - An interval ENDS when avg_detections < end_threshold for at least min_end_duration frames.
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
            # Condition to start a new interval
            if value > start_threshold:
                in_interval = True
                start_frame_candidate = int(centers[i])
                potential_end_index = -1
        else:
            # Condition to check for the end of the current interval
            if value <= end_threshold:
                if potential_end_index == -1:
                    # Mark the beginning of a potential end
                    potential_end_index = i

                # Check if the duration of being below threshold is sufficient
                start_check_frame = centers[potential_end_index]
                current_check_frame = centers[i]

                if (current_check_frame - start_check_frame) >= min_end_duration:
                    # End condition met, finalize the interval
                    end_frame = int(centers[potential_end_index])
                    intervals.append((start_frame_candidate, end_frame))
                    in_interval = False
                    potential_end_index = -1
            else:
                # Value went back up, reset the potential end
                potential_end_index = -1
        i += 1

    # If the video ends while still in an interval, close the interval at the last frame
    if in_interval:
        end_frame = int(centers[-1])
        intervals.append((start_frame_candidate, end_frame))

    return intervals


def filter_short_intervals(
        intervals: List[Tuple[int, int]],
        min_duration: int
) -> List[Tuple[int, int]]:
    """
    Filters out action intervals that are shorter than min_duration.
    """
    if not intervals:
        return []

    return [
        (start, end) for start, end in intervals if (end - start) >= min_duration
    ]


def filter_low_max_value_intervals(
        intervals: List[Tuple[int, int]],
        centers: List[float],
        avg_detections: List[float],
        max_val_threshold: float
) -> List[Tuple[int, int]]:
    """
    Filters out intervals where the peak avg_detection value is below a threshold.
    """
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


# ... (The rest of the file: split_cluster_by_valley, group_peaks_adaptive, etc. remains unchanged)
def split_cluster_by_valley(cluster_peaks: List[int], centers: List[float], avg_detections: List[float]) -> List[
    List[int]]:
    if len(cluster_peaks) <= 1:
        return [cluster_peaks]
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
        threshold_val = action_config.valley_ratio * min(amp1, amp2)
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


def group_peaks_adaptive(peaks: np.ndarray, centers: List[float], avg_detections: List[float]) -> List[int]:
    if len(peaks) <= 1:
        return peaks.tolist()
    sorted_peaks = sorted(peaks, key=lambda i: centers[i])
    distances = []
    for i in range(1, len(sorted_peaks)):
        prev_center = centers[sorted_peaks[i - 1]]
        curr_center = centers[sorted_peaks[i]]
        distances.append(curr_center - prev_center)
    median_dist = np.median(distances) if len(distances) > 0 else 0
    threshold = action_config.adaptive_factor * median_dist
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
    final_clusters = []
    for cluster_peaks in clusters:
        if len(cluster_peaks) == 1:
            final_clusters.append(cluster_peaks)
        else:
            sub_clusters = split_cluster_by_valley(cluster_peaks, centers, avg_detections)
            final_clusters.extend(sub_clusters)
    representative_peaks = []
    for c in final_clusters:
        best_idx = max(c, key=lambda i: avg_detections[i])
        representative_peaks.append(best_idx)
    return representative_peaks


def coalesce_close_peaks(rep_peaks: List[int], centers: List[float], avg_detections: List[float],
                         coalesce_time_thr: float) -> List[int]:
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