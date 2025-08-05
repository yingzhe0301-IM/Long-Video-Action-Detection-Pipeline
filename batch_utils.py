# batch_utils.py

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks
from typing import List, Tuple, Dict, Optional, Union, NamedTuple

from config import action_config
from action_detection import (
    group_peaks_adaptive,
    coalesce_close_peaks,
    detect_action_intervals,
    detect_action_intervals_with_end_condition,
    filter_short_intervals,
    filter_low_max_value_intervals
)
from detection_utils import slide_window_average


class PostprocessingResult(NamedTuple):
    evaluation: Dict[str, Union[bool, int]]
    plot_data: Dict[str, Union[List, str]]
    effective_mode: str
    skipped: bool = False


def get_video_files(video_root: Union[str, Path]) -> List[Path]:
    video_root = Path(video_root)
    return list(video_root.rglob("*.mp4"))


def parse_ground_truth_suffix(video_stem: str, action_type: str) -> Optional[int]:
    suffix = video_stem.split('_')[-1]
    if suffix.isdigit():
        if action_type in ['pumping', 'setting']:
            if len(suffix) % 2 == 0:
                return len(suffix) // 2
        else:
            if len(suffix) % 2 == 0:
                return len(suffix)
    return None


def compute_avg_detections(detections_per_frame: List[int], window_size: int, effective_skip: int = 1) -> Tuple[
    List[float], List[float]]:
    actual_window = max(1, window_size // max(effective_skip, 1))
    centers, avg_detections = slide_window_average(detections_per_frame, actual_window)
    return centers, avg_detections


def detect_peaks(centers: List[float], avg_detections: List[float]) -> np.ndarray:
    if len(avg_detections) == 0:
        return np.array([])
    max_detection = np.max(avg_detections)
    peak_prominence = action_config.peak_prominence_factor * max_detection
    min_height_dynamic = action_config.fraction_of_max * max_detection
    peaks, _ = find_peaks(avg_detections, prominence=peak_prominence, height=min_height_dynamic)
    return peaks


def evaluate_detections(video_stem: str, final_actions: list, action_type: str) -> Dict[str, Union[bool, int]]:
    gt_actions = parse_ground_truth_suffix(video_stem, action_type)
    if gt_actions is None:
        return {"skipped": True, "detected_actions": len(final_actions), "gt_actions": 0, "success": False}
    success = (len(final_actions) == gt_actions)
    return {"skipped": False, "detected_actions": len(final_actions), "gt_actions": gt_actions, "success": success}


def plot_detections(plot_data: Dict, plot_folder: Union[str, Path], detection_mode: str,
                    output_subdir: str = "latest") -> str:
    weight_name = Path(plot_data["model_weight"]).stem
    sub_folder = Path(plot_folder) / output_subdir
    sub_folder.mkdir(parents=True, exist_ok=True)
    plot_file = sub_folder / f"{plot_data['video_stem']}_{weight_name}_{detection_mode}_detections.png"
    plt.figure(figsize=(12, 6))
    plt.plot(plot_data["centers"], plot_data["avg_detections"], color='blue', label='Avg Detections')
    if detection_mode == 'interval':
        for i, interval in enumerate(plot_data["final_actions"]):
            start_frame, end_frame = interval
            plt.axvline(x=start_frame, color='red', linestyle='--', label=f'Action{i + 1} Start - {start_frame}')
            plt.axvline(x=end_frame, color='green', linestyle='--', label=f'Action{i + 1} End - {end_frame}')
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
    else:
        for i, idx in enumerate(plot_data["final_actions"]):
            frame_num = int(plot_data["centers"][idx])
            peak_val = plot_data["avg_detections"][idx]
            plt.scatter(frame_num, peak_val, color='red', s=100, marker='x', label=f'Action{i + 1} - {frame_num}')
        if plot_data["final_actions"]:
            plt.legend()
    plt.title(f'Average Detections vs. Frame Number ({plot_data["video_name"]}) - Mode: {detection_mode.capitalize()}')
    plt.xlabel('Window Center (Frame Index)')
    plt.ylabel('Average Detections per Frame')
    plt.grid(True)
    plt.savefig(str(plot_file))
    plt.close()
    return str(plot_file)


def postprocess_detections(
        video_name: str,
        video_stem: str,
        model_weight: str,
        detections_per_frame: List[int],
        window_size: int,
        action_type: str,
        detection_mode: str,
        effective_skip_for_window: int = 1,
        is_fixed: bool = False
) -> PostprocessingResult:
    centers, avg_detections = compute_avg_detections(detections_per_frame, window_size, effective_skip_for_window)
    if not avg_detections:
        return PostprocessingResult(skipped=True, evaluation={}, plot_data={}, effective_mode=detection_mode)

    final_actions = []
    effective_mode = detection_mode

    if action_type == 'setting':
        scaled_min_end_duration = action_config.setting_action_end_duration_threshold / max(effective_skip_for_window,
                                                                                            1)
        potential_intervals = detect_action_intervals_with_end_condition(
            centers,
            avg_detections,
            start_threshold=action_config.setting_action_start_threshold,
            end_threshold=action_config.setting_action_end_threshold,
            min_end_duration=scaled_min_end_duration
        )
        scaled_min_duration = action_config.setting_action_min_duration / max(effective_skip_for_window, 1)
        duration_filtered_intervals = filter_short_intervals(
            potential_intervals, min_duration=scaled_min_duration
        )
        final_actions = filter_low_max_value_intervals(
            duration_filtered_intervals,
            centers,
            avg_detections,
            max_val_threshold=action_config.setting_action_max_val_threshold
        )
        effective_mode = 'interval'
    else:
        # MODIFIED: For 'pumping' and 'catch', allow user to specify mode. Default to 'peak' for 'haul' and 'catch'.
        if action_type in ['pumping', 'catch']:
            effective_mode = detection_mode
        else:  # 'haul' will default to peak
            effective_mode = 'peak'

        if effective_mode == 'interval':
            final_actions = detect_action_intervals(centers, avg_detections)
        else:  # Peak mode
            peaks = detect_peaks(centers, avg_detections)
            if is_fixed:
                local_coalesce_time_thr = action_config.coalesce_time_thr / max(effective_skip_for_window, 1)
            else:
                local_coalesce_time_thr = action_config.coalesce_time_thr
            if peaks.size > 0:
                rep_peaks = group_peaks_adaptive(peaks, centers, avg_detections)
                final_actions = coalesce_close_peaks(rep_peaks, centers, avg_detections,
                                                     coalesce_time_thr=local_coalesce_time_thr)

    evaluation = evaluate_detections(video_stem, final_actions, action_type)
    if evaluation["skipped"]:
        return PostprocessingResult(skipped=True, evaluation=evaluation, plot_data={}, effective_mode=effective_mode)

    evaluation["detected_peaks"] = evaluation.pop("detected_actions")
    evaluation["gt_peaks"] = evaluation.pop("gt_actions")

    plot_data = {
        "centers": centers,
        "avg_detections": avg_detections,
        "final_actions": final_actions,
        "video_name": video_name,
        "model_weight": model_weight,
        "video_stem": video_stem,
    }

    return PostprocessingResult(
        evaluation=evaluation,
        plot_data=plot_data,
        effective_mode=effective_mode,
        skipped=False
    )