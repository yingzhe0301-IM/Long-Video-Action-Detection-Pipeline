# video_batch_processor_unified.py
from pathlib import Path
import time
import torch.multiprocessing as mp
from tqdm import tqdm
from typing import Union, Dict, Any, Optional, Tuple
import torch

from batch_utils import get_video_files, postprocess_detections, plot_detections, PostprocessingResult
from detection_utils import load_yolo_model
from video_processor_async import process_video_async
from experiment_config import ExperimentConfig


def process_single_video_for_mp(args: Tuple) -> Dict[str, Any]:
    video_path, config, process_idx = args

    device = torch.device(config.device)
    model = load_yolo_model(config.model_weight, device=device, confidence=config.confidence)
    video_name = video_path.name
    video_stem = video_path.stem
    output_dir = Path(config.plot_folder) / f"output_frames_{video_stem}"

    # MODIFIED: Pass target_class_id from config
    detections_per_frame = process_video_async(
        str(video_path), model, str(output_dir),
        batch_size=config.batch_size, frame_skip=config.frame_skip,
        conf=config.confidence, device=device, display=config.display,
        save_annotated_frames=config.save_frames,
        target_class_id=config.target_class_id
    )

    post_result = postprocess_detections(
        video_name=video_name,
        video_stem=video_stem,
        model_weight=config.model_weight,
        detections_per_frame=detections_per_frame,
        window_size=config.window_size,
        action_type=config.action_type,
        detection_mode=config.detection_mode,
        effective_skip_for_window=config.frame_skip,
        is_fixed=True
    )

    if post_result.skipped:
        print(f"[Process {process_idx}] {video_name}: SKIPPED (cannot parse GT or no detections)")
        return {"skipped": True, "video_name": video_name, "success": False}

    plot_path = plot_detections(
        plot_data=post_result.plot_data,
        plot_folder=config.plot_folder,
        detection_mode=post_result.effective_mode,
        output_subdir="latest"
    )

    eval_dict = post_result.evaluation
    detected_peaks = eval_dict["detected_peaks"]
    gt_peaks = eval_dict["gt_peaks"]
    success = eval_dict["success"]

    result = {
        "skipped": False, "video_name": video_name, "detected_peaks": detected_peaks,
        "gt_peaks": gt_peaks, "success": success, "plot_path": plot_path
    }
    print(
        f"[Process {process_idx}] {video_name}: DETECTED={detected_peaks} | GT={gt_peaks} | {'SUCCESS' if success else 'FAIL'}")
    return result


def process_all_videos_async(config: ExperimentConfig, model: object) -> float:
    start_total = time.time()
    video_files = get_video_files(config.video_path)
    latest_folder = Path(config.plot_path) / "latest"
    latest_folder.mkdir(parents=True, exist_ok=True)
    total_videos, success_count = 0, 0

    for video_path in video_files:
        video_name = video_path.name
        video_stem = video_path.stem
        output_dir = Path(config.plot_path) / f"output_frames_{video_stem}"

        # MODIFIED: Pass target_class_id from config
        detections_per_frame = process_video_async(
            str(video_path), model, str(output_dir),
            batch_size=config.batch_size, frame_skip=config.frame_skip,
            conf=config.confidence, device=config.device, display=config.display,
            save_annotated_frames=config.save_frames,
            target_class_id=config.target_class_id
        )

        post_result = postprocess_detections(
            video_name=video_name,
            video_stem=video_stem,
            model_weight=config.model_weight,
            detections_per_frame=detections_per_frame,
            window_size=config.window_size,
            action_type=config.action_type,
            detection_mode=config.detection_mode,
            effective_skip_for_window=config.frame_skip,
            is_fixed=True
        )

        if post_result.skipped:
            print(f"Warning: cannot parse ground truth from '{video_name}'. Skipping this video.")
            continue

        plot_detections(
            plot_data=post_result.plot_data,
            plot_folder=config.plot_folder,
            detection_mode=post_result.effective_mode,
            output_subdir="latest"
        )

        eval_dict = post_result.evaluation
        detected_peaks = eval_dict["detected_peaks"]
        gt_peaks = eval_dict["gt_peaks"]
        if eval_dict["success"]:
            success_count += 1
            print(f"[{video_name}] DETECTED={detected_peaks} | GT={gt_peaks} | SUCCESS")
        else:
            print(f"[{video_name}] DETECTED={detected_peaks} | GT={gt_peaks} | FAIL")
        total_videos += 1

    print("\nAll videos processed successfully!")
    total_time = time.time() - start_total
    if total_videos > 0:
        accuracy = success_count / total_videos * 100
        print(f"Video-Level Accuracy: {success_count}/{total_videos} = {accuracy:.2f}%, done in {total_time:.2f}s")
    else:
        accuracy = 0.0
        print("No valid videos with parsable ground truth suffix were found.")
    return accuracy


# ... the rest of the file (process_all_videos_mp, process_all_videos_unified) is unchanged ...
def process_all_videos_mp(config: ExperimentConfig) -> float:
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    video_files = get_video_files(config.video_path)
    latest_folder = Path(config.plot_path) / "latest"
    latest_folder.mkdir(parents=True, exist_ok=True)

    process_args = []
    for i, video_path in enumerate(video_files):
        process_args.append((video_path, config, i))

    start_time = time.time()
    print(f"Processing {len(video_files)} videos with {config.num_workers} workers using torch.multiprocessing...")
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=config.num_workers) as pool:
        results = list(
            tqdm(pool.imap(process_single_video_for_mp, process_args), total=len(process_args), desc="Progress"))

    processing_time = time.time() - start_time
    print(f"\nAll videos processed in {processing_time:.2f} seconds!")

    total_videos, success_count = 0, 0
    for result in results:
        if not result["skipped"]:
            total_videos += 1
            if result["success"]:
                success_count += 1

    if total_videos > 0:
        accuracy = success_count / total_videos * 100
        print(f"Video-Level Accuracy: {success_count}/{total_videos} = {accuracy:.2f}%")
    else:
        accuracy = 0.0
        print("No valid videos with parsable ground truth suffix were found.")
    return accuracy


def process_all_videos_unified(
        config: ExperimentConfig,
        model: Optional[object],
) -> float:
    if config.use_multiprocessing:
        return process_all_videos_mp(config)
    else:
        return process_all_videos_async(config, model)