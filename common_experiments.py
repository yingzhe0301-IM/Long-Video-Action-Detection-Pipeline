# common_experiments.py

import time
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
from detection_utils import get_device, load_yolo_model
from video_batch_processor_unified import process_all_videos_unified
# REFACTORED: Import the new configuration class
from experiment_config import ExperimentConfig


def _prepare_env_and_model(
        config: ExperimentConfig
) -> Tuple[Optional[object], ExperimentConfig]:
    """
    Prepare environment and model, and update the config object.
    Returns the loaded model (or None for MP) and the updated config.
    """
    config.device = get_device()
    print(f"Using device: {config.device}")

    config.video_path = Path(config.video_root) / config.action_type
    print(f"Action Type: '{config.action_type}' -> Model: '{config.model_weight}', Videos: '{config.video_path}'")

    model = None
    if not config.use_multiprocessing:
        # Load model only in the main process for single-threaded mode
        model = load_yolo_model(config.model_weight, device=config.device, confidence=config.confidence)

    config.plot_path = Path(config.plot_folder)
    return model, config


def _save_results(
        df: pd.DataFrame,
        plot_path: Path,
        filename: str
) -> None:
    results_dir = plot_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_file = results_dir / filename
    df.to_csv(csv_file, index=False)


def _execute_fixed_skip_experiments(
        config: ExperimentConfig,
        model: Optional[object],
        frame_skip_values: List[int],
        tag: str,
) -> pd.DataFrame:
    single = len(frame_skip_values) == 1
    results = []

    for fs in frame_skip_values:
        if not single:
            idx = frame_skip_values.index(fs) + 1
            print(f"\n=== ({idx}/{len(frame_skip_values)}) {tag} frame_skip = {fs} ===")

        # REFACTORED: Update the config with the current frame skip value
        current_run_config = config
        current_run_config.frame_skip = fs

        start = time.time()
        # REFACTORED: Pass the config object and model down
        accuracy = process_all_videos_unified(
            current_run_config,
            model=model,
        )
        runtime = time.time() - start

        result_data = {
            "frame_skip": fs,
            "accuracy": round(accuracy, 2),
            "runtime": round(runtime, 2),
            "detection_mode": config.detection_mode,
        }

        if config.use_multiprocessing:
            result_data.update({"use_multiprocessing": True, "num_workers": config.num_workers})
        else:
            result_data.update({"use_multiprocessing": False, "num_workers": 1})

        results.append(result_data)

    return pd.DataFrame(results)


def run_single_fixed_experiment(config: ExperimentConfig) -> None:
    """
    REFACTORED: This function now takes a single ExperimentConfig object.
    """
    model, updated_config = _prepare_env_and_model(config)

    df = _execute_fixed_skip_experiments(
        config=updated_config,
        model=model,
        frame_skip_values=[updated_config.frame_skip],
        tag="[Fixed Skip]",
    )
    mode_suffix = f"_mp{updated_config.num_workers}" if updated_config.use_multiprocessing else "_single"
    filename = f"{updated_config.action_type}_{updated_config.detection_mode}_fixed_single_skip{updated_config.frame_skip}{mode_suffix}.csv"
    _save_results(df, updated_config.plot_path, filename)


def run_fixed_frame_skip_scan(config: ExperimentConfig) -> None:
    """
    REFACTORED: This function now takes a single ExperimentConfig object.
    """
    model, updated_config = _prepare_env_and_model(config)
    skips = updated_config.custom_skips if updated_config.custom_skips is not None else list(
        range(updated_config.min_skip, updated_config.max_skip + 1))
    print(f"Testing skips: {skips}")

    df = _execute_fixed_skip_experiments(
        config=updated_config,
        model=model,
        frame_skip_values=skips,
        tag="[Fixed Skip Scan]",
    )
    mode_suffix = f"_mp{updated_config.num_workers}" if updated_config.use_multiprocessing else "_single"
    filename = f"{updated_config.action_type}_{updated_config.detection_mode}_frame_skip_scan{mode_suffix}.csv"
    _save_results(df, updated_config.plot_path, filename)