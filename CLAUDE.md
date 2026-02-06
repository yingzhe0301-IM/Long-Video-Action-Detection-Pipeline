# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Haul is a lightweight, reproducible **long-video action detection pipeline** for identifying hauling events in hours-long fishing footage. It uses a Detector-Analyzer architecture: YOLO-based object detection with temporal reasoning (peak detection) to detect and classify actions.

## Build & Development Commands

```bash
# One-click setup (creates haul_env conda environment with Python 3.12)
bash scripts/install_dependencies.sh

# Single video detection demo
python experiments.py --action_type haul --single --frame_skip 5 \
  --model_weight model_weights/haul.pt \
  --video_root video/selected_test_video

# Frame-skip scan (test accuracy vs speed trade-offs)
python experiments.py --action_type haul --scan \
  --min_skip 2 --max_skip 8 \
  --video_root video/selected_test_video

# Custom skip values
python experiments.py --action_type haul --scan \
  --custom_skips "1,2,4,8,16" \
  --video_root video/selected_test_video

# Run tests
python -m pytest tests/
```

Key CLI flags: `--confidence` (YOLO threshold), `--batch_size`, `--window_size`, `--save_frames`, `--no-prefetch`, `--prefetch_batches`

## Architecture

```
experiments.py (CLI entry point)
    ↓
ExperimentConfig (haul/config/experiment_config.py)
    ↓
batch_runner.py (orchestration)
    ├→ inference/video_inference.py (async frame loading + YOLO batch inference)
    └→ post_inference/post_inference.py (signal construction + peak detection)
       └→ post_inference/peak_detection.py (prominence-based peak finding)
```

**Data flow**: Video → async frame decode → YOLO batch prediction → binary detection signal → sliding window average → peak detection with temporal coalescing → evaluation against ground truth

**Ground truth encoding**: Video filename suffix encodes expected action count (e.g., `video_1111.mp4` → 2 hauls based on digit parsing)

## Key Configuration Files

- `haul/config/experiment_config.py`: Experiment defaults (frame_skip=5, confidence=0.5, batch_size=64, window_size=100)
- `haul/config/unified_config.py`: Peak detection parameters (prominence_factor=0.05, fraction_of_max=0.3, coalesce_time_thr=1200.0) - scaled by frame_skip

## Device Support

Auto-detects CUDA/MPS/CPU via `haul/inference/inference_utils.py`. macOS uses MPS (Metal Performance Shaders), Linux defaults to CPU (CUDA if available).

## Output Artifacts

- `plot/latest/`: Detection timeline plots (PNG)
- `plot/results/`: Scan CSVs (frame_skip vs accuracy/runtime)
- `plot/output_frames_<video>/`: Annotated frames (with `--save_frames`)

## Code Style

PEP 8, 4-space indentation, snake_case functions, PascalCase classes. Explicit type hints on public APIs. Concise docstrings (see `inference_utils.py` for examples).
