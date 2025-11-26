# Repository Guidelines

## Project Structure & Module Organization
Core orchestration lives at the repository root: `experiments.py` loads YOLO checkpoints, runs batch processing, and evaluates detections. Supporting modules now sit inside the `haul/` package (for example `haul/processing/batch_processor_simplified.py`, `haul/actions/action_registry.py`, and `haul/actions/action_evaluator.py`) to keep batching, heuristics, and scoring concerns organised. Model assets belong in `model_weights/`, demo clips in `selected_test_video/`, and generated plots or CSVs in `plot/`. Keep exploratory notebooks (`haul.ipynb`) and staging folders (`history_data/`, `new_video/`) isolated from production edits.

## Build, Test, and Development Commands
Run `bash install_dependencies.sh` once to create the `haul_env` Conda environment with Python 3.12 and the expected PyTorch build. Use `python experiments.py --action_type pumping --single --frame_skip 5 --video_root selected_test_video --model_weight model_weights/pumping.pt` for the default demo, which writes outputs to `plot/latest/`. Sweep frame-skip options with `python experiments.py --action_type haul --scan --min_skip 2 --max_skip 8` to populate comparison CSVs under `plot/results/`.

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indentation, snake_case functions, and PascalCase classes. Keep modules narrowly focused and inject dependencies rather than relying on globals. Write concise docstrings mirroring `detection_utils.py`, and prefer explicit type hints on new public APIs.

## Testing Guidelines
Add focused tests under `tests/` and run them with `python -m pytest`. Mock model outputs or reuse sample clips to keep suites lightweight. After code changes, review metrics and artifacts in `plot/latest/` to confirm expected behavior.

## Commit & Pull Request Guidelines
Author commits in a short, imperative style (e.g., `Refine README wording`) and limit each commit to related changes. Pull requests should explain motivation, summarize functional impact, list executed commands, and link tracking issues. Include representative plots or metrics when detection quality shifts, and note any new model weights.

## Data & Configuration Tips
Store new checkpoints in `model_weights/` with action-specific filenames and document pairings in `haul/actions/action_registry.py`. Update `haul/config/experiment_config.py` when default behaviors shift, and keep environment overrides in `.env` or private configs rather than editing shared settings.
