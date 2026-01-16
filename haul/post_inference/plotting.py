"""Plotting helpers for post-inference analysis."""

from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_results(result: Dict[str, Any], plot_folder: Path, model_weight: str, video_name: str) -> str:
    """Generate and save detection plot."""
    sub_folder = plot_folder / "latest"
    sub_folder.mkdir(parents=True, exist_ok=True)

    mode = result.get("detection_mode", "peak")
    weight_name = Path(model_weight).stem
    plot_file = sub_folder / f"{result.get('video_stem', video_name)}_{weight_name}_{mode}.png"

    fig, ax = plt.subplots(figsize=(10, 5))
    centers = result.get("centers", [])
    averages = result.get("avg_detections", [])
    ax.plot(centers, averages, label="Avg detections")

    for idx in result.get("final_actions", []):
        if 0 <= idx < len(centers):
            ax.scatter(centers[idx], averages[idx], color="#d62728")

    ax.set_title(f"{result.get('action_type', 'Action').title()} Detection - {video_name}")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Average detections")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_file, dpi=100)
    plt.close(fig)
    return str(plot_file)
