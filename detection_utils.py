# detection_utils.py

import os
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional, Union


def get_device() -> torch.device:
    """
    Return the highest‑priority available torch.device in the order:
    1. CUDA
    2. MPS
    3. CPU
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def load_yolo_model(model_weight: str, device: Optional[torch.device] = None, confidence: float = 0.5) -> YOLO:
    # This function remains the same, but the model it loads might be a pre-trained one.
    device = get_device()
    model = YOLO(model_weight).to(device)
    return model


def process_batch_base(model: YOLO, frames_batch: List[np.ndarray], frame_numbers_batch: List[int],
                       output_dir: str, detections_per_frame: List[int], display: bool = False,
                       conf: float = 0.5, device: Optional[torch.device] = None,
                       save_annotated_frames: bool = True,
                       target_class_id: Optional[int] = None) -> None:  # MODIFIED: Add target_class_id
    """
    Process a batch of frames:
      - Run inference and save annotated frames (if needed)
      - Extract the number of detections per frame for a specific class if provided.
    """
    if not frames_batch:
        return

    # MODIFIED: Add the 'classes' argument to the predict call if a target_class_id is specified.
    predict_args = {
        "conf": conf,
        "device": device,
        "verbose": False,
        "half": True
    }
    if target_class_id is not None:
        predict_args["classes"] = [target_class_id]

    results = model.predict(frames_batch, **predict_args)

    for i, result in enumerate(results):
        # We plot the frame with all detections for context, even if we only count one class.
        annotated_frame = result.plot()
        frame_idx = frame_numbers_batch[i]

        if save_annotated_frames:
            out_path = os.path.join(output_dir, f"annotated_frame_{frame_idx:05d}.jpg")
            cv2.imwrite(out_path, annotated_frame)

        if display:
            cv2.imshow("YOLO Inference", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # The number of detections is now implicitly filtered by the predict call
        detections_per_frame.append(len(result.boxes))


def slide_window_average(data: List[Union[int, float]], window_size: int) -> Tuple[List[float], List[float]]:
    """
    Apply sliding window average to data list and return (window center positions, average values).
    """
    n = len(data)
    if n < window_size:
        return [], []

    averages = []
    centers = []
    for i in range(n - window_size + 1):
        window = data[i:i + window_size]
        avg = sum(window) / window_size
        averages.append(avg)
        centers.append(i + window_size // 2)

    return centers, averages


def plot_data(x: List[Union[int, float]], y: List[Union[int, float]], title: str,
              xlabel: str, ylabel: str, save_path: str, plot_type: str = 'scatter') -> None:
    """
    Plot and save figure, can choose scatter plot or line plot.
    """
    plt.figure(figsize=(12, 6))
    if plot_type == 'scatter':
        plt.scatter(x, y, alpha=0.5)
    else:
        plt.plot(x, y, color='blue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()