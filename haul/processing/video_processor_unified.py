"""Video processing with optional async frame prefetching."""

import os
from pathlib import Path
from queue import Queue
from threading import Thread, Event
from typing import Any, Dict, List, Optional, Union

import cv2
import torch

from haul.detection.detection_utils import supports_half_precision


def _decode_frames(video_path: str, frame_skip: int, batch_size: int, queue: Queue, stop: Event):
    """Worker thread: decode video frames into batches."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        queue.put(None)
        return

    try:
        idx = 0
        frames, numbers = [], []

        while not stop.is_set():
            ok, frame = cap.read()
            if not ok:
                break

            idx += 1
            if frame_skip > 1 and idx % frame_skip != 0:
                continue

            frames.append(frame)
            numbers.append(idx)

            if len(frames) >= batch_size:
                queue.put((frames.copy(), numbers.copy()))
                frames.clear()
                numbers.clear()

        if frames:
            queue.put((frames, numbers))
    finally:
        cap.release()
        queue.put(None)


def _run_inference(model, frames: List, frame_numbers: List[int], conf: float,
                   device, save_dir: Optional[str], display: bool) -> Dict[str, Any]:
    """Run YOLO inference on a batch of frames."""
    if not frames:
        return {"detections": [], "stop": False}

    results = model.predict(
        frames, conf=conf, device=device, verbose=False,
        half=supports_half_precision(device)
    )

    detections = []
    stop_requested = False

    for i, result in enumerate(results):
        if save_dir:
            cv2.imwrite(f"{save_dir}/annotated_frame_{frame_numbers[i]:05d}.jpg", result.plot())

        detections.append(len(result.boxes) if result.boxes else 0)

        if display:
            cv2.imshow("Detections", result.plot())
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_requested = True
                break

    return {"detections": detections, "stop": stop_requested}


def process_video(
    model,
    video_path: Union[str, Path],
    output_dir: str,
    *,
    batch_size: int,
    frame_skip: int,
    conf: float,
    device: Optional[torch.device],
    display: bool,
    save_annotated_frames: bool,
    prefetch_batches: int = 2,
    use_async: bool = True,
) -> List[int]:
    """
    Process video with optional async frame prefetching.

    Args:
        use_async: If True, decode frames in background thread (faster).
                   If False, use synchronous decoding (for comparison).
    """
    if save_annotated_frames:
        os.makedirs(output_dir, exist_ok=True)

    save_dir = output_dir if save_annotated_frames else None
    all_detections: List[int] = []

    if use_async:
        # Async mode: decode in background thread
        queue: Queue = Queue(maxsize=prefetch_batches)
        stop = Event()
        thread = Thread(target=_decode_frames, daemon=True,
                       args=(str(video_path), frame_skip or 1, batch_size, queue, stop))
        thread.start()

        try:
            while True:
                batch = queue.get(timeout=30)
                if batch is None:
                    break

                frames, numbers = batch
                result = _run_inference(model, frames, numbers, conf, device, save_dir, display)
                all_detections.extend(result["detections"])

                if result["stop"]:
                    stop.set()
                    break
        finally:
            stop.set()
            thread.join(timeout=2)
    else:
        # Sync mode: decode in main thread
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        try:
            idx = 0
            frames, numbers = [], []

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                idx += 1
                if (frame_skip or 1) > 1 and idx % frame_skip != 0:
                    continue

                frames.append(frame)
                numbers.append(idx)

                if len(frames) >= batch_size:
                    result = _run_inference(model, frames, numbers, conf, device, save_dir, display)
                    all_detections.extend(result["detections"])
                    frames.clear()
                    numbers.clear()

                    if result["stop"]:
                        break

            if frames:
                result = _run_inference(model, frames, numbers, conf, device, save_dir, display)
                all_detections.extend(result["detections"])
        finally:
            cap.release()

    if display:
        cv2.destroyAllWindows()

    return all_detections