
import os
import cv2
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import List, Optional, Union, Callable, Tuple, Generator
import torch
import numpy as np
from detection_utils import process_batch_base

USE_GRAB_RETRIEVE_THRESHOLD = 3

def _make_reader(cap: cv2.VideoCapture, frame_skip: int) -> Callable[[], Generator[Tuple[int, np.ndarray], None, None]]:
    # This function is unchanged
    if frame_skip < USE_GRAB_RETRIEVE_THRESHOLD:
        def _reader() -> Generator[Tuple[int, np.ndarray], None, None]:
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                if frame_skip == 0 or idx % frame_skip == 0:
                    yield idx, frame
        return _reader
    else:
        def _reader() -> Generator[Tuple[int, np.ndarray], None, None]:
            idx = 0
            while cap.grab():
                idx += 1
                if idx % frame_skip == 0:
                    ok, frame = cap.retrieve()
                    if not ok:
                        break
                    yield idx, frame
        return _reader


def process_video_async(
    video_path: Union[str, Path],
    model: object,
    output_dir: str,
    batch_size: int = 32,
    frame_skip: int = 0,
    conf: float = 0.5,
    device: Optional[torch.device] = None,
    display: bool = False,
    save_annotated_frames: bool = True,
    target_class_id: Optional[int] = None, # MODIFIED: Add target_class_id
) -> List[int]:
    """
    Asynchronously process a single video.
    """
    detections_per_frame: List[int] = []
    frames_batch: List[np.ndarray] = []
    frame_numbers_batch: List[int] = []

    frame_queue: Queue = Queue(maxsize=100)
    sentinel = object()

    def read_frames_worker() -> None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Error: cannot open video {video_path}")
        reader = _make_reader(cap, frame_skip)
        for idx, frame in reader():
            frame_queue.put((idx, frame))
        cap.release()
        frame_queue.put(sentinel)

    reader_thread = Thread(target=read_frames_worker, daemon=True)
    reader_thread.start()

    if save_annotated_frames:
        os.makedirs(output_dir, exist_ok=True)

    while True:
        item = frame_queue.get()
        if item is sentinel:
            break

        idx, frame = item
        frames_batch.append(frame)
        frame_numbers_batch.append(idx)

        if len(frames_batch) >= batch_size:
            process_batch_base(
                model, frames_batch, frame_numbers_batch, output_dir,
                detections_per_frame, display=display, conf=conf,
                device=device, save_annotated_frames=save_annotated_frames,
                target_class_id=target_class_id, # MODIFIED: Pass it down
            )
            frames_batch.clear()
            frame_numbers_batch.clear()

    if frames_batch:
        process_batch_base(
            model, frames_batch, frame_numbers_batch, output_dir,
            detections_per_frame, display=display, conf=conf,
            device=device, save_annotated_frames=save_annotated_frames,
            target_class_id=target_class_id, # MODIFIED: Pass it down
        )

    reader_thread.join()
    if display:
        cv2.destroyAllWindows()
    return detections_per_frame