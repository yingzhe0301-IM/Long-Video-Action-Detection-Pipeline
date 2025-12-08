"""Video processing with optional async frame prefetching."""

import os
from pathlib import Path
from queue import Queue
from threading import Event, Thread
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import cv2
import torch

from haul.detection.detection_utils import supports_half_precision


def _decode_frames(video_path: str, frame_skip: int, batch_size: int, queue: Queue, stop: Event) -> None:
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


def _run_inference(
    model,
    frames: List,
    frame_numbers: List[int],
    conf: float,
    device,
    display: bool,
    save_dir: Optional[str],
) -> Dict[str, Any]:
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


def _process_video_async(
    model,
    video_path: Union[str, Path],
    *,
    batch_size: int,
    frame_skip: int,
    conf: float,
    device: Optional[torch.device],
    display: bool,
    save_dir: Optional[str],
    prefetch_batches: int,
) -> List[int]:
    """Decode frames on a background thread and run inference in batches."""
    queue: Queue = Queue(maxsize=prefetch_batches)
    stop = Event()
    thread = Thread(
        target=_decode_frames,
        daemon=True,
        args=(str(video_path), frame_skip or 1, batch_size, queue, stop),
    )
    thread.start()

    def _batch_iter() -> Iterable[Tuple[List, List[int]]]:
        try:
            while True:
                batch = queue.get(timeout=30)
                if batch is None:
                    break
                yield batch
        finally:
            stop.set()
            thread.join(timeout=2)

    return _process_batch_stream(
        _batch_iter(),
        model,
        conf=conf,
        device=device,
        display=display,
        save_dir=save_dir,
        stop_callback=stop.set,
    )


def _process_video_sync(
    model,
    video_path: Union[str, Path],
    *,
    batch_size: int,
    frame_skip: int,
    conf: float,
    device: Optional[torch.device],
    display: bool,
    save_dir: Optional[str],
) -> List[int]:
    """Decode frames on the main thread and run inference in batches."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    def _batch_iter() -> Iterable[Tuple[List, List[int]]]:
        frames: List = []
        numbers: List[int] = []
        idx = 0
        try:
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
                    yield frames, numbers
                    frames, numbers = [], []

            if frames:
                yield frames, numbers
        finally:
            cap.release()

    return _process_batch_stream(
        _batch_iter(),
        model,
        conf=conf,
        device=device,
        display=display,
        save_dir=save_dir,
    )


def _process_batch_stream(
    batches: Iterable[Tuple[List, List[int]]],
    model,
    *,
    conf: float,
    device: Optional[torch.device],
    display: bool,
    save_dir: Optional[str],
    stop_callback: Optional[Callable[[], None]] = None,
) -> List[int]:
    """Consume batches of frames, run inference, and collect detections."""
    all_detections: List[int] = []

    for frames, numbers in batches:
        result = _run_inference(model, frames, numbers, conf, device, display, save_dir)
        all_detections.extend(result["detections"])

        if result["stop"]:
            if stop_callback:
                stop_callback()
            break

    return all_detections


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

    if use_async:
        detections = _process_video_async(
            model,
            video_path,
            batch_size=batch_size,
            frame_skip=frame_skip,
            conf=conf,
            device=device,
            display=display,
            save_dir=save_dir,
            prefetch_batches=prefetch_batches,
        )
    else:
        detections = _process_video_sync(
            model,
            video_path,
            batch_size=batch_size,
            frame_skip=frame_skip,
            conf=conf,
            device=device,
            display=display,
            save_dir=save_dir,
        )

    if display:
        cv2.destroyAllWindows()

    return detections
