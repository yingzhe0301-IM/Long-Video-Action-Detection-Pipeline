"""Video inference with optional async frame prefetching."""

import os
import time
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import cv2
import torch

from .adaptive_sampler import AdaptiveFrameSampler
from .inference_utils import supports_half_precision
from .pipeline_profiler import PipelineProfiler, SyncProfiler


def _decode_frames(
    video_path: str,
    frame_skip: int,
    batch_size: int,
    queue: Queue,
    stop: Event,
    producer_done: Event,
    decode_all: bool = False,
) -> None:
    """Worker thread: decode video frames into batches."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        producer_done.set()
        queue.put(None)
        return

    try:
        idx = 0
        frames, numbers = [], []
        batch_start = None

        while not stop.is_set():
            if decode_all:
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                if frame_skip > 1 and idx % frame_skip != 0:
                    continue
            else:
                ok = cap.grab()
                if not ok:
                    break
                idx += 1
                if frame_skip > 1 and idx % frame_skip != 0:
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    break

            frames.append(frame)
            numbers.append(idx)

            if batch_start is None:
                batch_start = time.monotonic()

            if len(frames) >= batch_size:
                prepare_ms = (time.monotonic() - batch_start) * 1000.0 if batch_start else 0.0
                put_start = time.monotonic()
                queue.put((frames.copy(), numbers.copy(), prepare_ms, (time.monotonic() - put_start) * 1000.0))
                frames.clear()
                numbers.clear()
                batch_start = None

        if frames:
            prepare_ms = (time.monotonic() - batch_start) * 1000.0 if batch_start else 0.0
            put_start = time.monotonic()
            queue.put((frames, numbers, prepare_ms, (time.monotonic() - put_start) * 1000.0))
    finally:
        cap.release()
        producer_done.set()
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
    prefetch_summary_collector: Optional[Callable[[Dict[str, float]], None]] = None,
    prefetch_log_stdout: bool = True,
    enable_profiler: bool = False,
    decode_all: bool = False,
) -> List[int]:
    """Decode frames on a background thread and run inference in batches."""
    queue: Queue = Queue(maxsize=prefetch_batches)
    stop = Event()
    producer_done = Event()
    thread = Thread(
        target=_decode_frames,
        daemon=True,
        args=(str(video_path), frame_skip or 1, batch_size, queue, stop, producer_done, decode_all),
    )
    thread.start()

    all_detections: List[int] = []

    # Fast path: no profiler
    if not enable_profiler:
        try:
            while True:
                try:
                    batch = queue.get(timeout=30)
                except Empty:
                    stop.set()
                    break

                if batch is None:
                    break

                frames, numbers = batch[0], batch[1]
                result = _run_inference(model, frames, numbers, conf, device, display, save_dir)
                all_detections.extend(result["detections"])

                if result["stop"]:
                    stop.set()
                    break
        finally:
            stop.set()
            thread.join(timeout=2)
        return all_detections

    # Profiler path
    video_name = Path(video_path).name
    profiler = PipelineProfiler(
        video_name, verbose=prefetch_log_stdout, warmup_batches=max(1, prefetch_batches)
    )

    # Get video metadata
    total_frames = 0
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    profiler.set_video_info(
        total_frames=total_frames,
        batch_size=batch_size,
        frame_skip=frame_skip or 1,
        prefetch_batches=prefetch_batches,
    )

    try:
        while True:
            with profiler.time_wait():
                try:
                    batch = queue.get(timeout=30)
                except Empty:
                    stop.set()
                    break

            if batch is None:
                break

            if len(batch) == 4:
                frames, numbers, prepare_ms, put_blocked_ms = batch
            else:
                frames, numbers = batch
                prepare_ms = 0.0
                put_blocked_ms = 0.0

            with profiler.time_infer():
                result = _run_inference(model, frames, numbers, conf, device, display, save_dir)

            profiler.record(
                prepare_ms=prepare_ms,
                put_blocked_ms=put_blocked_ms,
                queue_size=queue.qsize(),
                frame_count=len(frames),
                producer_done=producer_done.is_set(),
            )

            all_detections.extend(result["detections"])
            if result["stop"]:
                stop.set()
                break
    finally:
        stop.set()
        thread.join(timeout=2)

    summary = profiler.summary()
    if prefetch_summary_collector:
        prefetch_summary_collector(summary)

    return all_detections


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
    prefetch_batches: int,
    prefetch_summary_collector: Optional[Callable[[Dict[str, float]], None]] = None,
    prefetch_log_stdout: bool = True,
    enable_profiler: bool = False,
    decode_all: bool = False,
) -> List[int]:
    """Decode frames on the main thread and run inference in batches."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    if not enable_profiler:
        def _batch_iter() -> Iterable[Tuple[List, List[int]]]:
            frames: List = []
            numbers: List[int] = []
            idx = 0
            try:
                while True:
                    if decode_all:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        idx += 1
                        if (frame_skip or 1) > 1 and idx % frame_skip != 0:
                            continue
                    else:
                        ok = cap.grab()
                        if not ok:
                            break
                        idx += 1
                        if (frame_skip or 1) > 1 and idx % frame_skip != 0:
                            continue
                        ok, frame = cap.retrieve()
                        if not ok:
                            break

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

    # Profiler path
    video_name = Path(video_path).name
    profiler = SyncProfiler(
        video_name,
        verbose=prefetch_log_stdout,
        warmup_batches=max(1, prefetch_batches),
    )
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    profiler.set_video_info(
        total_frames=total_frames,
        batch_size=batch_size,
        frame_skip=frame_skip or 1,
    )

    all_detections: List[int] = []
    frames: List = []
    numbers: List[int] = []
    idx = 0
    batch_start = None
    stop_requested = False

    try:
        while True:
            if decode_all:
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                if (frame_skip or 1) > 1 and idx % frame_skip != 0:
                    continue
            else:
                ok = cap.grab()
                if not ok:
                    break
                idx += 1
                if (frame_skip or 1) > 1 and idx % frame_skip != 0:
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    break

            frames.append(frame)
            numbers.append(idx)

            if batch_start is None:
                batch_start = time.monotonic()

            if len(frames) >= batch_size:
                prepare_ms = (time.monotonic() - batch_start) * 1000.0 if batch_start else 0.0
                infer_start = time.monotonic()
                result = _run_inference(model, frames, numbers, conf, device, display, save_dir)
                infer_ms = (time.monotonic() - infer_start) * 1000.0

                profiler.record(prepare_ms=prepare_ms, infer_ms=infer_ms)
                all_detections.extend(result["detections"])

                if result["stop"]:
                    stop_requested = True
                    break

                frames, numbers = [], []
                batch_start = None

        if frames and not stop_requested:
            prepare_ms = (time.monotonic() - batch_start) * 1000.0 if batch_start else 0.0
            infer_start = time.monotonic()
            result = _run_inference(model, frames, numbers, conf, device, display, save_dir)
            infer_ms = (time.monotonic() - infer_start) * 1000.0

            profiler.record(prepare_ms=prepare_ms, infer_ms=infer_ms)
            all_detections.extend(result["detections"])
    finally:
        cap.release()

    summary = profiler.summary()
    if prefetch_summary_collector:
        prefetch_summary_collector(summary)

    return all_detections


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
    decode_all: bool = False,
    prefetch_summary_collector: Optional[Callable[[Dict[str, float]], None]] = None,
    prefetch_log_stdout: bool = True,
    enable_profiler: bool = False,
) -> List[int]:
    """
    Process video with optional async frame prefetching.

    Args:
        use_async: If True, decode frames in background thread (faster).
                   If False, use synchronous decoding (for comparison).
        enable_profiler: If True, enable pipeline profiler for timing analysis.
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
            prefetch_summary_collector=prefetch_summary_collector,
            prefetch_log_stdout=prefetch_log_stdout,
            enable_profiler=enable_profiler,
            decode_all=decode_all,
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
            prefetch_batches=prefetch_batches,
            prefetch_summary_collector=prefetch_summary_collector,
            prefetch_log_stdout=prefetch_log_stdout,
            enable_profiler=enable_profiler,
            decode_all=decode_all,
        )

    if display:
        cv2.destroyAllWindows()

    return detections


def process_video_adaptive(
    model,
    video_path: Union[str, Path],
    output_dir: str,
    *,
    conf: float,
    device: Optional[torch.device],
    display: bool,
    save_annotated_frames: bool,
    initial_skip: int = 2,
    consecutive_negative_threshold: int = 5,
    max_skip: int = 256,
) -> Tuple[List[Tuple[int, int]], int]:
    """
    Process video with adaptive frame skip strategy.

    Returns:
        Tuple of (detections, total_frames) where detections is a list of
        (frame_idx, detection_count) pairs sorted by frame index.
    """
    if save_annotated_frames:
        os.makedirs(output_dir, exist_ok=True)

    save_dir = output_dir if save_annotated_frames else None

    def inference_fn(frames: List) -> List[int]:
        """Run YOLO inference on a batch of frames."""
        if not frames:
            return []

        results = model.predict(
            frames, conf=conf, device=device, verbose=False,
            half=supports_half_precision(device)
        )

        detections = []
        for i, result in enumerate(results):
            det_count = len(result.boxes) if result.boxes else 0
            detections.append(det_count)

            if save_dir:
                # Use a placeholder frame number for adaptive mode
                cv2.imwrite(f"{save_dir}/annotated_frame_{i:05d}.jpg", result.plot())

            if display:
                cv2.imshow("Detections", result.plot())
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        return detections

    sampler = AdaptiveFrameSampler(
        initial_skip=initial_skip,
        consecutive_negative_threshold=consecutive_negative_threshold,
        max_skip=max_skip,
    )

    detections, total_frames = sampler.sample_video(str(video_path), inference_fn)

    if display:
        cv2.destroyAllWindows()

    return detections, total_frames
