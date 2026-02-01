"""Adaptive frame sampling with dynamic skip rate adjustment."""

from typing import Callable, Dict, List, Tuple

import cv2


class AdaptiveFrameSampler:
    """
    Adaptive frame sampler that adjusts skip rate based on detection results.

    Algorithm:
    1. Forward pass with adaptive skip (double after threshold consecutive negatives)
    2. On positive detection, backtrack with initial_skip to sample skipped region
    3. Reset to initial_skip and continue
    """

    def __init__(
        self,
        initial_skip: int = 2,
        consecutive_negative_threshold: int = 5,
        max_skip: int = 256,
    ):
        self.initial_skip = initial_skip
        self.consecutive_negative_threshold = consecutive_negative_threshold
        self.max_skip = max_skip

    def sample_video(
        self,
        video_path: str,
        inference_fn: Callable[[List], List[int]],
    ) -> Tuple[List[Tuple[int, int]], int]:
        """
        Sample video frames adaptively and run inference.

        Args:
            video_path: Path to the video file.
            inference_fn: Function that takes a list of frames and returns
                         a list of detection counts (one per frame).

        Returns:
            Tuple of (detections, total_frames) where detections is a list of
            (frame_idx, detection_count) pairs sorted by frame index.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        detections: Dict[int, int] = {}

        current_skip = self.initial_skip
        consecutive_negatives = 0
        frame_idx = 0
        prev_frame_idx = -1
        last_positive_idx = -1

        try:
            while frame_idx < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    break

                # Run inference on single frame
                det_counts = inference_fn([frame])
                det_count = det_counts[0] if det_counts else 0
                detections[frame_idx] = det_count

                if det_count > 0:
                    # Positive detection - densely sample the skipped region
                    if prev_frame_idx >= 0 and frame_idx - prev_frame_idx > 1:
                        backtrack_detections = self._dense_backtrack(
                            cap, inference_fn, prev_frame_idx, frame_idx
                        )
                        detections.update(backtrack_detections)

                    last_positive_idx = frame_idx
                    consecutive_negatives = 0
                    current_skip = self.initial_skip
                else:
                    # Negative detection
                    consecutive_negatives += 1
                    if consecutive_negatives >= self.consecutive_negative_threshold:
                        current_skip = min(current_skip * 2, self.max_skip)
                        consecutive_negatives = 0

                prev_frame_idx = frame_idx
                frame_idx += current_skip

        finally:
            cap.release()

        # Sort detections by frame index
        sorted_detections = sorted(detections.items(), key=lambda x: x[0])
        return sorted_detections, total_frames

    def _dense_backtrack(
        self,
        cap: cv2.VideoCapture,
        inference_fn: Callable[[List], List[int]],
        start: int,
        end: int,
    ) -> Dict[int, int]:
        """
        Sample the skipped region with initial_skip when a positive is detected.

        When we jump from 'start' to 'end' and find a positive at 'end',
        we go back and sample the region with initial_skip interval.

        Args:
            cap: OpenCV VideoCapture object.
            inference_fn: Function that takes a list of frames and returns detection counts.
            start: Start frame index (last sampled frame before the jump).
            end: End frame index (current positive detection).

        Returns:
            Dictionary mapping frame indices to detection counts.
        """
        detections: Dict[int, int] = {}

        frame_idx = start + self.initial_skip
        while frame_idx < end:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break

            det_counts = inference_fn([frame])
            det_count = det_counts[0] if det_counts else 0
            detections[frame_idx] = det_count

            frame_idx += self.initial_skip

        return detections


__all__ = ["AdaptiveFrameSampler"]
