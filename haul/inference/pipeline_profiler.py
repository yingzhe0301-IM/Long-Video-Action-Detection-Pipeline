"""Pipeline profiler for analyzing prepare→infer timing distribution."""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional


@dataclass
class _BatchRecord:
    """Internal record for a single batch."""
    wait_s: float = 0.0
    infer_s: float = 0.0
    prepare_ms: float = 0.0
    put_blocked_ms: float = 0.0
    queue_size: int = 0
    frame_count: int = 0
    phase: str = "warmup"


class PipelineProfiler:
    """
    Analyze prepare→infer pipeline timing to detect CPU/GPU bottlenecks.

    Usage:
        profiler = PipelineProfiler("video.mp4", verbose=True)
        profiler.set_video_info(total_frames=1000, batch_size=64,
                                frame_skip=5, prefetch_batches=3)

        for batch in batches:
            with profiler.time_wait():
                batch_data = queue.get()

            with profiler.time_infer():
                result = model.predict(...)

            profiler.record(prepare_ms=..., queue_size=..., producer_done=...)

        summary = profiler.summary()
    """

    def __init__(
        self,
        video_name: str,
        *,
        verbose: bool = False,
        warmup_batches: int = 3,
    ) -> None:
        self._video_name = video_name
        self._verbose = verbose
        self._warmup_batches = warmup_batches

        # Video metadata
        self._total_frames = 0
        self._batch_size = 64
        self._frame_skip = 1
        self._prefetch_batches = 3
        self._total_batches_est = 0

        # Timing state
        self._batch_idx = 0
        self._current_wait_s = 0.0
        self._current_infer_s = 0.0
        self._records: List[_BatchRecord] = []

        # EWMA estimates
        self._ewma_alpha = 0.2
        self._tc_hat_ms = 0.0  # CPU (prepare) time estimate
        self._tg_hat_ms = 0.0  # GPU (infer) time estimate
        self._window_start = time.monotonic()
        self._window_wait_s = 0.0
        self._window_infer_s = 0.0

    def set_video_info(
        self,
        total_frames: int,
        batch_size: int,
        frame_skip: int,
        prefetch_batches: int,
    ) -> None:
        """Set video metadata for logging."""
        self._total_frames = total_frames
        self._batch_size = batch_size
        self._frame_skip = frame_skip
        self._prefetch_batches = prefetch_batches

        effective_skip = frame_skip or 1
        total_infer_frames = (
            (total_frames + effective_skip - 1) // effective_skip
            if total_frames else 0
        )
        self._total_batches_est = (
            (total_infer_frames + batch_size - 1) // batch_size
            if total_infer_frames else 0
        )

        if self._verbose:
            print(
                f"prefetch_video video={self._video_name} skip={frame_skip} "
                f"batch_size={batch_size} prefetch_batches={prefetch_batches} "
                f"total_frames={total_frames} total_batches_est={self._total_batches_est}"
            )

    @contextmanager
    def time_wait(self) -> Iterator[None]:
        """Context manager to time queue wait."""
        start = time.monotonic()
        try:
            yield
        finally:
            self._current_wait_s = time.monotonic() - start

    @contextmanager
    def time_infer(self) -> Iterator[None]:
        """Context manager to time inference."""
        start = time.monotonic()
        try:
            yield
        finally:
            self._current_infer_s = time.monotonic() - start

    def record(
        self,
        prepare_ms: float,
        put_blocked_ms: float,
        queue_size: int,
        frame_count: int,
        producer_done: bool,
    ) -> None:
        """Record timing for a completed batch."""
        self._batch_idx += 1
        wait_s = self._current_wait_s
        infer_s = self._current_infer_s
        wait_ms = wait_s * 1000.0
        infer_ms = infer_s * 1000.0

        # Update EWMA estimates
        if prepare_ms > 0:
            self._tc_hat_ms = (
                prepare_ms if self._tc_hat_ms <= 0
                else self._ewma_alpha * prepare_ms + (1.0 - self._ewma_alpha) * self._tc_hat_ms
            )
        if infer_ms > 0:
            self._tg_hat_ms = (
                infer_ms if self._tg_hat_ms <= 0
                else self._ewma_alpha * infer_ms + (1.0 - self._ewma_alpha) * self._tg_hat_ms
            )

        self._window_wait_s += wait_s
        self._window_infer_s += infer_s
        now = time.monotonic()
        if now - self._window_start >= 1.0:
            self._window_start = now
            self._window_wait_s = 0.0
            self._window_infer_s = 0.0

        wait_ratio_window = self._window_wait_s / (
            self._window_wait_s + self._window_infer_s + 1e-9
        )

        # Determine phase
        if producer_done:
            phase = "drain"
        elif self._batch_idx <= self._warmup_batches:
            phase = "warmup"
        else:
            phase = "steady"

        # Store record
        record = _BatchRecord(
            wait_s=wait_s,
            infer_s=infer_s,
            prepare_ms=prepare_ms,
            put_blocked_ms=put_blocked_ms,
            queue_size=queue_size,
            frame_count=frame_count,
            phase=phase,
        )
        self._records.append(record)

        # Log if verbose
        if self._verbose:
            backlog_ms = queue_size * self._tg_hat_ms
            print(
                f"prefetch_batch batch_idx={self._batch_idx} phase={phase} "
                f"qsize_batches={queue_size} "
                f"wait_ms={wait_ms:.2f} infer_ms={infer_ms:.2f} "
                f"prepare_ms={prepare_ms:.2f} put_blocked_ms={put_blocked_ms:.2f} "
                f"tc_hat_ms={self._tc_hat_ms:.2f} tg_hat_ms={self._tg_hat_ms:.2f} "
                f"wait_ratio_window={wait_ratio_window:.2f} backlog_ms={backlog_ms:.2f}"
            )

    @staticmethod
    def _summarize(records: List[_BatchRecord]) -> Dict[str, float]:
        batch_count = len(records)
        total_wait_s = sum(r.wait_s for r in records)
        total_infer_s = sum(r.infer_s for r in records)
        total_prepare_ms = sum(r.prepare_ms for r in records)

        if batch_count:
            avg_wait_ms = (total_wait_s * 1000.0) / batch_count
            avg_infer_ms = (total_infer_s * 1000.0) / batch_count
            avg_prepare_ms = total_prepare_ms / batch_count
            wait_ratio = total_wait_s / (total_wait_s + total_infer_s + 1e-9)
        else:
            avg_wait_ms = 0.0
            avg_infer_ms = 0.0
            avg_prepare_ms = 0.0
            wait_ratio = 0.0

        return {
            "batches": float(batch_count),
            "avg_wait_ms": avg_wait_ms,
            "avg_infer_ms": avg_infer_ms,
            "avg_prepare_ms": avg_prepare_ms,
            "wait_ratio": wait_ratio,
        }

    def summary(self) -> Dict[str, float]:
        """Compute and return summary statistics."""
        total_stats = self._summarize(self._records)
        steady_stats = self._summarize([r for r in self._records if r.phase == "steady"])

        result = {
            "video": self._video_name,
            "batches": total_stats["batches"],
            "total_batches_est": float(self._total_batches_est),
            "avg_wait_ms": total_stats["avg_wait_ms"],
            "avg_infer_ms": total_stats["avg_infer_ms"],
            "avg_prepare_ms": total_stats["avg_prepare_ms"],
            "wait_ratio": total_stats["wait_ratio"],
            "steady_batches": steady_stats["batches"],
            "steady_avg_wait_ms": steady_stats["avg_wait_ms"],
            "steady_avg_infer_ms": steady_stats["avg_infer_ms"],
            "steady_avg_prepare_ms": steady_stats["avg_prepare_ms"],
            "steady_wait_ratio": steady_stats["wait_ratio"],
        }

        if self._verbose:
            print(
                f"prefetch_summary video={self._video_name} batches={int(total_stats['batches'])} "
                f"total_batches_est={self._total_batches_est} "
                f"avg_wait_ms={total_stats['avg_wait_ms']:.2f} "
                f"avg_infer_ms={total_stats['avg_infer_ms']:.2f} "
                f"avg_prepare_ms={total_stats['avg_prepare_ms']:.2f} "
                f"wait_ratio={total_stats['wait_ratio']:.4f} "
                f"steady_batches={int(steady_stats['batches'])} "
                f"steady_avg_wait_ms={steady_stats['avg_wait_ms']:.2f} "
                f"steady_avg_infer_ms={steady_stats['avg_infer_ms']:.2f} "
                f"steady_avg_prepare_ms={steady_stats['avg_prepare_ms']:.2f} "
                f"steady_wait_ratio={steady_stats['wait_ratio']:.4f}"
            )

        return result

    @staticmethod
    def aggregate(summaries: List[Dict[str, float]]) -> Dict[str, Optional[float]]:
        """
        Aggregate summary statistics from multiple videos.

        Returns a dict with prefetch_* keys for compatibility with existing CSV output.
        """
        if not summaries:
            return {
                "prefetch_videos": None,
                "prefetch_total_batches": None,
                "prefetch_avg_wait_ms": None,
                "prefetch_avg_infer_ms": None,
                "prefetch_avg_prepare_ms": None,
                "prefetch_wait_ratio": None,
                "prefetch_steady_batches": None,
                "prefetch_steady_avg_wait_ms": None,
                "prefetch_steady_avg_infer_ms": None,
                "prefetch_steady_avg_prepare_ms": None,
                "prefetch_steady_wait_ratio": None,
            }

        def _weighted_sum(value_key: str, weight_key: str) -> float:
            return sum(s.get(value_key, 0.0) * s.get(weight_key, 0.0) for s in summaries)

        def _safe_avg(total: float, weight: float) -> float:
            return total / weight if weight else 0.0

        total_batches = sum(s.get("batches", 0.0) for s in summaries)
        steady_batches = sum(s.get("steady_batches", 0.0) for s in summaries)

        total_wait_ms = _weighted_sum("avg_wait_ms", "batches")
        total_infer_ms = _weighted_sum("avg_infer_ms", "batches")
        total_prepare_ms = _weighted_sum("avg_prepare_ms", "batches")
        avg_wait_ms = _safe_avg(total_wait_ms, total_batches)
        avg_infer_ms = _safe_avg(total_infer_ms, total_batches)
        avg_prepare_ms = _safe_avg(total_prepare_ms, total_batches)
        wait_ratio = total_wait_ms / (total_wait_ms + total_infer_ms + 1e-9)

        steady_wait_ms = _weighted_sum("steady_avg_wait_ms", "steady_batches")
        steady_infer_ms = _weighted_sum("steady_avg_infer_ms", "steady_batches")
        steady_prepare_ms = _weighted_sum("steady_avg_prepare_ms", "steady_batches")
        steady_avg_wait_ms = _safe_avg(steady_wait_ms, steady_batches)
        steady_avg_infer_ms = _safe_avg(steady_infer_ms, steady_batches)
        steady_avg_prepare_ms = _safe_avg(steady_prepare_ms, steady_batches)
        steady_wait_ratio = steady_wait_ms / (steady_wait_ms + steady_infer_ms + 1e-9)

        return {
            "prefetch_videos": len(summaries),
            "prefetch_total_batches": total_batches,
            "prefetch_avg_wait_ms": avg_wait_ms,
            "prefetch_avg_infer_ms": avg_infer_ms,
            "prefetch_avg_prepare_ms": avg_prepare_ms,
            "prefetch_wait_ratio": wait_ratio,
            "prefetch_steady_batches": steady_batches,
            "prefetch_steady_avg_wait_ms": steady_avg_wait_ms,
            "prefetch_steady_avg_infer_ms": steady_avg_infer_ms,
            "prefetch_steady_avg_prepare_ms": steady_avg_prepare_ms,
            "prefetch_steady_wait_ratio": steady_wait_ratio,
        }

    @staticmethod
    def print_aggregate(
        summaries: List[Dict[str, float]],
        skip: int,
    ) -> None:
        """Print aggregated summary for a scan iteration."""
        stats = PipelineProfiler.aggregate(summaries)
        if stats["prefetch_videos"] is None:
            return

        print(
            f"prefetch_scan_summary skip={skip} "
            f"videos={stats['prefetch_videos']} "
            f"batches={int(stats['prefetch_total_batches'] or 0)} "
            f"avg_wait_ms={stats['prefetch_avg_wait_ms']:.2f} "
            f"avg_infer_ms={stats['prefetch_avg_infer_ms']:.2f} "
            f"avg_prepare_ms={stats['prefetch_avg_prepare_ms']:.2f} "
            f"wait_ratio={stats['prefetch_wait_ratio']:.4f} "
            f"steady_batches={int(stats['prefetch_steady_batches'] or 0)} "
            f"steady_avg_wait_ms={stats['prefetch_steady_avg_wait_ms']:.2f} "
            f"steady_avg_infer_ms={stats['prefetch_steady_avg_infer_ms']:.2f} "
            f"steady_avg_prepare_ms={stats['prefetch_steady_avg_prepare_ms']:.2f} "
            f"steady_wait_ratio={stats['prefetch_steady_wait_ratio']:.4f}"
        )


@dataclass
class _SyncBatchRecord:
    """Internal record for a single sync batch."""
    prepare_ms: float = 0.0
    infer_ms: float = 0.0
    phase: str = "warmup"


class SyncProfiler:
    """Profile synchronous decode+infer timing (no prefetch queue)."""

    def __init__(
        self,
        video_name: str,
        *,
        verbose: bool = False,
        warmup_batches: int = 3,
    ) -> None:
        self._video_name = video_name
        self._verbose = verbose
        self._warmup_batches = warmup_batches

        # Batch tracking
        self._batch_idx = 0
        self._records: List[_SyncBatchRecord] = []

        # Video metadata
        self._total_frames = 0
        self._batch_size = 64
        self._frame_skip = 1
        self._total_batches_est = 0

    def set_video_info(
        self,
        total_frames: int,
        batch_size: int,
        frame_skip: int,
    ) -> None:
        """Set video metadata for logging."""
        self._total_frames = total_frames
        self._batch_size = batch_size
        self._frame_skip = frame_skip

        effective_skip = frame_skip or 1
        total_infer_frames = (
            (total_frames + effective_skip - 1) // effective_skip
            if total_frames else 0
        )
        self._total_batches_est = (
            (total_infer_frames + batch_size - 1) // batch_size
            if total_infer_frames else 0
        )

        if self._verbose:
            print(
                f"sync_video video={self._video_name} skip={frame_skip} "
                f"batch_size={batch_size} total_frames={total_frames} "
                f"total_batches_est={self._total_batches_est}"
            )

    def record(
        self,
        *,
        prepare_ms: float,
        infer_ms: float,
    ) -> None:
        """Record timing for a completed batch."""
        self._batch_idx += 1
        phase = "warmup" if self._batch_idx <= self._warmup_batches else "steady"
        self._records.append(
            _SyncBatchRecord(prepare_ms=prepare_ms, infer_ms=infer_ms, phase=phase)
        )

        if self._verbose:
            total_ms = prepare_ms + infer_ms
            wait_ratio = prepare_ms / (total_ms + 1e-9)
            print(
                f"sync_batch batch_idx={self._batch_idx} phase={phase} "
                f"prepare_ms={prepare_ms:.2f} infer_ms={infer_ms:.2f} "
                f"wait_ratio={wait_ratio:.4f}"
            )

    @staticmethod
    def _summarize(records: List[_SyncBatchRecord]) -> Dict[str, float]:
        batch_count = len(records)
        total_prepare_ms = sum(r.prepare_ms for r in records)
        total_infer_ms = sum(r.infer_ms for r in records)

        if batch_count:
            avg_prepare_ms = total_prepare_ms / batch_count
            avg_infer_ms = total_infer_ms / batch_count
            wait_ratio = total_prepare_ms / (total_prepare_ms + total_infer_ms + 1e-9)
        else:
            avg_prepare_ms = 0.0
            avg_infer_ms = 0.0
            wait_ratio = 0.0

        return {
            "batches": float(batch_count),
            "avg_prepare_ms": avg_prepare_ms,
            "avg_infer_ms": avg_infer_ms,
            "wait_ratio": wait_ratio,
        }

    def summary(self) -> Dict[str, float]:
        """Compute and return summary statistics."""
        total_stats = self._summarize(self._records)
        steady_stats = self._summarize([r for r in self._records if r.phase == "steady"])

        serial_ms = sum(r.prepare_ms + r.infer_ms for r in self._records)
        prep_end_ms = 0.0
        infer_end_ms = 0.0
        for record in self._records:
            prep_end_ms += record.prepare_ms
            infer_start_ms = max(prep_end_ms, infer_end_ms)
            infer_end_ms = infer_start_ms + record.infer_ms
        ideal_overlap_ms = infer_end_ms
        ideal_saved_ms = serial_ms - ideal_overlap_ms

        result = {
            "video": self._video_name,
            "batches": total_stats["batches"],
            "total_batches_est": float(self._total_batches_est),
            "avg_prepare_ms": total_stats["avg_prepare_ms"],
            "avg_infer_ms": total_stats["avg_infer_ms"],
            "wait_ratio": total_stats["wait_ratio"],
            "serial_ms": serial_ms,
            "ideal_overlap_ms": ideal_overlap_ms,
            "ideal_saved_ms": ideal_saved_ms,
            "steady_batches": steady_stats["batches"],
            "steady_avg_prepare_ms": steady_stats["avg_prepare_ms"],
            "steady_avg_infer_ms": steady_stats["avg_infer_ms"],
            "steady_wait_ratio": steady_stats["wait_ratio"],
        }

        if self._verbose:
            print(
                f"sync_summary video={self._video_name} batches={int(total_stats['batches'])} "
                f"total_batches_est={self._total_batches_est} "
                f"avg_prepare_ms={total_stats['avg_prepare_ms']:.2f} "
                f"avg_infer_ms={total_stats['avg_infer_ms']:.2f} "
                f"wait_ratio={total_stats['wait_ratio']:.4f} "
                f"steady_batches={int(steady_stats['batches'])} "
                f"steady_avg_prepare_ms={steady_stats['avg_prepare_ms']:.2f} "
                f"steady_avg_infer_ms={steady_stats['avg_infer_ms']:.2f} "
                f"steady_wait_ratio={steady_stats['wait_ratio']:.4f}"
            )

        return result

    @staticmethod
    def aggregate(summaries: List[Dict[str, float]]) -> Dict[str, Optional[float]]:
        """Aggregate summary statistics from multiple videos."""
        if not summaries:
            return {
                "sync_videos": None,
                "sync_total_batches": None,
                "sync_total_prepare_ms": None,
                "sync_total_infer_ms": None,
                "sync_avg_prepare_ms": None,
                "sync_avg_infer_ms": None,
                "sync_wait_ratio": None,
                "sync_ideal_serial_ms": None,
                "sync_ideal_overlap_ms": None,
                "sync_ideal_saved_ms": None,
                "sync_steady_batches": None,
                "sync_steady_total_prepare_ms": None,
                "sync_steady_total_infer_ms": None,
                "sync_steady_avg_prepare_ms": None,
                "sync_steady_avg_infer_ms": None,
                "sync_steady_wait_ratio": None,
            }

        def _weighted_sum(value_key: str, weight_key: str) -> float:
            return sum(s.get(value_key, 0.0) * s.get(weight_key, 0.0) for s in summaries)

        def _safe_avg(total: float, weight: float) -> float:
            return total / weight if weight else 0.0

        total_batches = sum(s.get("batches", 0.0) for s in summaries)
        steady_batches = sum(s.get("steady_batches", 0.0) for s in summaries)
        total_serial_ms = sum(s.get("serial_ms", 0.0) for s in summaries)
        total_overlap_ms = sum(s.get("ideal_overlap_ms", 0.0) for s in summaries)
        total_saved_ms = sum(s.get("ideal_saved_ms", 0.0) for s in summaries)

        total_prepare_ms = _weighted_sum("avg_prepare_ms", "batches")
        total_infer_ms = _weighted_sum("avg_infer_ms", "batches")
        avg_prepare_ms = _safe_avg(total_prepare_ms, total_batches)
        avg_infer_ms = _safe_avg(total_infer_ms, total_batches)
        wait_ratio = total_prepare_ms / (total_prepare_ms + total_infer_ms + 1e-9)

        steady_prepare_ms = _weighted_sum("steady_avg_prepare_ms", "steady_batches")
        steady_infer_ms = _weighted_sum("steady_avg_infer_ms", "steady_batches")
        steady_avg_prepare_ms = _safe_avg(steady_prepare_ms, steady_batches)
        steady_avg_infer_ms = _safe_avg(steady_infer_ms, steady_batches)
        steady_wait_ratio = steady_prepare_ms / (steady_prepare_ms + steady_infer_ms + 1e-9)

        return {
            "sync_videos": len(summaries),
            "sync_total_batches": total_batches,
            "sync_total_prepare_ms": total_prepare_ms,
            "sync_total_infer_ms": total_infer_ms,
            "sync_avg_prepare_ms": avg_prepare_ms,
            "sync_avg_infer_ms": avg_infer_ms,
            "sync_wait_ratio": wait_ratio,
            "sync_ideal_serial_ms": total_serial_ms,
            "sync_ideal_overlap_ms": total_overlap_ms,
            "sync_ideal_saved_ms": total_saved_ms,
            "sync_steady_batches": steady_batches,
            "sync_steady_total_prepare_ms": steady_prepare_ms,
            "sync_steady_total_infer_ms": steady_infer_ms,
            "sync_steady_avg_prepare_ms": steady_avg_prepare_ms,
            "sync_steady_avg_infer_ms": steady_avg_infer_ms,
            "sync_steady_wait_ratio": steady_wait_ratio,
        }

    @staticmethod
    def print_aggregate(
        summaries: List[Dict[str, float]],
        skip: int,
    ) -> None:
        """Print aggregated summary for a scan iteration."""
        stats = SyncProfiler.aggregate(summaries)
        if stats["sync_videos"] is None:
            return

        print(
            f"sync_scan_summary skip={skip} "
            f"videos={stats['sync_videos']} "
            f"batches={int(stats['sync_total_batches'] or 0)} "
            f"avg_prepare_ms={stats['sync_avg_prepare_ms']:.2f} "
            f"avg_infer_ms={stats['sync_avg_infer_ms']:.2f} "
            f"wait_ratio={stats['sync_wait_ratio']:.4f} "
            f"steady_batches={int(stats['sync_steady_batches'] or 0)} "
            f"steady_avg_prepare_ms={stats['sync_steady_avg_prepare_ms']:.2f} "
            f"steady_avg_infer_ms={stats['sync_steady_avg_infer_ms']:.2f} "
            f"steady_wait_ratio={stats['sync_steady_wait_ratio']:.4f}"
        )
