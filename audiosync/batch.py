"""Run analysis across many pairs with bounded concurrency and real progress.

Concurrency is capped. The original code used a bare ``ThreadPoolExecutor()``,
which defaults to a worker per CPU and spawns one ffmpeg decode per worker; a
season of episodes decoding long segments simultaneously exhausts memory and
thrashes the disk to the point of being slower than running fewer at once.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence

from .analyze import PairResult, analyze_pair
from .matching import MatchPair
from .media import Cancelled, CancellationToken

# ffmpeg decoding is IO- and memory-heavy; more than a few at once is
# counter-productive on any consumer machine.
DEFAULT_MAX_WORKERS = 3


@dataclass
class BatchOptions:
    window_s: float = 45.0
    window_count: int = 6
    max_offset_ms: float = 60000.0
    max_workers: int = DEFAULT_MAX_WORKERS


class BatchEvents:
    """Callbacks fired as a batch progresses. All are optional."""

    def __init__(
        self,
        on_log: Optional[Callable[[str], None]] = None,
        on_pair_start: Optional[Callable[[str], None]] = None,
        on_pair_progress: Optional[Callable[[str, int], None]] = None,
        on_pair_done: Optional[Callable[[PairResult], None]] = None,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> None:
        self.on_log = on_log
        self.on_pair_start = on_pair_start
        self.on_pair_progress = on_pair_progress
        self.on_pair_done = on_pair_done
        self.on_progress = on_progress

    def log(self, message: str) -> None:
        if self.on_log:
            self.on_log(message)


def run_batch(
    pairs: Sequence[MatchPair],
    options: Optional[BatchOptions] = None,
    token: Optional[CancellationToken] = None,
    events: Optional[BatchEvents] = None,
) -> List[PairResult]:
    """Analyse every pair, returning results in the input order.

    Results are ordered deterministically regardless of completion order, so a
    re-run of the same batch produces an identical report.
    """
    options = options or BatchOptions()
    events = events or BatchEvents()
    token = token or CancellationToken()

    total = len(pairs)
    if total == 0:
        return []

    results: Dict[int, PairResult] = {}
    completed = 0

    workers = max(1, min(options.max_workers, total))
    events.log(f"Analysing {total} pair(s) with {workers} worker(s).")

    def work(index: int, pair: MatchPair) -> PairResult:
        name = pair.primary_name
        if events.on_pair_start:
            events.on_pair_start(name)

        started = time.monotonic()
        try:
            result = analyze_pair(
                pair.primary_path,
                pair.secondary_path,
                window_s=options.window_s,
                window_count=options.window_count,
                max_offset_ms=options.max_offset_ms,
                token=token,
                primary_track=pair.primary_track,
                secondary_track=pair.secondary_track,
                progress=(
                    (lambda percent: events.on_pair_progress(name, percent))
                    if events.on_pair_progress
                    else None
                ),
            )
        except Cancelled:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the batch
            result = PairResult(
                pair.primary_path,
                pair.secondary_path,
                error=f"{type(exc).__name__}: {exc}",
            )
        result.elapsed_ms = int((time.monotonic() - started) * 1000)
        return result

    executor = ThreadPoolExecutor(max_workers=workers)
    futures: Dict[Future, int] = {}
    try:
        for index, pair in enumerate(pairs):
            futures[executor.submit(work, index, pair)] = index

        for future in _as_completed(futures, token):
            index = futures[future]
            try:
                result = future.result()
            except Cancelled:
                break
            except Exception as exc:  # noqa: BLE001
                pair = pairs[index]
                result = PairResult(
                    pair.primary_path,
                    pair.secondary_path,
                    error=f"{type(exc).__name__}: {exc}",
                )

            results[index] = result
            completed += 1

            if events.on_pair_done:
                events.on_pair_done(result)
            if events.on_progress:
                events.on_progress(completed, total, result.primary_name)
    finally:
        if token.cancelled:
            for future in futures:
                future.cancel()
        executor.shutdown(wait=not token.cancelled)

    # Preserve input order; missing entries mean the run was cancelled.
    return [results[i] for i in sorted(results)]


def _as_completed(futures: Dict[Future, int], token: CancellationToken):
    """Yield futures as they finish, bailing out promptly on cancellation."""
    from concurrent.futures import as_completed as _base

    for future in _base(futures):
        if token.cancelled:
            return
        yield future


def summarize(results: Sequence[PairResult]) -> dict:
    """Aggregate counts for a finished batch."""
    matched = [r for r in results if r.error is None and r.delay_ms is not None]
    failed = [r for r in results if r.error is not None]
    drifting = [r for r in matched if r.has_significant_drift]
    cuts = [r for r in results if r.is_likely_cut]
    rate_mismatches = [r for r in matched if r.is_rate_mismatch]
    high = [r for r in matched if r.confidence >= 0.75]
    medium = [r for r in matched if 0.5 <= r.confidence < 0.75]
    low = [r for r in matched if r.confidence < 0.5]
    return {
        "total": len(results),
        "matched": len(matched),
        "failed": len(failed),
        "drifting": len(drifting),
        "cuts": len(cuts),
        "rateMismatches": len(rate_mismatches),
        "high": len(high),
        "medium": len(medium),
        "low": len(low),
    }
