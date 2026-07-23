from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class LatencyStats:
    mean_ms: float
    median_ms: float
    p95_ms: float
    std_ms: float
    n: int


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = pct / 100.0 * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def profile_latency(
    fn: Callable[[], Any],
    *,
    warmup: int,
    repeats: int,
    clock: Callable[[], float] = time.perf_counter,
) -> LatencyStats:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    for _ in range(warmup):
        start = clock()
        fn()
        end = clock()
    samples_ms: list[float] = []
    for _ in range(repeats):
        start = clock()
        fn()
        end = clock()
        samples_ms.append((end - start) * 1000.0)
    ordered = sorted(samples_ms)
    return LatencyStats(
        mean_ms=statistics.fmean(samples_ms),
        median_ms=statistics.median(samples_ms),
        p95_ms=_percentile(ordered, 95.0),
        std_ms=statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0,
        n=len(samples_ms),
    )
