from __future__ import annotations

import threading
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def _default_sampler() -> float:
    import psutil

    return psutil.Process().memory_info().rss / (1024 * 1024)


def sample_peak_rss_mb(
    fn: Callable[[], T],
    *,
    interval_s: float = 0.01,
    sampler: Callable[[], float] = _default_sampler,
) -> tuple[T, float]:
    peak = sampler()  # guarantee at least one reading
    stop = threading.Event()
    lock = threading.Lock()

    def poll() -> None:
        nonlocal peak
        while not stop.is_set():
            value = sampler()
            with lock:
                if value > peak:
                    peak = value
            if interval_s:
                time.sleep(interval_s)

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    try:
        result = fn()
    finally:
        stop.set()
        thread.join(timeout=1.0)
    with lock:
        final_peak = peak
    return result, final_peak
