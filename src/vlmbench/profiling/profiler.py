from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from .energy import measure_energy_j
from .latency import LatencyStats, profile_latency
from .memory import sample_peak_rss_mb


@dataclass
class StageCallables:
    pre: Callable[[], Any]
    infer: Callable[[Any], Any]
    post: Callable[[Any], Any]


@dataclass
class CellProfile:
    pre: LatencyStats
    infer: LatencyStats
    post: LatencyStats
    peak_rss_mb: float
    energy_j: float | None
    output: Any


def profile_cell(
    stages: StageCallables,
    *,
    warmup: int,
    repeats: int,
    interval_s: float = 0.005,
    clock: Callable[[], float] = time.perf_counter,
    sampler: Callable[[], float] | None = None,
    energy_reader: Callable[[], int] | None = None,
) -> CellProfile:
    pre_out_holder: dict[str, Any] = {}
    infer_out_holder: dict[str, Any] = {}

    pre_stats = profile_latency(
        lambda: pre_out_holder.__setitem__("v", stages.pre()),
        warmup=warmup, repeats=repeats, clock=clock,
    )
    pre_out = pre_out_holder["v"]
    infer_stats = profile_latency(
        lambda: infer_out_holder.__setitem__("v", stages.infer(pre_out)),
        warmup=warmup, repeats=repeats, clock=clock,
    )
    infer_out = infer_out_holder["v"]
    post_out_holder: dict[str, Any] = {}
    post_stats = profile_latency(
        lambda: post_out_holder.__setitem__("v", stages.post(infer_out)),
        warmup=warmup, repeats=repeats, clock=clock,
    )

    def full_pass() -> Any:
        return stages.post(stages.infer(stages.pre()))

    mem_kwargs: dict[str, Any] = {"interval_s": interval_s}
    if sampler is not None:
        mem_kwargs["sampler"] = sampler
    energy_kwargs: dict[str, Any] = {}
    if energy_reader is not None:
        energy_kwargs["reader"] = energy_reader

    def measured() -> Any:
        result, joules = measure_energy_j(full_pass, **energy_kwargs)
        measured.energy = joules  # type: ignore[attr-defined]
        return result

    output, peak = sample_peak_rss_mb(measured, **mem_kwargs)
    return CellProfile(
        pre=pre_stats, infer=infer_stats, post=post_stats,
        peak_rss_mb=peak, energy_j=measured.energy,  # type: ignore[attr-defined]
        output=post_out_holder["v"] if post_out_holder else output,
    )
