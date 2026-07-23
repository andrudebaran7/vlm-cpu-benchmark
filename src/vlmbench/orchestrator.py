from __future__ import annotations

import traceback
from typing import Callable

from .backends.registry import backend_supports
from .config import BenchConfig
from .models.base import VLModel
from .profiling.profiler import CellProfile, StageCallables, profile_cell
from .report.records import CellResult, CellStatus, JsonlStore
from .tasks.base import Example, TaskSpec, run_task, subsample

ModelFactory = Callable[[str], VLModel]
TaskFactory = Callable[[str], "tuple[list[Example], TaskSpec]"]


def _profile_first_example(model: VLModel, example: Example,
                           warmup: int, repeats: int) -> CellProfile:
    """Profile one example (pre/infer/post latency + peak RAM + energy)."""
    stages = StageCallables(
        pre=lambda: (example.image, example.prompt),
        infer=lambda inp: model.infer(inp[0], inp[1]),
        post=lambda out: out,
    )
    return profile_cell(stages, warmup=warmup, repeats=repeats)


def run_matrix(config: BenchConfig, model_factory: ModelFactory,
               task_factory: TaskFactory, store: JsonlStore,
               *, mem_limit_mb: float | None = None) -> list[CellResult]:
    done = store.completed_keys()
    results: list[CellResult] = []
    for model_name in config.models:
        for task_name in config.tasks:
            try:
                examples_all, spec = task_factory(task_name)
                examples = subsample(examples_all, config.subsample_n, config.seed)
            except Exception as exc:
                for backend in config.backends:
                    key = (model_name, backend, task_name)
                    if key in done:
                        continue
                    result = CellResult(
                        model=model_name, backend=backend, task=task_name,
                        status=CellStatus.FAILED, metric_name=None, metric_value=None,
                        infer_ms_mean=None, infer_ms_p95=None, peak_rss_mb=None,
                        energy_j=None,
                        error=f"task load failed: {type(exc).__name__}: {exc}",
                    )
                    store.append(result)
                    results.append(result)
                    done.add(key)
                continue
            for backend in config.backends:
                key = (model_name, backend, task_name)
                if key in done:
                    continue
                result = _run_cell(model_name, backend, task_name, examples,
                                   spec, model_factory, config, mem_limit_mb)
                store.append(result)
                results.append(result)
                done.add(key)
    return results


def _run_cell(model_name, backend, task_name, examples, spec,
              model_factory, config, mem_limit_mb) -> CellResult:
    def base(status, **kw):
        fields = dict(model=model_name, backend=backend, task=task_name,
                      status=status, metric_name=None, metric_value=None,
                      infer_ms_mean=None, infer_ms_p95=None, peak_rss_mb=None,
                      energy_j=None, error=None)
        fields.update(kw)
        return CellResult(**fields)

    try:
        model = model_factory(model_name)
        if not backend_supports(model.meta, backend):
            return base(CellStatus.UNSUPPORTED)
        model.load(backend=backend, dtype="float32")
        score, _preds = run_task(model, examples, spec)
        profile = _profile_first_example(model, examples[0],
                                         config.warmup, config.repeats)
        if mem_limit_mb is not None and profile.peak_rss_mb > mem_limit_mb:
            return base(CellStatus.OOM, peak_rss_mb=profile.peak_rss_mb)
        return base(CellStatus.OK, metric_name=spec.name, metric_value=score,
                    infer_ms_mean=profile.infer.mean_ms,
                    infer_ms_p95=profile.infer.p95_ms,
                    peak_rss_mb=profile.peak_rss_mb, energy_j=profile.energy_j)
    except MemoryError:
        return base(CellStatus.OOM, error="MemoryError")
    except Exception as exc:  # isolation: never abort the run
        return base(CellStatus.FAILED,
                    error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
