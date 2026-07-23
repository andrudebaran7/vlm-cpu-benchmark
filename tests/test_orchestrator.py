from vlmbench.config import BenchConfig
from vlmbench.models.base import ModelMeta
from vlmbench.models.fake import FakeVLModel
from vlmbench.orchestrator import run_matrix
from vlmbench.report.records import CellStatus, JsonlStore
from vlmbench.tasks.base import Example, TaskSpec
from vlmbench.tasks.metrics import exact_match


def _task_factory(name):
    examples = [Example(image=None, prompt="cat", answers=["cat"])]
    return examples, TaskSpec(name=name, metric=exact_match)


def _make_model(name, backends, responder=None):
    meta = ModelMeta(name=name, params_b=0.1, license="Apache-2.0",
                     source=name, supported_backends=tuple(backends))
    return FakeVLModel(meta=meta,
                       responder=responder or (lambda image, prompt: prompt))


def test_matrix_records_ok_unsupported_and_failed(tmp_path):
    def model_factory(name):
        if name == "good":
            return _make_model("good", ["fp32"])
        if name == "partial":
            return _make_model("partial", ["fp32"])  # onnx-int8 unsupported
        def boom(image, prompt):
            raise RuntimeError("kaboom")
        return _make_model("broken", ["fp32"], responder=boom)

    cfg = BenchConfig(models=["good", "partial", "broken"],
                      backends=["fp32", "onnx-int8"], tasks=["toy"],
                      warmup=0, repeats=1, subsample_n=1, seed=1)
    store = JsonlStore(tmp_path / "r.jsonl")
    results = run_matrix(cfg, model_factory, _task_factory, store)

    by_key = {(r.model, r.backend): r.status for r in results}
    assert by_key[("good", "fp32")] is CellStatus.OK
    assert by_key[("good", "onnx-int8")] is CellStatus.UNSUPPORTED
    assert by_key[("broken", "fp32")] is CellStatus.FAILED
    # Everything persisted, run did not abort on the failing cell.
    assert len(store.load()) == len(results) == 6


def test_matrix_isolates_task_load_failure(tmp_path):
    def bad_task_factory(name):
        raise RuntimeError("dataset download failed")
    cfg = BenchConfig(models=["good"], backends=["fp32", "onnx-int8"],
                      tasks=["boom"], warmup=0, repeats=1, subsample_n=1, seed=1)
    store = JsonlStore(tmp_path / "r.jsonl")
    results = run_matrix(cfg, lambda n: _make_model("good", ["fp32"]),
                         bad_task_factory, store)
    assert len(results) == 2  # one FAILED per backend, run did not abort
    assert all(r.status is CellStatus.FAILED for r in results)
    assert len(store.load()) == 2


def test_matrix_resumes_and_skips_completed(tmp_path):
    store = JsonlStore(tmp_path / "r.jsonl")
    cfg = BenchConfig(models=["good"], backends=["fp32"], tasks=["toy"],
                      warmup=0, repeats=1, subsample_n=1, seed=1)
    run_matrix(cfg, lambda n: _make_model("good", ["fp32"]), _task_factory, store)
    # Second run should add nothing (already completed).
    run_matrix(cfg, lambda n: _make_model("good", ["fp32"]), _task_factory, store)
    assert len(store.load()) == 1
