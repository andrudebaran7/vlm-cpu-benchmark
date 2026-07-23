from vlmbench.report.records import CellResult, CellStatus, JsonlStore


def make_result(model="smolvlm-256m", backend="fp32", task="docvqa",
                status=CellStatus.OK):
    return CellResult(
        model=model, backend=backend, task=task, status=status,
        metric_name="anls", metric_value=0.5, infer_ms_mean=120.0,
        infer_ms_p95=140.0, peak_rss_mb=900.0, energy_j=None, error=None,
    )


def test_roundtrip_and_completed_keys(tmp_path):
    store = JsonlStore(tmp_path / "results.jsonl")
    store.append(make_result())
    store.append(make_result(backend="onnx-int8"))
    loaded = store.load()
    assert len(loaded) == 2
    assert loaded[0].status is CellStatus.OK
    assert store.completed_keys() == {
        ("smolvlm-256m", "fp32", "docvqa"),
        ("smolvlm-256m", "onnx-int8", "docvqa"),
    }


def test_failed_record_preserves_error(tmp_path):
    store = JsonlStore(tmp_path / "r.jsonl")
    store.append(CellResult(
        model="m", backend="gguf-q4", task="ocrbench",
        status=CellStatus.FAILED, metric_name=None, metric_value=None,
        infer_ms_mean=None, infer_ms_p95=None, peak_rss_mb=None,
        energy_j=None, error="boom",
    ))
    assert store.load()[0].error == "boom"
