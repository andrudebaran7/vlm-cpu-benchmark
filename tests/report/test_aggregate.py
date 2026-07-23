import pandas as pd
from vlmbench.report.aggregate import to_dataframe, write_csv
from vlmbench.report.records import CellResult, CellStatus


def _ok(model, backend, value):
    return CellResult(model=model, backend=backend, task="docvqa",
                      status=CellStatus.OK, metric_name="anls",
                      metric_value=value, infer_ms_mean=100.0, infer_ms_p95=120.0,
                      peak_rss_mb=800.0, energy_j=None, error=None)


def _failed(model, backend):
    return CellResult(model=model, backend=backend, task="docvqa",
                      status=CellStatus.FAILED, metric_name=None,
                      metric_value=None, infer_ms_mean=None, infer_ms_p95=None,
                      peak_rss_mb=None, energy_j=None, error="oom-ish")


def test_dataframe_has_row_per_result_and_status_strings():
    df = to_dataframe([_ok("a", "fp32", 0.5), _failed("a", "onnx-int8")])
    assert len(df) == 2
    assert set(df["status"]) == {"ok", "failed"}
    assert df.loc[df["backend"] == "fp32", "metric_value"].iloc[0] == 0.5
    assert pd.isna(df.loc[df["backend"] == "onnx-int8", "metric_value"].iloc[0])


def test_write_csv_roundtrips(tmp_path):
    path = write_csv([_ok("a", "fp32", 0.5)], tmp_path / "out.csv")
    reread = pd.read_csv(path)
    assert reread["model"].iloc[0] == "a"
    assert reread["status"].iloc[0] == "ok"
