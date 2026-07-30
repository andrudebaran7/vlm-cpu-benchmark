import matplotlib
matplotlib.use("Agg")

from vlmbench.report.figures import latex_results_table, save_tradeoff_plot
from vlmbench.report.records import CellResult, CellStatus


def _ok(model, ms, value):
    return CellResult(model=model, backend="fp32", task="docvqa",
                      status=CellStatus.OK, metric_name="anls",
                      metric_value=value, infer_ms_mean=ms, infer_ms_p95=ms + 10,
                      peak_rss_mb=800.0, energy_j=None, error=None)


def _failed(model):
    return CellResult(model=model, backend="onnx-int8", task="docvqa",
                      status=CellStatus.FAILED, metric_name=None,
                      metric_value=None, infer_ms_mean=None, infer_ms_p95=None,
                      peak_rss_mb=None, energy_j=None, error="x")


def test_latex_table_marks_failed_and_na():
    table = latex_results_table([_ok("a", 100.0, 0.5), _failed("b")])
    assert "tabular" in table
    assert "failed" in table
    assert "lllrrrr" in table       # combined table keeps the task column
    assert "energy\\_j" in table


def test_per_task_table_drops_task_column_and_filters():
    a = _ok("a", 100.0, 0.5)        # task docvqa (from _ok)
    b = _ok("b", 200.0, 0.6)
    b.task = "ocrbench"
    table = latex_results_table([a, b], task="docvqa")
    assert "llrrrr" in table        # one fewer column (no task)
    assert "docvqa" not in table    # constant task column omitted
    assert "\na & fp32 & 0.500" in table  # model a present, no task cell
    assert " b " not in table       # ocrbench cell filtered out


def test_tradeoff_plot_written(tmp_path):
    path = save_tradeoff_plot([_ok("a", 100.0, 0.5), _ok("b", 200.0, 0.6)],
                              tmp_path / "tradeoff.png")
    assert path.exists() and path.stat().st_size > 0
