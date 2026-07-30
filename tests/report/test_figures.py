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
    assert "lllrlrrr" in table
    assert "energy\\_j" in table
    assert "metric 95\\% CI" in table


def test_ci_column_present_and_dashed_without_scores():
    # A cell with no per-example scores shows an em-dash in the CI column.
    table = latex_results_table([_ok("a", 100.0, 0.5)])
    assert "---" in table


def test_ci_column_shows_interval_with_scores():
    cell = _ok("a", 100.0, 0.5)
    cell.per_example_scores = [0.0, 1.0] * 40  # mean 0.5
    table = latex_results_table([cell])
    assert "[0." in table  # a bracketed interval rendered


def test_tradeoff_plot_written(tmp_path):
    path = save_tradeoff_plot([_ok("a", 100.0, 0.5), _ok("b", 200.0, 0.6)],
                              tmp_path / "tradeoff.png")
    assert path.exists() and path.stat().st_size > 0
