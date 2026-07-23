from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .records import CellResult, CellStatus


def _fmt(value) -> str:
    return "n/a" if value is None else (f"{value:.3f}" if isinstance(value, float) else str(value))


def latex_results_table(results: list[CellResult]) -> str:
    header = "\\begin{tabular}{lllrrr}\n\\toprule\n"
    header += "model & backend & task & metric & infer\\_ms & peak\\_mb \\\\\n\\midrule\n"
    lines = []
    for r in results:
        if r.status is CellStatus.OK:
            metric = _fmt(r.metric_value)
            infer = _fmt(r.infer_ms_mean)
            peak = _fmt(r.peak_rss_mb)
        else:
            metric = infer = peak = f"\\text{{{r.status.value}}}"
        lines.append(f"{r.model} & {r.backend} & {r.task} & {metric} & {infer} & {peak} \\\\")
    body = "\n".join(lines)
    footer = "\n\\bottomrule\n\\end{tabular}"
    return header + body + footer


def save_tradeoff_plot(results: list[CellResult], path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = [r for r in results if r.status is CellStatus.OK
          and r.infer_ms_mean is not None and r.metric_value is not None]
    fig, ax = plt.subplots(figsize=(5, 4))
    for r in ok:
        ax.scatter(r.infer_ms_mean, r.metric_value)
        ax.annotate(r.model, (r.infer_ms_mean, r.metric_value))
    ax.set_xlabel("inference latency (ms)")
    ax.set_ylabel("accuracy metric")
    ax.set_title("Accuracy vs latency (CPU)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
