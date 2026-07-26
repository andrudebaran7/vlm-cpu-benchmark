from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .records import CellResult, CellStatus


def _fmt(value) -> str:
    return "n/a" if value is None else (f"{value:.3f}" if isinstance(value, float) else str(value))


def _tex(s: str) -> str:
    """Escape LaTeX-special characters in a text cell (e.g. underscores in
    model ids like ``internvl2_5-2b``)."""
    return str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&")


def latex_results_table(results: list[CellResult]) -> str:
    header = "\\begin{tabular}{lllrrrr}\n\\toprule\n"
    header += ("model & backend & task & metric & infer\\_ms & peak\\_mb "
               "& energy\\_j \\\\\n\\midrule\n")
    lines = []
    for r in results:
        if r.status is CellStatus.OK:
            metric = _fmt(r.metric_value)
            infer = _fmt(r.infer_ms_mean)
            peak = _fmt(r.peak_rss_mb)
            energy = _fmt(r.energy_j)
        else:
            metric = infer = peak = energy = f"\\text{{{r.status.value}}}"
        lines.append(f"{_tex(r.model)} & {_tex(r.backend)} & {_tex(r.task)} "
                     f"& {metric} & {infer} & {peak} & {energy} \\\\")
    body = "\n".join(lines)
    footer = "\n\\bottomrule\n\\end{tabular}"
    return header + body + footer


# Okabe-Ito colourblind-safe qualitative palette (validated: adjacent CVD
# deltaE >= 11). Identity is carried by colour (per model) AND by the direct
# label, so no point relies on colour alone.
_MODEL_COLORS = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9")
_BACKEND_MARKERS = {"fp32": "o", "onnx-int8": "^"}


def _plot_tradeoff_ax(ax, rows, color_of) -> None:
    """Draw one accuracy-vs-latency panel (log-x, colour=model, marker=backend,
    direct labels that diverge same-model pairs to avoid collisions)."""
    ax.set_xscale("log")  # latency spans ~2s..215s; log separates the cluster
    ax.grid(True, which="both", linewidth=0.4, color="0.85", zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    x_max = max((r.infer_ms_mean for r in rows), default=1.0)
    for r in rows:
        marker = _BACKEND_MARKERS.get(r.backend, "s")
        ax.scatter(r.infer_ms_mean, r.metric_value, s=70, marker=marker,
                   color=color_of[r.model], edgecolors="white", linewidths=0.8,
                   zorder=3)
        if r.backend == "fp32":
            label = r.model
            right_edge = r.infer_ms_mean > x_max / 3
            dx, ha = (-6, "right") if right_edge else (6, "left")
        else:
            label, dx, ha = r.backend, -6, "right"
        ax.annotate(label, (r.infer_ms_mean, r.metric_value),
                    textcoords="offset points", xytext=(dx, 5), ha=ha,
                    fontsize=7, color="0.15")
    ax.set_xlabel("inference latency (ms, log scale)")
    ax.set_ylim(-0.05, max((r.metric_value for r in rows), default=1.0) + 0.12)


def save_tradeoff_plot(results: list[CellResult], path) -> Path:
    from matplotlib.lines import Line2D

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = [r for r in results if r.status is CellStatus.OK
          and r.infer_ms_mean is not None and r.metric_value is not None]

    # Stable model -> colour assignment, shared across task panels.
    models = sorted({r.model for r in ok})
    color_of = {m: _MODEL_COLORS[i % len(_MODEL_COLORS)] for i, m in enumerate(models)}
    tasks = sorted({r.task for r in ok})

    fig, axes = plt.subplots(1, max(len(tasks), 1), figsize=(5.0 * max(len(tasks), 1), 3.6),
                             squeeze=False)
    axes = axes[0]
    for ax, task in zip(axes, tasks):
        _plot_tradeoff_ax(ax, [r for r in ok if r.task == task], color_of)
        ax.set_title(f"{task} (CPU)")
    axes[0].set_ylabel("score (task metric)")

    # One backend legend (marker shape); model identity is colour + direct label.
    handles = [Line2D([0], [0], marker=m, linestyle="none", color="0.35",
                      markerfacecolor="0.35", markersize=7, label=b)
               for b, m in _BACKEND_MARKERS.items()]
    axes[-1].legend(handles=handles, title="backend", fontsize=7,
                    title_fontsize=7, frameon=False, loc="lower right")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
