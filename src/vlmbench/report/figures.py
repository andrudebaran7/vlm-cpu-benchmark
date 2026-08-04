from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .records import CellResult, CellStatus


def _fmt(value) -> str:
    return "n/a" if value is None else (f"{value:.3f}" if isinstance(value, float) else str(value))


def _fmt_int(value) -> str:
    """Round a wide numeric column (ms / MB / J) to an integer; the
    sub-unit decimals are noise and widen the table."""
    return "n/a" if value is None else f"{value:.0f}"


def _tex(s: str) -> str:
    """Escape LaTeX-special characters in a text cell (e.g. underscores in
    model ids like ``internvl2_5-2b``)."""
    return str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&")


# Paper display names for the internal model ids, so tables and figures show the
# same names the prose uses (audit E8).
_DISPLAY = {
    "smolvlm-256m": "SmolVLM-256M",
    "moondream2": "Moondream2",
    "florence2-base": "Florence-2",
    "internvl2_5-2b": "InternVL2.5-2B",
    "yolo11n": "YOLO11n",
    "rt-detr": "RT-DETR",
    "yolo-world": "YOLO-World",
}


def _name(model: str) -> str:
    return _DISPLAY.get(model, model)


def latex_results_table(results: list[CellResult], task: str | None = None,
                        exclude: tuple[str, ...] = ()) -> str:
    """Render a booktabs results table with a bootstrap 95% CI column.

    If ``task`` is given, only that task's cells are shown and the
    (now-constant) task column is omitted. Models in ``exclude`` are dropped
    (e.g. Florence-2 on the presence tasks, whose ``<OCR>`` adapter emits no
    yes/no answer)."""
    from .stats import bootstrap_ci
    rows = [r for r in results
            if (task is None or r.task == task) and r.model not in exclude]
    if task is None:
        colspec = "lllrlrrr"
        head = "model & backend & task & metric & 95\\% CI"
    else:
        colspec = "llrlrrr"
        head = "model & backend & metric & 95\\% CI"
    header = (f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n"
              f"{head} & infer\\_ms & peak\\_mb & energy\\_j \\\\\n\\midrule\n")
    lines = []
    for r in rows:
        if r.status is CellStatus.OK:
            metric = _fmt(r.metric_value)
            ci = bootstrap_ci(r.per_example_scores) if r.per_example_scores else None
            ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "---"
            infer = _fmt_int(r.infer_ms_mean)
            peak = _fmt_int(r.peak_rss_mb)
            energy = _fmt_int(r.energy_j)
        else:
            metric = ci_str = infer = peak = energy = f"\\text{{{r.status.value}}}"
        cells = [_tex(_name(r.model)), _tex(r.backend)]
        if task is None:
            cells.append(_tex(r.task))
        cells += [metric, ci_str, infer, peak, energy]
        lines.append(" & ".join(cells) + " \\\\")
    body = "\n".join(lines)
    footer = "\n\\bottomrule\n\\end{tabular}"
    return header + body + footer


# Fixed per-model colours (Okabe-Ito, colourblind-safe). Keyed by model id so a
# model keeps the SAME colour in every figure regardless of which subset is
# present -- colour follows the entity, never its rank in the current subset.
_MODEL_COLOR = {
    "smolvlm-256m":   "#D55E00",
    "moondream2":     "#E69F00",
    "florence2-base": "#009E73",
    "internvl2_5-2b": "#0072B2",
    "yolo11n":        "#56B4E9",
    "rt-detr":        "#CC79A7",
    "yolo-world":     "#F0E442",
}
_BACKEND_MARKERS = {"fp32": "o", "onnx-int8": "^"}
_LOG_X_FLOOR = 1.0  # ms; short-circuit detectors report 0 ms (log axis needs >0)


def _color_of(model: str) -> str:
    return _MODEL_COLOR.get(model, "#333333")


def _plot_tradeoff_ax(ax, rows) -> None:
    """Draw one accuracy-vs-latency panel (log-x, colour=model, marker=backend).

    Model identity is carried by the (fixed) colour plus the figure's model
    legend, so points are not directly labelled --- which keeps coincident
    points (e.g. two fixed detectors that both short-circuit at 0 ms) legible."""
    ax.set_xscale("log")  # latency spans ~1ms..215s; log separates the cluster
    ax.grid(True, which="both", linewidth=0.4, color="0.85", zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for r in rows:
        marker = _BACKEND_MARKERS.get(r.backend, "s")
        ax.scatter(max(r.infer_ms_mean, _LOG_X_FLOOR), r.metric_value, s=70,
                   marker=marker, color=_color_of(r.model), edgecolors="white",
                   linewidths=0.8, zorder=3)
    ax.set_xlabel("inference latency (ms, log scale)")
    ax.set_ylim(-0.05, max((r.metric_value for r in rows), default=1.0) + 0.12)


def save_tradeoff_plot(results: list[CellResult], path,
                       exclude: tuple[str, ...] = ()) -> Path:
    from matplotlib.lines import Line2D

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = [r for r in results if r.status is CellStatus.OK
          and r.infer_ms_mean is not None and r.metric_value is not None
          and r.model not in exclude]

    tasks = sorted({r.task for r in ok})

    fig, axes = plt.subplots(1, max(len(tasks), 1), figsize=(5.0 * max(len(tasks), 1), 3.6),
                             squeeze=False)
    axes = axes[0]
    for ax, task in zip(axes, tasks):
        _plot_tradeoff_ax(ax, [r for r in ok if r.task == task])
        ax.set_title(f"{task} (CPU)")
    axes[0].set_ylabel("score (task metric)")

    backends = {r.backend for r in ok}
    # Backend legend (marker shape) only when more than one backend is shown.
    if len(backends) > 1:
        bh = [Line2D([0], [0], marker=m, linestyle="none", color="0.35",
                     markersize=7, label=b)
              for b, m in _BACKEND_MARKERS.items() if b in backends]
        axes[-1].legend(handles=bh, title="backend", fontsize=7,
                        title_fontsize=7, frameon=False, loc="lower right")

    # Shared model legend (fixed colour -> name), in fixed order, below the
    # panels: identity is colour + legend, consistent across every figure.
    present = {r.model for r in ok}
    models_present = ([m for m in _MODEL_COLOR if m in present]
                      + sorted(present - set(_MODEL_COLOR)))
    mh = [Line2D([0], [0], marker="o", linestyle="none", color=_color_of(m),
                 markeredgecolor="white", markersize=8, label=_name(m))
          for m in models_present]
    if mh:
        fig.legend(handles=mh, ncol=min(len(mh), 4), fontsize=8, frameon=False,
                   loc="lower center", bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
