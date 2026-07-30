"""CLI: build figures/tables from a results.jsonl."""
from __future__ import annotations

import argparse
from pathlib import Path

from vlmbench.report.figures import latex_results_table, save_tradeoff_plot
from vlmbench.report.records import JsonlStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out-dir", default="paper_artifacts")
    args = parser.parse_args()
    results = JsonlStore(args.results).load()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_tradeoff_plot(results, out / "tradeoff.png")
    # Combined table (kept for reference) plus one per-task table, which the
    # paper prefers for readability (they mirror the two figure panels).
    (out / "results_table.tex").write_text(latex_results_table(results))
    tasks = sorted({r.task for r in results})
    for task in tasks:
        (out / f"results_table_{task}.tex").write_text(
            latex_results_table(results, task=task))
    print(f"wrote figures/tables to {out} (per-task tables: {tasks})")


if __name__ == "__main__":
    main()
