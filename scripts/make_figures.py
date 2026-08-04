"""CLI: build figures/tables from a results.jsonl."""
from __future__ import annotations

import argparse
from pathlib import Path

from vlmbench.report.figures import latex_results_table, save_tradeoff_plot
from vlmbench.report.records import JsonlStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, nargs="+",
                        help="one or more results.jsonl files (merged)")
    parser.add_argument("--out-dir", default="paper_artifacts")
    parser.add_argument("--figure", default="tradeoff.png",
                        help="filename for the trade-off figure")
    parser.add_argument("--exclude", default="",
                        help="comma-separated model ids to drop (e.g. Florence-2 "
                             "on presence, whose <OCR> adapter emits no yes/no)")
    args = parser.parse_args()
    results = [r for f in args.results for r in JsonlStore(f).load()]
    exclude = tuple(m for m in args.exclude.split(",") if m)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_tradeoff_plot(results, out / args.figure, exclude=exclude)
    # Combined table (kept for reference) plus one per-task table, which the
    # paper prefers for readability (they mirror the two figure panels).
    (out / "results_table.tex").write_text(latex_results_table(results, exclude=exclude))
    tasks = sorted({r.task for r in results})
    for task in tasks:
        (out / f"results_table_{task}.tex").write_text(
            latex_results_table(results, task=task, exclude=exclude))
    print(f"wrote figures/tables to {out} (per-task tables: {tasks}; "
          f"excluded: {exclude or 'none'})")


if __name__ == "__main__":
    main()
