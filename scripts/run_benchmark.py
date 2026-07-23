"""CLI: run the benchmark matrix from a YAML config."""
from __future__ import annotations

import argparse
from pathlib import Path

from vlmbench.config import load_config
from vlmbench.models.registry import build_model
from vlmbench.orchestrator import run_matrix
from vlmbench.report.aggregate import write_csv
from vlmbench.report.records import JsonlStore
from vlmbench.tasks.registry import build_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    cfg = load_config(args.config)
    store = JsonlStore(Path(args.out) / "results.jsonl")
    results = run_matrix(cfg, build_model, build_task, store)
    write_csv(results, Path(args.out) / "results.csv")
    print(f"wrote {len(results)} cells to {args.out}")


if __name__ == "__main__":
    main()
