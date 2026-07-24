"""Patch energy_j into an existing results.jsonl.

RAPL energy is captured only in the per-cell profiling block (one example x
warmup+repeats), not in the N-example accuracy pass, so energy can be measured
without re-running the whole benchmark. For each ``ok`` cell this reproduces
the same profiling protocol and fills in ``energy_j``, leaving accuracy and
latency untouched. Requires readable powercap counters (see docs/known-issues
I2).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlmbench.config import load_config
from vlmbench.models.registry import build_model
from vlmbench.orchestrator import _profile_first_example
from vlmbench.tasks.base import subsample
from vlmbench.tasks.registry import build_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    task = cfg.tasks[0]
    examples_all, _spec = build_task(task)
    examples = subsample(examples_all, cfg.subsample_n, cfg.seed)
    example0 = examples[0]

    rows = [json.loads(l) for l in Path(args.results).read_text().splitlines()]
    for row in rows:
        if row["status"] != "ok" or row["task"] != task:
            continue
        model = build_model(row["model"])
        model.load(backend=row["backend"], dtype="float32")
        profile = _profile_first_example(model, example0, cfg.warmup, cfg.repeats)
        row["energy_j"] = profile.energy_j
        print(f"{row['model']:16} {row['backend']:10} energy_j="
              f"{profile.energy_j}")

    with open(args.results, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
