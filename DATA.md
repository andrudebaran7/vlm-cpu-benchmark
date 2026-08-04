# Benchmark data

This directory tree contains the canonical measurement data that backs the
paper, so the tables and figures can be reproduced and audited without
re-running the benchmark. These files are intentionally versioned (not
git-ignored).

## Files

| Path | Produced by | Contents |
|---|---|---|
| `results/results.jsonl` | `configs/large_eval.yaml` | DocVQA + OCRBench, 4 VLMs × {fp32, onnx-int8}, N=120 (16 cells) |
| `results_presence_coco/results.jsonl` | `configs/presence_coco.yaml` | Binary object-presence, closed-vocab COCO classes, 7 models, N=120 |
| `results_presence_openvocab/results.jsonl` | `configs/presence_openvocab.yaml` | Binary object-presence, open-vocab apparel (Fashionpedia), 7 models, N=120 |
| `<dir>/results.csv` | `scripts/run_benchmark.py` | Flat summary of the same cells (no per-example scores); regenerated from the JSONL so the two never diverge |
| `paper_artifacts/*.tex`, `*.png` | `scripts/make_figures.py` | Generated LaTeX tables and trade-off figures synced into the paper |

Each JSONL is one JSON object per line (one benchmark *cell* =
model × backend × task). The `.jsonl` is authoritative; the `.csv` is a
convenience view.

## Row schema (JSONL)

```
model, backend, task            identifiers
status                          ok | unsupported | failed | oom
metric_name, metric_value       task accuracy (ANLS / containment / yes-no)
infer_ms_mean, infer_ms_p95     inference latency (see measurement note below)
peak_rss_mb                     peak process resident memory (see note)
energy_j                        per-inference RAPL package energy (see note)
per_example_scores              length-N list of per-example task scores
                                (empty for Moondream2, a point estimate)
error                           traceback string when status=failed
```

## Measurement notes (read before reusing the numbers)

These reflect exactly what the code does, and match the paper's methodology
section:

- **Accuracy** (`metric_value`, `per_example_scores`) is computed over the full
  **N=120** subsample of each task (seeded).
- **Latency, memory, and energy** (`infer_ms_*`, `peak_rss_mb`, `energy_j`) are
  profiled on **one representative example** (the first of the subsample) with
  **1 warm-up iteration and 3 timed repeats**, not averaged over the 120
  accuracy examples. `infer_ms_p95` is therefore the p95 over 3 repeats of the
  same input (machine jitter), not variability across inputs.
- **`peak_rss_mb` is whole-process RSS**, not a model-isolated figure. Models
  are not freed between cells in a run, so a cell's peak can include memory
  retained from earlier cells; treat it as an upper bound, not the model's
  working set.
- **`energy_j`** is package-domain RAPL and is populated by
  `scripts/measure_energy.py`, which re-profiles each cell in a **separate
  session** from the accuracy/latency run. Energy and latency for a given cell
  therefore come from different runs. RAPL measures the whole package including
  idle baseline (no baseline subtraction).
- Fixed-vocabulary detectors (yolo11n, rt-detr) short-circuit to "no" on
  out-of-vocabulary classes without running, so in
  `results_presence_openvocab/` their `infer_ms`/`energy_j` are 0; their
  `peak_rss_mb` still reflects process RSS at that point.

## Regenerating tables and figures

```bash
python scripts/make_figures.py --results results/results.jsonl --out paper_artifacts
# presence tables/figures similarly from the presence result dirs
```
