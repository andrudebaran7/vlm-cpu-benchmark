# Known issues & reproducibility notes

This log records practical inconveniences encountered while running the
benchmark on real hardware. It is intended both as engineering
documentation and as source material for the "lessons learned /
reproducibility" discussion in the accompanying paper.

Each entry: symptom → root cause → resolution → status.

---

## I1 — `AutoModelForVision2Seq` removed in `transformers` 5.x

- **Symptom:** running `smolvlm-256m` (fp32) failed immediately with
  `ImportError: cannot import name 'AutoModelForVision2Seq' from
  'transformers'`.
- **Environment:** `transformers==5.14.1`.
- **Root cause:** the `AutoModelForVision2Seq` auto class was renamed to
  `AutoModelForImageTextToText` in `transformers` 5.x. The old name is no
  longer exported.
- **Resolution:** `src/vlmbench/models/smolvlm.py` now imports
  `AutoModelForImageTextToText` with a fallback to the old name for
  older `transformers` installs. The same rename affects
  `optimum.onnxruntime.ORTModelForVision2Seq` used by the onnx-int8
  path (`backends/onnx_export.py`, `smolvlm.py`) — not yet exercised, see
  I4.
- **Status:** RESOLVED (fp32 path).

## I2 — CPU energy (RAPL) not measurable without privileges

- **Symptom:** `energy_j` column is empty for every cell.
- **Root cause:** Intel RAPL counters (`/sys/class/powercap/...` or perf
  energy events) are not readable by an unprivileged user on this
  machine, so the energy profiler yields no reading.
- **Resolution options (undecided):** (a) run the benchmark with
  elevated privileges / adjusted `perf_event_paranoid`; or (b) drop the
  energy dimension from the paper and report latency + peak RSS only.
- **Status:** OPEN — decision pending.

## I3 — First real cell was silently skipped on re-run (resume dedup)

- **Symptom:** after fixing I1, re-running produced `wrote 0 cells` and an
  empty results table.
- **Root cause:** `run_matrix` treats any cell already present in
  `results.jsonl` — including `failed` cells from a previous run — as
  "completed" and skips it (resume/idempotency behaviour). The failed
  cell from the pre-fix run masked the fixed re-run.
- **Resolution:** delete (or use a fresh) `results/` output directory
  when re-running after a fix. Candidate improvement: exclude `failed`
  cells from `completed_keys()` so failures are retried automatically.
- **Status:** WORKAROUND (manual); code improvement candidate.

## I4 — onnx-int8 backend dependencies not installed

- **Symptom:** the `onnx-int8` backend cannot run.
- **Root cause:** it requires the `quant` extra (`onnxruntime`,
  `optimum`, ...) which is not installed; and its export path uses the
  renamed `ORTModelForVision2Seq` (see I1), so a version fix will likely
  be needed there too.
- **Status:** OPEN — deferred; scaling runs use `fp32` only for now.
