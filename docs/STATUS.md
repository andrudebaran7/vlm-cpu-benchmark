# Project status

_Snapshot of where the `vlmbench` CPU benchmark and its companion paper
stand. Last updated: 2026-07-25._

Two repositories:

- **`vlm-cpu-benchmark`** (this repo) — the benchmark framework + results.
- **`vlm-cpu-benchmark-paper`** — the arXiv-style paper. Figures/tables are
  generated here and synced across via `make sync`.

## Current results (DocVQA, N=100, CPU)

Per-inference latency/energy; ANLS is the DocVQA metric.

| Model | Backend | ANLS | infer (ms) | peak RSS (MB) | energy (J) |
|---|---|---:|---:|---:|---:|
| moondream2 | fp32 | **0.775** | 215289 | 9557 | 3187 |
| internvl2_5-2b | fp32 | 0.506 | 9665 | 12101 | **133** |
| smolvlm-256m | onnx-int8 | 0.385 | 13153 | 9616 | 193 |
| smolvlm-256m | fp32 | 0.205 | 23272 | 6505 | 332 |
| florence2-base | onnx-int8 | 0.000 | 6494 | 5215 | 101 |
| florence2-base | fp32 | 0.000 | 9444 | 5105 | 139 |

Two INT8 points (SmolVLM, Florence-2): both ~1.45–1.8× faster and lower-energy
than fp32, at no accuracy cost.

**Headline findings (in the paper):**

1. **The accuracy leader is not the practical choice.** Moondream2 is most
   accurate but ~22× slower and ~24× more energy-hungry than InternVL2.5-2B,
   which is the *fastest* model while being second most accurate — it
   strictly dominates Florence-2 and offers the best trade-off overall.
2. **Reproducibility is a first-class cost.** A single `transformers`
   major-version bump disabled 3/4 community models; InternVL2.5 loaded only
   after tracing a crash to one null-character token in its tokenizer.
3. **Quantization tooling is immature.** optimum cannot export/load either
   Idefics3 (SmolVLM) or Florence-2, so a bespoke ONNX Runtime backend was
   hand-written for each. Both INT8 points were faster and lower-energy than
   fp32 at no accuracy cost; moondream2 (f16 GGUF only) and InternVL2.5 (no
   quantized artifact) were not reachable with standard tooling.
4. **Measured accuracy is adapter-sensitive.** A SmolVLM adapter bug had hidden
   a real score (0.000 → 0.205 once fixed); Florence-2's 0.000 is instead a
   genuine task/model mismatch (no VQA head).
5. **Energy tracks latency** on this CPU (~14–15 W package power throughout),
   so it quantifies cost rather than reordering the ranking.

## What works today

- **4/4 models load and run** on CPU (`transformers==4.49`, pinned `<5`).
- **Real DocVQA task** (`nielsr/docvqa_1200_examples`, ANLS).
- **Two quantized backends:** SmolVLM (`backends/onnx_smolvlm.py`, decoder-only)
  and Florence-2 (`backends/onnx_florence2.py`, encoder-decoder), both custom
  ONNX Runtime loops over the models' pre-exported graphs.
- **RAPL energy** measured per cell; readable persistently via a udev rule
  (see below).
- 43 fast tests pass (`pytest -m 'not slow'`).

The full issue log (I1–I16, with root causes and fixes) is in
[`known-issues.md`](known-issues.md).

## Environment notes

- Pinned `transformers==4.49.0`; extra deps `sentencepiece`, `protobuf`.
- InternVL2.5 tokenizer is auto-patched at load (NUL-piece sanitization).
- **RAPL energy** requires readable `energy_uj` counters. Made persistent with
  `/etc/udev/rules.d/99-rapl-readable.rules`:
  ```
  SUBSYSTEM=="powercap", ACTION=="add", RUN+="/bin/chmod o+r /sys%p/energy_uj"
  ```
  (Security note: exposing RAPL enables a power side-channel; fine on a
  personal dev machine, reconsider on shared/production hosts.)

## Pending / future work

- **Quantization for moondream2 / InternVL2.5 — not feasible with standard
      tooling (investigated, closed).** moondream2 ships only an f16 GGUF (no
      INT8) and would need the llama.cpp multimodal toolchain; InternVL2.5-2B
      has no published quantized artifact. Reaching either would mean
      self-quantizing and building a new runtime path — deferred (see I16).
- [ ] **Remaining backends declared but unrun:** fp16, OpenVINO INT8,
      GGUF-q4/q8. (No model currently advertises fp16.)
- [ ] **Larger evaluation:** bigger DocVQA subsample and/or a second task
      (e.g. OCRBench) for more robust accuracy numbers. Cost is dominated by
      moondream2 (~215 s/inference).
- [ ] **YOLO comparison** — the planned "bridge" paper connecting CPU
      efficiency to detector-substitution accuracy.
- [ ] **Additional SmolVLM scales** (500M, 2.2B) for a within-family scaling
      curve.
- [ ] **Florence-2**: only OCR is wired; it has no VQA head, so its DocVQA 0.000
      is expected (documented, not a bug to "fix").

## Reproducing a run

```bash
uv pip install -e '.[models]' 'optimum[onnxruntime]>=1.20'
.venv/bin/python scripts/run_benchmark.py --config configs/docvqa_cpu.yaml --out results
.venv/bin/python scripts/measure_energy.py --config configs/docvqa_cpu.yaml --results results/results.jsonl
.venv/bin/python scripts/make_figures.py --results results/results.jsonl --out-dir paper_artifacts
# then, in the paper repo:  make sync CODE_REPO=../vlm-cpu-benchmark && make pdf
```
