# Paper improvement roadmap (review response)

_Consolidates the peer-style review (`Revision.MD`) and to-do (`ToDo.md`) with
our own pending items. Each item is tagged by **impact**, **effort**, and
whether it **needs a model re-run** (the expensive constraint — Moondream2 is
~85–216 s/inference, so a full re-run of one task is hours). Last updated:
2026-07-29._

## Legend
- 🟢 no re-run (pure post-processing or prose on existing data) — cheapest
- 🟡 partial re-run (fast models cheap; Moondream2 dominates)
- 🔴 full re-run or new experiment

---

## Quick wins (do first — mostly 🟢)

_Done 2026-07-30 (merged to both repos): QW1, QW2, QW3, W1, W2, ST1, ST2.
Also split Table I into per-task tables and kept the figure error-bar-free
(CIs live in a "Statistical reliability" prose paragraph). Paper is 7 pages._

- [x] **QW1 · Reproducibility-failure taxonomy (Section V).** DONE. Section V
  opens with a named taxonomy (version drift, undeclared deps, corrupt
  artifacts, silent wrong-typed failures, missing artifacts), offered as a
  reusable checklist.
- [x] **QW2 · Energy → cost / normalization.** DONE. Added EUR/1k-inference
  cost at a stated 0.25 EUR/kWh; kept the honest "energy follows latency"
  caveat. (J/token skipped — token counts not stored.)
- [x] **QW3 · Composite efficiency score (supplementary).** DONE as a prose
  sentence (E = acc/(latency×memory)); explicitly framed as a summary that
  does not replace the per-axis tables.

## Statistical rigor (review's #2/#3 — the most defensible criticism)

- [x] **ST1 · Save per-example scores.** DONE. `run_task` returns them,
  `CellResult.per_example_scores` persists them (defaulted, back-compatible).
- [x] **ST2 · Bootstrap CIs.** DONE. Re-ran the three fast models (Moondream2
  kept as point estimate); `report/stats.py` bootstrap; CIs reported in a
  "Statistical reliability" paragraph. Findings: SmolVLM INT8 gain on DocVQA
  is real (intervals clear); OCRBench top-three (Moondream2/InternVL/Florence)
  is a statistical tie — prose corrected accordingly.

## Big lever (review's #1 — the central missing piece)

- [x] **BL1 · Detector baseline (YOLO / RT-DETR on CPU).** DONE (2026-07-30).
  Three detectors (yolo11n, rt-detr, yolo-world) vs the four VLMs on a shared
  binary object-presence task (COCO classes, N=120), same Ryzen, same harness
  (`presence-coco`; Table III in the paper). Findings: detectors dominate the
  trade-off on their home turf — Moondream2 (0.950) is a statistical tie with
  RT-DETR (0.933) but ~410x slower and ~300x more energy; yolo11n does it at
  50 ms / <1 J (~4600x faster, ~3500x cheaper than Moondream2). RF-DETR was
  dropped (requires transformers>=5, incompatible with the VLM env — added to
  the reproducibility taxonomy). **Deferred → "B" below:** the open-vocabulary
  variant (LVIS/OpenImages ground truth) where fixed detectors can't compete.
- [x] **BL1-B · Open-vocabulary detector comparison.** DONE (2026-08-01).
  The `presence-openvocab` variant: five out-of-COCO apparel classes (dress,
  skirt, jacket, hat, shoe) with Fashionpedia ground truth (LVIS/OpenImages
  were unavailable as streamable HF datasets; Fashionpedia streams in the same
  parquet format as COCO). Same seven models, N=120, same harness (Table IV).
  The mirror image of BL1: fixed detectors (yolo11n, rt-detr) can't name the
  classes, short-circuit to "no" at 0 ms / 0 J and land at chance (0.450),
  while the VLMs win decisively (InternVL2.5-2B 0.892, Moondream2 0.867;
  intervals clear YOLO-World's 0.625). So the replace-a-detector answer flips
  with the task: detectors own closed-vocab, VLMs earn their cost on open-vocab.
  Paper integrated (abstract/intro/method/results/future-work). Merged to both
  repos. Minor code follow-ups still open (deferred, non-blocking): cap the
  cached presence example set; thread `config.seed` into the loader.

## Generalization (review's #4 scope)

- [ ] **GEN1 · A third, non-OCR task.** 🔴 full re-run. Captioning (COCO
  subset), visual grounding, or ChartQA to show the ranking generalizes beyond
  OCR-heavy tasks. Costly (Moondream2).
- [ ] **GEN2 · Prompt-robustness probe.** 🟡 re-run (fast models cheap). Vary
  the prompt (e.g. "What does the image say?" / "Extract all text" / "Read the
  text exactly") and report accuracy/latency variance. Small, informative.

## Already-pending (from docs/STATUS.md, still open)

- [ ] **P1 · Florence-2 INT8 regression on OCRBench.** 🟡 investigate why INT8
  was slower + less accurate there (shipped-ONNX quantization quality?).
- [ ] **P2 · Remaining backends** (fp16, OpenVINO INT8, GGUF). 🔴 mostly blocked
  by tooling (documented I13/I16); no model advertises fp16.
- [ ] **P3 · SmolVLM scaling curve** (256M/500M/2.2B). 🔴 within-family scaling —
  overlaps the review's "scaling law" (ToDo #2). Full re-run.
- [ ] **P4 · arXiv submission** (`make arxiv` ready; needs `ARXIV_SUBMISSION.md`).
  🟢 packaging.
- [ ] **P5 · Deploy the lean demo to a free host** (Oracle A1 12 GB ARM). 🟢
  ops, not paper.

## Writing polish (review's §"Mejoras de escritura") — 🟢 all prose

- [x] **W1** DONE. Abstract trimmed 302→199 words, leads with the contribution.
- [x] **W2** DONE. Intro now states "published VLM efficiency figures are not
  comparable under CPU constraints" as the central claim.
- [x] **W3** Discussion already reads findings→implications; left as is.
- [ ] **W4** (optional) more pointed title, e.g. "Do Small VLMs Replace
  Detectors on CPU? A Reproducible Benchmark."

## Notes on the review itself (context for future us)
- The central criticism (**no detector baseline**) is correct and was already
  our planned "bridge" work — highest priority for a venue submission.
- The statistical-rigor criticism is fair and largely 🟢/🟡.
- Some points are partly already addressed and should not be over-corrected:
  energy "follows latency" **is** the honest finding on a single-socket CPU
  (keep the caveat, add cost); the quantization section already has a clear
  supported/unsupported table and the non-trivial "INT8 not always better"
  result; the composite efficiency score is useful only as a supplement.
- For Zenodo/arXiv the paper is essentially ready; the review's asks are what
  separate it from a *workshop/conference* bar.
