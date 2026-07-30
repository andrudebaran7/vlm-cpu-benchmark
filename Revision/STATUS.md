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

- [ ] **BL1 · Detector baseline (YOLO / RT-DETR on CPU).** 🔴 separate
  experiment; highest impact. The paper claims VLMs could replace detectors but
  never measures a detector. Add YOLOv8n / YOLOv5n (and RT-DETR if feasible) on
  CPU on an equivalent task (e.g. binary "is there text?" or approximate
  localization), measuring latency/memory/accuracy. This is the "bridge" work
  and pairs with the sibling repo `cv-detection-seg-report`. Large but converts
  the paper into "VLMs vs detectors on CPU."

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
