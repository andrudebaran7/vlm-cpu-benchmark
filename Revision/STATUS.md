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

- [ ] **QW1 · Reproducibility-failure taxonomy (Section V).** 🟢 low-effort,
  high-differentiator. Restructure the discussion's environment findings into a
  named taxonomy (dependency drift, tokenizer corruption, missing artifacts,
  API instability), each with the concrete instance we hit. This is our unique
  angle and the review's #4 "formalize reproducibility cost." Pure prose.
- [ ] **QW2 · Energy → cost / normalization.** 🟢 low-effort. We already report
  J/inference; add derived cost (€/1k inferences at a stated kWh price) and a
  per-task J/inference framing. Answers review weakness #5. Pure arithmetic on
  existing `energy_j`. (Note: J/**token** needs generated-token counts we do
  not currently store — either estimate at the 64-token cap or skip.)
- [ ] **QW3 · Composite efficiency score (supplementary).** 🟢 low-effort, use
  with care. Add `Efficiency = Accuracy / (Latency × Memory)` (or normalized)
  as a *supplementary* column/caption, NOT the headline — a single number hides
  the accuracy/efficiency trade-off the paper deliberately exposes. Pure
  arithmetic.

## Statistical rigor (review's #2/#3 — the most defensible criticism)

- [ ] **ST1 · Save per-example scores.** 🟢 (code only). `run_task` computes
  per-example scores then discards them; extend it + `CellResult` +
  `results.jsonl` to persist them. Prerequisite for any bootstrap CI. Cheap
  code change; no re-run to *add the capability*.
- [ ] **ST2 · Bootstrap CIs + error bars / boxplots.** 🟡 needs a re-run to
  populate per-example scores (fast models cheap: Florence ~5 min, InternVL
  ~20–30 min, SmolVLM ~30–55 min per task; Moondream2 is the multi-hour
  bottleneck). Once data exists, CIs and boxplots are pure post-processing.
  Decision needed: re-run all four, or report CIs for the three fast models and
  cite Moondream2's N=120 mean as-is.

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

- [ ] **W1** Trim the abstract ~15%; lead with the contribution.
- [ ] **W2** Add a sharp claim to the intro (e.g. "published VLM efficiency
  claims are not comparable under CPU constraints").
- [ ] **W3** Restructure discussion as *findings → implications*.
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
