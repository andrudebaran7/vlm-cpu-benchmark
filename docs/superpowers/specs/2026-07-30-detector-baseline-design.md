# Detector baseline — shared presence benchmark (design)

_Date: 2026-07-30. Repo: `vlm-cpu-benchmark`. Responds to the review's #1
("no detector baseline") and the roadmap item BL1._

## Goal

Make the paper's central framing — "small VLMs proposed as replacements for
detectors" — testable, by benchmarking object detectors against our VLMs on
the **same CPU, same protocol, and a shared task**, so the comparison is fair
(the paper's whole selling point is fixed-hardware control). The result is a
hybrid: efficiency (latency / memory / energy) plus a binary-presence
accuracy probe, both measured through the existing `vlmbench` harness.

## Motivating context

The sibling report `cv-detection-seg-report` measured detectors (yolo11n
186 ms, rfdetr-nano 446 ms, yolo-world 1057 ms, at 640 px) **on a different
machine (Intel Core i5)** with mAP on COCO. Those numbers are neither on our
hardware (AMD Ryzen 5 5500U) nor on a task comparable to our VLMs'
ANLS/containment. Reusing them would violate the fixed-hardware control we
criticize others for. So we **re-measure detectors on this machine** and make
accuracy comparable via a shared binary task both model families can answer.

## Scope

**In scope**
- Four detectors: **yolo11n**, **rfdetr-nano**, **yolo-world**, **rt-detr**
  (all CPU, fp32/native), wrapped so they answer a binary presence question.
- Our four VLMs on the same task.
- Two task variants:
  - **presence-coco**: target class ∈ the 80 COCO classes; ground truth from
    COCO annotations; balanced ~50/50 yes/no.
  - **presence-openvocab**: target class outside COCO (fine-grained/rare);
    ground truth from LVIS (annotates COCO images with 1200+ categories);
    balanced ~50/50. Fixed-vocab detectors (yolo11n, rfdetr, rt-detr) cannot
    name these classes and answer "no" — a recorded limitation, not omitted.
- Two result tables (accuracy + latency + memory + energy), one per variant,
  covering detectors and VLMs on this machine.
- Prose findings for the paper.

**Out of scope (YAGNI)**
- mAP / box-level accuracy (not comparable to VLM VQA; efficiency + presence
  accuracy is the comparable surface).
- GPU, quantized detectors, segmentation models (SAM2/Mask2Former).
- Re-using the sibling repo's i5 numbers (different hardware).

## Approach

Drive detectors and VLMs through the **same harness** on a shared task. A
detector "answers" the presence question by running detection and checking
whether the queried class is present above a confidence threshold; a VLM is
asked "Is there a {class} in this image? Answer yes or no" and its text is
normalized to yes/no. Ground truth is yes/no from the annotation source.
This makes accuracy apples-to-apples and gives efficiency for free (the
orchestrator already profiles `infer()` for latency, peak RSS, and RAPL
energy). It reuses `models/registry`, `orchestrator`, `profiling/*`, and the
`exact_match` metric unchanged.

## Components

1. **Presence dataset builder** (`tasks/presence.py`).
   - `load_presence(vocab: "coco" | "openvocab", cap, seed) -> (examples, TaskSpec)`.
   - Each `Example` is `(image, "Is there a {class} in this image? Answer yes
     or no.", ["yes"] or ["no"])` with the target class recorded (see the
     class-passing note below). Balanced ~50/50 yes/no via seeded sampling.
   - `coco`: targets drawn from a small set of common COCO classes (e.g.
     person, car, dog, chair, bottle); ground truth = COCO instance
     annotations for the image.
   - `openvocab`: targets drawn from LVIS categories not in COCO's 80; ground
     truth = LVIS annotations for the image.
   - Metric: `exact_match` on the normalized yes/no.

2. **Passing the target class to the model.** The adapter must know which
   class to look for. The prompt text already names it, but parsing free text
   is brittle for detectors. Decision: extend `Example` with an optional
   `meta: dict` (e.g. `{"target_class": "person"}`), defaulted so existing
   tasks are unaffected; `run_task` passes `ex.meta` to `infer` when the model
   accepts it. VLM adapters ignore `meta` (they read the prompt); detector
   adapters read `meta["target_class"]`. (Alternative if `infer` signature
   change is undesirable: encode the class canonically at the start of the
   prompt and have detector adapters parse the first token — decided against
   for brittleness.)

3. **Detector adapters** (`models/detectors/`).
   - `yolo11n.py`, `yolo_world.py`, `rtdetr.py` via `ultralytics`;
     `rfdetr_nano.py` via the `rfdetr` package.
   - Each implements `meta`, `load(backend, dtype)`, and
     `infer(image, prompt, meta=None) -> "yes" | "no"`:
     - Fixed-vocab (yolo11n, rt-detr, rfdetr): map `target_class` to the
       model's class list; if absent, return "no". Else run detection, return
       "yes" iff any box of that class scores ≥ threshold.
     - Open-vocab (yolo-world): set the class as the text prompt, run, return
       "yes" iff any detection ≥ threshold.
   - `ModelMeta` records `params_b`, `license` (ultralytics = **AGPL-3.0**,
     relevant to the paper's licensing discussion), `supported_backends`.

4. **Registry wiring** (`models/registry.py`). Add the four detector keys so
   `build_model` resolves them; detectors coexist with VLMs.

5. **Dependencies** (`pyproject.toml`). New `[detectors]` extra:
   `ultralytics`, `rfdetr` (CPU torch already available via `[models]`).

6. **Configs** (`configs/presence_coco.yaml`, `configs/presence_openvocab.yaml`)
   listing detectors + VLMs, the presence task, N, seed.

7. **Reporting.** Reuse `make_figures.py` / `latex_results_table` (per-task
   table already supported). Two tables; optionally a presence trade-off
   figure. Sync to the paper as before.

## Step 0 — Feasibility spike (gated, do first)

Before building adapters, a throwaway script verifies:
1. **Data**: COCO val images + annotations and an LVIS subset are obtainable
   (HF datasets or official) and we can compute per-image class presence for
   both a COCO class and an LVIS-only class on a handful of images.
2. **Detectors install and run on CPU**: each of the four loads and returns a
   detection on one image; capture per-inference latency and peak RSS.
3. **Presence wiring**: a detector correctly answers yes/no for a known image.

**Success criteria for the spike**: all four detectors run on CPU with
plausible latencies; COCO and LVIS ground truth computable for sample images.
If LVIS is too heavy or unavailable, fall back to a curated open-vocab set
with a documented ground-truth source, or descope open-vocab to a follow-up.
If a detector cannot install/run on CPU, drop it with a note.

## Testing

- Network-free unit tests: `presence` label balancing and row mapping (with a
  fake annotation source); detector-adapter class-mapping logic (fixed-vocab
  returns "no" for an unknown class) with a mocked detector; `exact_match`
  already covers scoring.
- The spike is the integration check (real detectors, real data).
- Slow tests (real model download) marked `slow`, as existing.

## Risks

- **LVIS availability / size** for open-vocab ground truth → spike gate;
  fallback to a curated set or descope.
- **Detector CPU install** (ultralytics/rfdetr wheels, torch versions) →
  spike; drop any detector that won't run, documented.
- **VLM latency on presence** (Moondream2) — small COCO images + short yes/no
  generation should be far faster than DocVQA, but confirm timing before a
  full run; subsample N accordingly.
- **Prompt sensitivity** of VLM yes/no answers — fix one clear prompt; note it.

## Success criteria

- Detectors and VLMs measured on **this machine** on presence-coco and
  presence-openvocab, yielding two tables with accuracy + latency + memory +
  energy.
- The comparison supports an honest, nuanced claim: detectors dominate on
  COCO efficiency; open-vocab generality is where VLMs (and yolo-world) matter
  and fixed detectors fail.
- Framework changes (Example.meta, detector adapters, presence task) land with
  tests and leave existing VLM runs unchanged.
