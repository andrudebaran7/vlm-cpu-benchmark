# Lean, portable Streamlit demo — design

_Date: 2026-07-28. Repo: `vlm-cpu-benchmark`._

## Goal

Make the benchmark practical to try, and demonstrate the paper's
CPU-portability thesis, with a **lightweight, torch-free** interactive demo:
pick a small VLM, upload an image, get an answer with its latency and peak
RAM. Small enough to run on constrained hosts (target: Streamlit Community
Cloud and, by extension, a Raspberry Pi 4/5).

**Verified host limits (2026-07-29).** Streamlit Community Cloud allocates a
*variable, shared* per-app budget: RAM **~690 MB (floor) to ~2.7 GB
(ceiling)**, CPU 0.078–2 cores, storage up to 50 GB — not a fixed 1 GB, and
subject to change. A Raspberry Pi 4/5 (8 GB) is the more headroom-generous
target. Implication: our lean footprint must be measured (Step 0) and the
deploy claim stated from that number — comfortably under ~700–900 MB fits
Streamlit Cloud reliably; ~1.5–2 GB would run there only intermittently
(fine on the Pi and locally).

## Why this is feasible (the motivating finding)

The INT8 ONNX weights of the two small models are tiny — SmolVLM-256M
~249 MB, Florence-2-base ~264 MB. The 6–12 GB peak RSS in the benchmark is
**runtime overhead** (PyTorch + transformers + activation memory + glibc
memory retention), not the weights. Dropping PyTorch and running pure ONNX
Runtime should cut the footprint dramatically. This demo both delivers a
practical tool and validates that claim by measuring the lean footprint.

## Scope

**In scope**
- Two models, chosen in the UI:
  - **SmolVLM-256M** — free-form visual question answering.
  - **Florence-2-base** — OCR (transcribe page text) via the `<OCR>` task.
- INT8 ONNX inference only, via the existing `OnnxSmolVLM` / `OnnxFlorence2`
  backends (already numpy + onnxruntime, no torch).
- A Streamlit app showing answer + latency + peak RSS, with the low
  footprint made prominent (the portability story).
- A torch-free `requirements` file and a short deploy note (Streamlit
  Community Cloud / local / Raspberry Pi).

**Out of scope (YAGNI)**
- Moondream2 / InternVL2.5-2B (no INT8 artifact; ~4 GB weights — won't fit).
- GPU, fp32 backends, quantized-weight generation.
- Any benchmarking/comparison UI (that lives in the paper and the CLI).

## Approach

Chosen: **reuse the ONNX backends + run transformers without PyTorch.** The
`OnnxSmolVLM` / `OnnxFlorence2` backends already do inference in numpy +
onnxruntime. The only torch-puller is the transformers `AutoProcessor`
(image preprocessing + tokenizer), and transformers treats torch as an
optional dependency: with `return_tensors="np"` the processor and tokenizer
work without torch installed. So the lean stack is `onnxruntime +
transformers (no torch) + tokenizers + sentencepiece + protobuf +
huggingface_hub + pillow + numpy + streamlit`.

Fallback: if the processor cannot run torch-free for a given model, replace
just that model's preprocessing with a manual PIL+numpy path (resize /
normalize / tokenize). Only used if the spike below fails.

## Components

1. **`demo/lean_infer.py` — torch-free inference wrapper.**
   - `LeanVLM(model_name)`: loads the `AutoProcessor` (numpy) and the
     matching ONNX backend (`OnnxSmolVLM` or `OnnxFlorence2`, INT8 variant),
     caching them.
   - `infer(image, prompt) -> (answer: str, latency_ms: float, peak_rss_mb: float)`:
     runs preprocessing → `backend.generate` → decode/post-process, timing
     the call and sampling peak RSS (reuse `profiling.memory.sample_peak_rss_mb`).
   - Must not import torch anywhere in its module or call graph.

2. **`demo/streamlit_app.py` — the UI.**
   - Model radio/select: "SmolVLM-256M (ask a question)" or
     "Florence-2 (read the text)".
   - Image uploader; a question textbox shown only for SmolVLM (Florence
     uses a fixed `<OCR>` task).
   - "Run" button → calls `LeanVLM.infer`, displays the answer and a small
     metrics row: latency (ms) and peak RSS (MB), with a caption framing the
     footprint ("runs in ~X GB — fits a Raspberry Pi").
   - Models are loaded lazily and cached in `st.session_state` /
     `st.cache_resource` so weights download once.

3. **`demo/requirements-streamlit.txt` — lean deps.** Pinned, torch-free.

4. **`demo/README.md` update** — how to run locally, deploy to Streamlit
   Community Cloud, and the measured footprint + Raspberry Pi note.

## Data flow

```
image (+ question) ──▶ AutoProcessor(np) ──▶ OnnxSmolVLM/OnnxFlorence2.generate
                                                        │
                          answer ◀── decode/post-process ┘
   (latency_ms, peak_rss_mb measured around the whole infer call)
```

## Step 0 — Feasibility spike (gated, do first)

Before building the UI, verify the lean path in a throwaway script:
1. In an environment/subprocess with torch NOT importable (or asserting
   `torch` is never imported), load `AutoProcessor` + the ONNX INT8 backend
   for each model and run one inference on a sample image.
2. Measure peak RSS.

**Success criteria for the spike:** both models produce a correct-looking
answer with no torch import, and peak RSS is measured. Ideally it lands
under ~700–900 MB (fits the Streamlit Cloud floor reliably); up to ~2.7 GB
still fits the Pi and local, and Streamlit Cloud intermittently. If a
processor forces torch, apply the manual-preproc fallback for that model. If
RSS is higher than hoped, record the real number and state the honest
deploy target (Pi / local / HF Spaces) rather than overclaiming the free
Streamlit tier.

## Deployment

- **Local:** `pip install -r demo/requirements-streamlit.txt` then
  `streamlit run demo/streamlit_app.py`.
- **Streamlit Community Cloud:** point the app at `demo/streamlit_app.py`
  with the lean requirements; first run downloads ~0.5 GB of INT8 ONNX.
- **Raspberry Pi 4/5 (8 GB):** same lean stack; ARM onnxruntime; slower but
  fits. Documented as "inferred/expected," measured footprint cited.

## Testing

- A network-free unit test for any pure helper added (e.g. model→variant
  mapping); the existing backend tests already cover generation.
- The spike doubles as the integration check (real inference, real RSS).
- No heavy CI for the Streamlit UI itself.

## Risks

- **Processor pulls torch.** Mitigated by the spike + manual-preproc
  fallback.
- **RSS higher than hoped.** Mitigated: measure first, then state the true
  target tier honestly (Pi / local are safe; the free Streamlit tier is
  borderline at ~690 MB–2.7 GB) rather than overpromising.
- **Cold-start download** on hosted tiers (~0.5 GB). Acceptable; cached
  after first run.

## Success criteria

- `streamlit run demo/streamlit_app.py` works locally with the lean
  (torch-free) requirements, for both models.
- Measured peak RSS is reported and is a large reduction vs the benchmark's
  6–12 GB (validating the portability claim), with the honest deploy target
  stated from the measured number.
- README documents running and deploying it.
