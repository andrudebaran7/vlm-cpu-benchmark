# vlmbench CPU demo

A small Gradio app that lets you pick a lightweight VLM (SmolVLM-256M,
Moondream2, Florence-2-base), upload an image, and see the model's answer
together with wall-clock inference latency and peak RSS memory — all
running on CPU only.

## Running locally

From the repo root, with the `models` and `demo` extras installed:

```bash
uv pip install -e '.[models]'
pip install -r demo/requirements.txt
python demo/app.py
```

This starts a local Gradio server (default `http://127.0.0.1:7860`).

## Deploying to Hugging Face Spaces

1. Create a new Space with SDK **Gradio** and hardware **CPU Basic**
   (no GPU needed — that's the point of this benchmark).
2. Push (or mirror) this repository to the Space's git remote, so that
   `demo/app.py`, `demo/requirements.txt`, and the `src/vlmbench` package
   are all present in the Space.
3. Set the Space's app file to `demo/app.py` (Space settings ->
   "App file", or add a top-level `app.py` that does
   `from demo.app import demo` and re-exports it, if the Space requires
   the entrypoint at the repo root).
4. Hugging Face Spaces installs `requirements.txt` automatically. If your
   Space only reads a root-level `requirements.txt`, copy or symlink
   `demo/requirements.txt` to the repo root.
5. First launch will download model weights from the Hugging Face Hub
   (SmolVLM-256M, Moondream2, Florence-2-base) — expect a slower cold
   start on first request per model, then cached for the life of the
   Space's persistent storage (if enabled) or the container's runtime.

## Notes

- The demo loads models lazily and caches one instance per model name in
  process memory (`_CACHE` in `app.py`), so switching between models in the
  same session keeps prior models resident (higher RAM, but no reload
  latency on repeat use).
- Backend is fixed to `fp32` in the demo; quantized (`onnx-int8`,
  `openvino-int8`, `gguf-*`) backends are exercised by
  `scripts/run_benchmark.py`, not by this interactive demo.
- Peak RSS is sampled via `vlmbench.profiling.memory.sample_peak_rss_mb`
  and reflects the whole-process resident memory during inference, not an
  isolated per-call delta.

## Lean, portable Streamlit demo (SmolVLM, torch-free)

A single-image demo of **SmolVLM-256M** running in **INT8 ONNX with no
PyTorch**. Upload one image, ask a question, get the answer with latency and
peak RAM. One image at a time, no history; the model loads once and memory is
released after each analysis.

### Run locally

```bash
python -m pip install -r demo/requirements-streamlit.txt   # no torch
streamlit run demo/streamlit_app.py
```

First run downloads ~0.25 GB of INT8 ONNX weights from the Hub.

### Measured footprint

Peak resident memory per inference on this machine (torch-free onnx-int8):
**~2.4 GB** — versus 6–12 GB through the full PyTorch stack. The weights are
~250 MB; the rest is the vision encoder's activation memory (SmolVLM splits
each image into 17 sub-images). Resizing the input bounds it but does not
lower this peak; memory is returned to the OS (`malloc_trim`) after each run.

### Where it runs (fits ~2.4 GB)

- **Local / Raspberry Pi 4/5 (8 GB):** the recommended target — ARM builds of
  onnxruntime work; slower than x86 but fits.
- **Oracle Cloud Always-Free (Ampere A1, 12 GB ARM):** a free public host that
  fits comfortably and reinforces the ARM-portability story (VM setup + Oracle
  A1 capacity permitting).
- **Hugging Face Spaces (CPU Basic, 16 GB):** fits, but creating a compute
  Space now requires a PRO plan.
- **Too small (do not use):** Streamlit Community Cloud (~1 GB) and Render free
  (512 MB).
- **Not included:** Moondream2 / InternVL2.5-2B (no INT8 artifact, ~4 GB
  weights) and Florence-2 (its remote processor hard-requires torch).
