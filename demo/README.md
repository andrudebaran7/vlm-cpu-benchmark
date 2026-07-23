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
