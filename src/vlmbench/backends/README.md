# Backends

This directory holds quantization/export helpers used by model adapters to
run inference through something other than plain fp32 PyTorch. Each model
adapter declares which backends it supports via `ModelMeta.supported_backends`
(see `src/vlmbench/models/base.py`); `KNOWN_BACKENDS` in `registry.py` is the
global set of backend identifiers the benchmark understands, and a given
model may support any subset of it (`unsupported` otherwise).

## Supported model x backend pairs

| Model            | fp32 | onnx-int8 | openvino-int8 | gguf-q4 / gguf-q8 |
|-------------------|:----:|:---------:|:--------------:|:-----------------:|
| smolvlm-256m       | yes  | yes       | unsupported    | unsupported        |
| florence2-base     | yes  | unsupported | unsupported  | unsupported        |
| internvl2_5        | yes  | unsupported | unsupported  | unsupported        |
| moondream2         | yes  | unsupported | unsupported  | gguf-q4, gguf-q8   |

Backends not listed as `yes` for a model raise/skip via
`backend_supports()` / `validate_backend()` in
`src/vlmbench/backends/registry.py` rather than being silently attempted.

## onnx-int8 (SmolVLM)

optimum cannot export or load SmolVLM/Idefics3 ONNX (see
`docs/known-issues.md` I13). Instead, `onnx_smolvlm.py` drives the model's own
pre-quantized ONNX graphs, which SmolVLM-256M ships on the Hub as three
separate files per precision (`vision_encoder_int8.onnx`,
`embed_tokens_int8.onnx`, `decoder_model_merged_int8.onnx`).

`download_onnx_variant(source, "int8")` downloads those three graphs, and
`OnnxSmolVLM` loads them as three ONNX Runtime sessions and runs a greedy
generation loop on CPU: token + image embeddings are merged at the
`image_token_id` positions, then the decoder is stepped autoregressively with
a KV cache threaded across calls.

`SmolVLMAdapter.load(backend="onnx-int8", ...)` wires this in: it downloads the
int8 variant, constructs `OnnxSmolVLM`, and routes `infer()` through it
(`self._runtime = "onnx"`), decoding only the newly generated tokens.

### Downloading ahead of time

Rather than paying the download cost on first `load()`, pre-fetch with the CLI
helper:

```bash
uv pip install -e '.[quant]' 'optimum[onnxruntime]>=1.20'
.venv/bin/python scripts/export_quantized.py --model smolvlm-256m --backend onnx-int8
```

## Requirements

The `optimum[onnxruntime]`, `onnxruntime`, and (transitively) `torch` /
`transformers` packages are required to run an export or `onnx-int8`
inference; they are intentionally *not* imported at module import time
anywhere in this package (all imports happen inside functions/methods), so
importing `vlmbench.backends.onnx_smolvlm`, `vlmbench.models.smolvlm`, or
`vlmbench._paths` never requires them to be installed. Install them with the
`quant` extra plus `optimum[onnxruntime]` (see command above) before actually
exporting or running the `onnx-int8` backend.
