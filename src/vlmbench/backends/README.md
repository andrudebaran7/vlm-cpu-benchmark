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

`onnx_export.py` provides `export_smolvlm_onnx_int8(source, out_dir) -> Path`,
which:

1. Exports the HuggingFace model at `source` to ONNX via
   `optimum.onnxruntime.ORTModelForVision2Seq.from_pretrained(source, export=True)`.
2. Saves the exported graph to `out_dir`.
3. Dynamically quantizes every `*.onnx` file in `out_dir` to INT8 using
   `optimum.onnxruntime.ORTQuantizer` with an AVX2 per-channel
   `AutoQuantizationConfig`.

`SmolVLMAdapter.load(backend="onnx-int8", dtype="int8")` uses this export
lazily: it checks the on-disk cache (`vlmbench._paths.quant_dir`) for an
existing `*.onnx` artifact and only triggers a fresh export if none is found,
then loads the model with
`optimum.onnxruntime.ORTModelForVision2Seq.from_pretrained(out)` and runs
inference through ONNX Runtime on CPU (`self._runtime = "onnx"`).

The cache directory defaults to `.vlmbench_cache/<model_name>/<backend>/`
relative to the current working directory, or `$VLMBENCH_CACHE/<model_name>/<backend>/`
if that environment variable is set.

### Exporting ahead of time

Rather than paying the export cost on first `load()`, pre-export with the
CLI helper:

```bash
uv pip install -e '.[quant]' 'optimum[onnxruntime]>=1.20'
.venv/bin/python scripts/export_quantized.py --model smolvlm-256m --backend onnx-int8
```

This populates the same cache directory `SmolVLMAdapter.load` looks for, so
subsequent benchmark runs with `backend="onnx-int8"` skip re-exporting.

## Requirements

The `optimum[onnxruntime]`, `onnxruntime`, and (transitively) `torch` /
`transformers` packages are required to run an export or `onnx-int8`
inference; they are intentionally *not* imported at module import time
anywhere in this package (all imports happen inside functions/methods), so
importing `vlmbench.backends.onnx_export`, `vlmbench.models.smolvlm`, or
`vlmbench._paths` never requires them to be installed. Install them with the
`quant` extra plus `optimum[onnxruntime]` (see command above) before actually
exporting or running the `onnx-int8` backend.
