# vlmbench

A CPU-only efficiency and quantization benchmark framework for small vision language models (VLMs). This tool enables benchmarking of small VLMs on CPU-only environments, measuring performance and resource utilization across different quantization methods and optimization techniques.

## Interactive demo

A lean, torch-free Streamlit demo runs **SmolVLM-256M in INT8 ONNX on CPU** —
upload an image, ask a question, and see the answer with its latency and peak
RAM. It fits modest hardware (Raspberry Pi 8 GB, a free ARM cloud VM, or
locally). See [`demo/README.md`](demo/README.md).

![vlmbench Streamlit demo answering a question about an image, showing latency and peak RAM with no PyTorch loaded](demo/screenshots/demo-result.png)

## Installation

Install the package in development mode with dev dependencies:

```bash
uv pip install -e '.[dev]'
```

To install with model support:

```bash
uv pip install -e '.[models]'
```

## Testing

Run the test suite:

```bash
.venv/bin/python -m pytest -v
```

Run only tests that don't require downloading models:

```bash
.venv/bin/python -m pytest -v -m 'not slow'
```
