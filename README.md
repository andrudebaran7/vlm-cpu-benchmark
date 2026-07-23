# vlmbench

A CPU-only efficiency and quantization benchmark framework for small vision language models (VLMs). This tool enables benchmarking of small VLMs on CPU-only environments, measuring performance and resource utilization across different quantization methods and optimization techniques.

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
