"""Portability spike: run the two demo models on the torch-free onnx-int8
path, assert no torch import, and print peak RSS. Run manually:

    .venv/bin/python demo/check_lean.py
"""
from __future__ import annotations

import sys

from PIL import Image

from vlmbench.models.registry import build_model
from vlmbench.profiling.memory import sample_peak_rss_mb

_CASES = [
    ("smolvlm-256m", "What is in this image?"),
    ("florence2-base", ""),
]


def main() -> None:
    img = Image.new("RGB", (512, 512), (120, 120, 120))
    for key, prompt in _CASES:
        model = build_model(key)
        model.load(backend="onnx-int8", dtype="int8")
        answer, peak_mb = sample_peak_rss_mb(lambda: model.infer(img, prompt))
        torch_loaded = "torch" in sys.modules
        print(f"{key:16} peak_rss={peak_mb:.0f} MB  torch_imported={torch_loaded}  "
              f"answer={answer!r:.60}")
        assert not torch_loaded, f"{key}: torch was imported on the onnx path!"
    print("OK: both models ran torch-free.")


if __name__ == "__main__":
    main()
