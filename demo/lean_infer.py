"""Torch-free, memory-disciplined inference for the lean SmolVLM demo.

Runs SmolVLM-256M through the standard adapter's ``onnx-int8`` path (ONNX
Runtime, no torch, no autograd). Loads the model once; the caller resizes the
image and releases memory after each inference.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import gc
import os
import time

from vlmbench.models.registry import build_model
from vlmbench.profiling.memory import sample_peak_rss_mb

MODEL_KEY = "smolvlm-256m"
MAX_SIDE = 512


def resize_max_side(image, max_side: int = MAX_SIDE):
    """Scale ``image`` down so its longest side is <= ``max_side`` (never up)."""
    w, h = image.size
    longest = max(w, h)
    if longest <= max_side:
        return image
    scale = max_side / longest
    return image.resize((round(w * scale), round(h * scale)))


def release_memory() -> None:
    """Return freed heap back to the OS (glibc); no-op elsewhere."""
    gc.collect()
    libc_name = ctypes.util.find_library("c")
    if not libc_name:
        return
    try:
        libc = ctypes.CDLL(libc_name)
        if hasattr(libc, "malloc_trim"):
            libc.malloc_trim(0)
    except OSError:
        pass


class LeanVLM:
    """Loads SmolVLM-256M once on the torch-free onnx-int8 path."""

    def __init__(self) -> None:
        os.environ.setdefault("VLMBENCH_ONNX_LOW_MEM", "1")
        self._model = build_model(MODEL_KEY)
        self._model.load(backend="onnx-int8", dtype="int8")

    def infer(self, image, prompt: str) -> tuple[str, float, float]:
        image = resize_max_side(image)
        start = time.perf_counter()
        try:
            answer, peak_mb = sample_peak_rss_mb(
                lambda: self._model.infer(image, prompt))
            latency_ms = (time.perf_counter() - start) * 1000.0
            return str(answer).strip(), latency_ms, peak_mb
        finally:
            release_memory()
