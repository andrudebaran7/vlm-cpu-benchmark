"""Network-free unit tests for the Florence-2 ONNX backend helpers."""
from __future__ import annotations

from vlmbench.backends.onnx_florence2 import (
    _HEAD_DIM, _N_HEADS, _N_LAYERS, OnnxFlorence2,
)


def _bare() -> OnnxFlorence2:
    return OnnxFlorence2.__new__(OnnxFlorence2)


def test_init_past_shapes():
    # decoder KV starts empty (grows); encoder KV is real-length zeros (the
    # merged decoder's first step needs the cross-attn length, not zero).
    past = _bare()._init_past(encoder_len=587)
    assert len(past) == 4 * _N_LAYERS  # decoder+encoder x key+value per layer
    assert past["past_key_values.0.decoder.key"].shape == (1, _N_HEADS, 0, _HEAD_DIM)
    assert past["past_key_values.0.encoder.key"].shape == (1, _N_HEADS, 587, _HEAD_DIM)
