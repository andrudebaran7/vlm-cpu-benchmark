"""Network-free unit tests for the custom SmolVLM ONNX backend helpers."""
from __future__ import annotations

import numpy as np

from vlmbench.backends.onnx_smolvlm import (
    _HEAD_DIM, _HIDDEN, _KV_HEADS, _N_LAYERS, OnnxSmolVLM,
)


def _bare_runner(image_token_id: int) -> OnnxSmolVLM:
    # Bypass __init__ (which would load ONNX sessions) to test pure logic.
    r = OnnxSmolVLM.__new__(OnnxSmolVLM)
    r._image_token_id = image_token_id
    return r


def test_merge_image_features_places_features_at_image_positions():
    r = _bare_runner(image_token_id=99)
    input_ids = np.array([[1, 99, 99, 2]])  # two image slots
    inputs_embeds = np.zeros((1, 4, _HIDDEN), dtype=np.float32)
    image_features = np.ones((2, 1, _HIDDEN), dtype=np.float32)  # -> 2 x hidden
    out = r._merge_image_features(input_ids, inputs_embeds, image_features)
    assert out[0, 1].sum() == _HIDDEN and out[0, 2].sum() == _HIDDEN  # filled
    assert out[0, 0].sum() == 0 and out[0, 3].sum() == 0  # text slots untouched


def test_empty_past_has_two_tensors_per_layer_with_zero_length():
    r = _bare_runner(image_token_id=0)
    past = r._empty_past(batch=1)
    assert len(past) == 2 * _N_LAYERS
    k0 = past["past_key_values.0.key"]
    assert k0.shape == (1, _KV_HEADS, 0, _HEAD_DIM)
