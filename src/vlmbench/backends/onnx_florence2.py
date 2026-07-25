"""Custom onnxruntime backend for Florence-2's pre-exported ONNX graphs.

Like SmolVLM (see onnx_smolvlm.py), optimum cannot load Florence-2 (custom
architecture), so we drive the onnx-community ONNX graphs directly. Florence-2
is encoder-decoder (BART-style language model over a DaViT vision encoder), so
there are four graphs and the decoder carries both self-attention and
cross-attention KV caches.

Two non-obvious quirks of the onnx-community export (docs/known-issues I16):
  * The ``decoder_with_past_model`` graphs have a static ``inputs_embeds``
    sequence length of 16 in *every* precision, so they cannot be stepped one
    token at a time. We use ``decoder_model_merged`` instead.
  * On the merged decoder's first step (``use_cache_branch=False``) the
    cross-attention past KV must be supplied as real-length zeros
    (``[batch, heads, encoder_len, head_dim]``), not zero-length, or the
    cross-attention MatMul fails.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

_N_LAYERS = 6
_N_HEADS = 12
_HEAD_DIM = 64


def download_onnx_variant(source: str, variant: str) -> Path:
    """Download the four Florence-2 ONNX graphs for a variant; return their dir."""
    from huggingface_hub import hf_hub_download

    names = [f"onnx/vision_encoder_{variant}.onnx",
             f"onnx/embed_tokens_{variant}.onnx",
             f"onnx/encoder_model_{variant}.onnx",
             f"onnx/decoder_model_merged_{variant}.onnx"]
    last = None
    for name in names:
        last = hf_hub_download(source, name)
    return Path(last).parent


class OnnxFlorence2:
    def __init__(self, onnx_dir: Path, variant: str,
                 decoder_start_token_id: int, eos_token_id: int) -> None:
        import onnxruntime as ort

        def sess(name: str):
            return ort.InferenceSession(
                str(Path(onnx_dir) / f"{name}_{variant}.onnx"),
                providers=["CPUExecutionProvider"])

        self._vision = sess("vision_encoder")
        self._embed = sess("embed_tokens")
        self._encoder = sess("encoder_model")
        self._decoder = sess("decoder_model_merged")
        self._dec_start = decoder_start_token_id
        self._eos = eos_token_id

    def _embed_ids(self, ids: np.ndarray) -> np.ndarray:
        return self._embed.run(None, {"input_ids": ids.astype(np.int64)})[0].astype(np.float32)

    def _init_past(self, encoder_len: int) -> dict[str, np.ndarray]:
        past = {}
        for i in range(_N_LAYERS):
            for kv in ("key", "value"):
                past[f"past_key_values.{i}.decoder.{kv}"] = np.zeros(
                    (1, _N_HEADS, 0, _HEAD_DIM), np.float32)
                past[f"past_key_values.{i}.encoder.{kv}"] = np.zeros(
                    (1, _N_HEADS, encoder_len, _HEAD_DIM), np.float32)
        return past

    def generate(self, enc: dict[str, Any], *, max_new_tokens: int) -> list[int]:
        pixel_values = np.asarray(enc["pixel_values"], dtype=np.float32)
        input_ids = np.asarray(enc["input_ids"], dtype=np.int64)

        image_features = self._vision.run(None, {"pixel_values": pixel_values})[0]
        text_embeds = self._embed_ids(input_ids)
        inputs_embeds = np.concatenate([image_features, text_embeds], axis=1).astype(np.float32)
        enc_mask = np.ones(inputs_embeds.shape[:2], dtype=np.int64)
        encoder_hidden = self._encoder.run(
            None, {"attention_mask": enc_mask, "inputs_embeds": inputs_embeds})[0]

        past = self._init_past(encoder_hidden.shape[1])
        generated: list[int] = []
        cur, first = self._dec_start, True
        for _ in range(max_new_tokens):
            feeds = {
                "encoder_attention_mask": enc_mask,
                "encoder_hidden_states": encoder_hidden,
                "inputs_embeds": self._embed_ids(np.array([[cur]])),
                "use_cache_branch": np.array([not first]),
            }
            feeds.update(past)
            names = [o.name for o in self._decoder.get_outputs()]
            od = dict(zip(names, self._decoder.run(None, feeds)))
            cur = int(np.argmax(od["logits"][0, -1]))
            generated.append(cur)
            if cur == self._eos:
                break
            for i in range(_N_LAYERS):
                for kv in ("key", "value"):
                    past[f"past_key_values.{i}.decoder.{kv}"] = od[f"present.{i}.decoder.{kv}"]
                    if first:
                        past[f"past_key_values.{i}.encoder.{kv}"] = od[f"present.{i}.encoder.{kv}"]
            first = False
        return generated
