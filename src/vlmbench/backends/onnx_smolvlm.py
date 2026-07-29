"""Custom onnxruntime backend for SmolVLM's pre-exported (Transformers.js-style)
ONNX graphs.

optimum cannot export or load SmolVLM/Idefics3 ONNX (see docs/known-issues.md
I13). SmolVLM-256M, however, ships pre-quantized ONNX on the Hub as three
separate graphs — ``vision_encoder``, ``embed_tokens`` and
``decoder_model_merged`` — each with ``_int8``/``_fp16``/... variants. This
module wires those three graphs into a greedy generation loop so the quantized
weights can actually be benchmarked.

Graph signatures (SmolVLM-256M):
  vision_encoder(pixel_values[b,n,3,512,512], pixel_attention_mask[b,n,512,512])
      -> image_features[n_valid, 64, 576]
  embed_tokens(input_ids[b,s]) -> inputs_embeds[b,s,576]
  decoder_model_merged(inputs_embeds[b,s,576], attention_mask[b,t],
      position_ids[b,s], past_key_values.{0..29}.{key,value}[b,3,p,64])
      -> logits[b,s,49280], present.{0..29}.{key,value}
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

_HIDDEN = 576
_N_LAYERS = 30
_KV_HEADS = 3
_HEAD_DIM = 64


def download_onnx_variant(source: str, variant: str) -> Path:
    """Download the three ONNX graphs for a variant (e.g. 'int8') and return
    the directory that holds them."""
    from huggingface_hub import hf_hub_download

    names = [f"onnx/vision_encoder_{variant}.onnx",
             f"onnx/embed_tokens_{variant}.onnx",
             f"onnx/decoder_model_merged_{variant}.onnx"]
    last = None
    for name in names:
        last = hf_hub_download(source, name)
    return Path(last).parent  # all three land in the same 'onnx' dir


class OnnxSmolVLM:
    """Loads the three ONNX sessions and runs greedy generation on CPU."""

    def __init__(self, onnx_dir: Path, variant: str, image_token_id: int) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 0  # let ORT choose
        # Opt-in low-memory mode (VLMBENCH_ONNX_LOW_MEM=1, e.g. set by the lean
        # demo): disabling the CPU memory arena trims peak RSS for portable
        # deployment at a small latency cost. Default keeps the arena on so the
        # benchmark's measurements are unchanged.
        if os.environ.get("VLMBENCH_ONNX_LOW_MEM") == "1":
            so.enable_cpu_mem_arena = False
            so.enable_mem_pattern = False

        def sess(name: str):
            return ort.InferenceSession(
                str(Path(onnx_dir) / f"{name}_{variant}.onnx"),
                sess_options=so, providers=["CPUExecutionProvider"])

        self._vision = sess("vision_encoder")
        self._embed = sess("embed_tokens")
        self._decoder = sess("decoder_model_merged")
        self._image_token_id = image_token_id
        # dtypes expected by each graph input, resolved from the model.
        self._dt = {i.name: i.type for i in self._vision.get_inputs()}
        self._dt.update({i.name: i.type for i in self._decoder.get_inputs()})

    @staticmethod
    def _np_dtype(onnx_type: str):
        return {"tensor(int64)": np.int64, "tensor(float)": np.float32,
                "tensor(bool)": np.bool_}.get(onnx_type, np.float32)

    def _cast(self, name: str, arr: np.ndarray) -> np.ndarray:
        return arr.astype(self._np_dtype(self._dt.get(name, "tensor(float)")))

    def _merge_image_features(self, input_ids, inputs_embeds, image_features):
        # image_features: [n_valid, 64, 576] -> [n_image_tokens, 576]
        feats = image_features.reshape(-1, _HIDDEN)
        mask = (input_ids[0] == self._image_token_id)
        n = int(mask.sum())
        if n != feats.shape[0]:  # be defensive; use as many as line up
            feats = feats[:n]
        inputs_embeds[0, mask] = feats.astype(inputs_embeds.dtype)
        return inputs_embeds

    def _empty_past(self, batch: int) -> dict[str, np.ndarray]:
        past = {}
        for i in range(_N_LAYERS):
            z = np.zeros((batch, _KV_HEADS, 0, _HEAD_DIM), dtype=np.float32)
            past[f"past_key_values.{i}.key"] = z
            past[f"past_key_values.{i}.value"] = z.copy()
        return past

    def generate(self, enc: dict[str, Any], *, max_new_tokens: int,
                 eos_token_ids: tuple[int, ...]) -> list[int]:
        input_ids = np.asarray(enc["input_ids"], dtype=np.int64)
        batch, seq = input_ids.shape

        # 1) token + image embeddings, merged.
        inputs_embeds = self._embed.run(
            None, {"input_ids": input_ids})[0]
        image_features = self._vision.run(None, {
            "pixel_values": self._cast("pixel_values", np.asarray(enc["pixel_values"])),
            "pixel_attention_mask": self._cast(
                "pixel_attention_mask", np.asarray(enc["pixel_attention_mask"])),
        })[0]
        inputs_embeds = self._merge_image_features(
            input_ids, inputs_embeds, image_features)

        # 2) autoregressive greedy decode.
        attention_mask = np.ones((batch, seq), dtype=np.int64)
        position_ids = np.arange(seq, dtype=np.int64)[None, :]
        past = self._empty_past(batch)
        generated: list[int] = []

        for step in range(max_new_tokens):
            feeds = {"inputs_embeds": self._cast("inputs_embeds", inputs_embeds),
                     "attention_mask": self._cast("attention_mask", attention_mask),
                     "position_ids": self._cast("position_ids", position_ids)}
            feeds.update({k: self._cast(k, v) for k, v in past.items()})
            outputs = self._decoder.run(None, feeds)

            logits = outputs[0]
            next_id = int(np.argmax(logits[0, -1]))
            generated.append(next_id)
            if next_id in eos_token_ids:
                break

            # collect present.* -> next step's past.*
            names = [o.name for o in self._decoder.get_outputs()]
            past = {f"past_key_values.{n.split('.')[1]}.{n.split('.')[2]}": arr
                    for n, arr in zip(names, outputs) if n.startswith("present.")}

            # next step feeds a single new token.
            cur_len = attention_mask.shape[1]
            inputs_embeds = self._embed.run(
                None, {"input_ids": np.array([[next_id]], dtype=np.int64)})[0]
            attention_mask = np.concatenate(
                [attention_mask, np.ones((batch, 1), dtype=np.int64)], axis=1)
            position_ids = np.array([[cur_len]], dtype=np.int64)

        return generated
