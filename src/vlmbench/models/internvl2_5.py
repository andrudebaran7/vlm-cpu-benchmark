from __future__ import annotations

from typing import Any

from .base import ModelMeta

_META = ModelMeta(
    name="internvl2_5-2b",
    params_b=2.0,
    license="MIT (project); verify base-LLM license of the checkpoint",
    source="OpenGVLab/InternVL2_5-2B",
    supported_backends=("fp32",),
)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class InternVL2_5Adapter:
    def __init__(self, meta: ModelMeta = _META) -> None:
        self._meta = meta
        self._model = None
        self._tokenizer = None

    @property
    def meta(self) -> ModelMeta:
        return self._meta

    def load(self, backend: str, dtype: str) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._model = AutoModel.from_pretrained(
            self._meta.source,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._meta.source, trust_remote_code=True, use_fast=False
        )

    def _pixel_values(self, image: Any):
        import numpy as np
        import torch

        img = image.convert("RGB").resize((448, 448))
        arr = np.asarray(img).astype("float32") / 255.0  # H, W, C
        mean = np.array(_IMAGENET_MEAN, dtype="float32")
        std = np.array(_IMAGENET_STD, dtype="float32")
        arr = (arr - mean) / std
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1, C, H, W

    def infer(self, image: Any, prompt: str) -> str:
        pixel_values = self._pixel_values(image)
        question = f"<image>\n{prompt}"
        response = self._model.chat(
            self._tokenizer,
            pixel_values,
            question,
            generation_config=dict(max_new_tokens=64, do_sample=False),
        )
        return str(response)
