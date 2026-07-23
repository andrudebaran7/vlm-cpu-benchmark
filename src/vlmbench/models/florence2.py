from __future__ import annotations

from typing import Any

from .base import ModelMeta

_META = ModelMeta(name="florence2-base", params_b=0.23, license="MIT",
                  source="microsoft/Florence-2-base",
                  supported_backends=("fp32",))


class Florence2Adapter:
    def __init__(self, meta: ModelMeta = _META) -> None:
        self._meta = meta
        self._model = None
        self._processor = None

    @property
    def meta(self) -> ModelMeta:
        return self._meta

    def load(self, backend: str, dtype: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self._meta.source, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._meta.source, torch_dtype=torch.float32,
            trust_remote_code=True).to("cpu").eval()

    def infer(self, image: Any, prompt: str) -> str:
        import torch

        task = "<MORE_DETAILED_CAPTION>"
        inputs = self._processor(text=task, images=image, return_tensors="pt")
        with torch.no_grad():
            ids = self._model.generate(input_ids=inputs["input_ids"],
                                       pixel_values=inputs["pixel_values"],
                                       max_new_tokens=64, num_beams=1)
        text = self._processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = self._processor.post_process_generation(
            text, task=task, image_size=(image.width, image.height))
        return str(parsed.get(task, text))
