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

        # Florence-2 is task-token driven and has no free-form VQA head: it
        # cannot consume `prompt` as a question. For a document task the
        # closest capability is OCR (transcribe the page text); we use that
        # rather than the previous `<MORE_DETAILED_CAPTION>`, which ignored the
        # task entirely and emitted a caption. Florence-2 still returns the
        # whole page rather than a targeted answer, so DocVQA ANLS stays low —
        # a genuine task/model mismatch, not an adapter artifact. See
        # docs/known-issues.md (I11).
        task = "<OCR>"
        inputs = self._processor(text=task, images=image, return_tensors="pt")
        with torch.no_grad():
            # Greedy (num_beams=1) to keep the latency comparison fair with
            # the other adapters, which all decode greedily.
            ids = self._model.generate(input_ids=inputs["input_ids"],
                                       pixel_values=inputs["pixel_values"],
                                       max_new_tokens=256, num_beams=1)
        text = self._processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = self._processor.post_process_generation(
            text, task=task, image_size=(image.width, image.height))
        return str(parsed.get(task, text))
