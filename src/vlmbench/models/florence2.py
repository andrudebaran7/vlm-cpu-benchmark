from __future__ import annotations

from typing import Any

from .base import ModelMeta

_META = ModelMeta(name="florence2-base", params_b=0.23, license="MIT",
                  source="microsoft/Florence-2-base",
                  supported_backends=("fp32", "onnx-int8"))

# Florence-2 is task-token driven and has no free-form VQA head: it cannot
# consume the question. For a document task the closest capability is OCR
# (transcribe the page text), so DocVQA ANLS stays low regardless of backend
# --- a genuine task/model mismatch, not an adapter artifact (I11).
_TASK = "<OCR>"
_MAX_NEW_TOKENS = 256
# onnx-community ships Florence-2 ONNX at microsoft/Florence-2-base-derived repo.
_ONNX_SOURCE = "onnx-community/Florence-2-base"


class Florence2Adapter:
    def __init__(self, meta: ModelMeta = _META) -> None:
        self._meta = meta
        self._model = None
        self._processor = None
        self._runtime = "torch"
        self._onnx = None

    @property
    def meta(self) -> ModelMeta:
        return self._meta

    def load(self, backend: str, dtype: str) -> None:
        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self._meta.source, trust_remote_code=True)
        if backend == "onnx-int8":
            self._load_onnx("int8")
            return
        self._load_torch()

    def _load_onnx(self, variant: str) -> None:
        from transformers import AutoConfig

        from ..backends.onnx_florence2 import OnnxFlorence2, download_onnx_variant

        tcfg = AutoConfig.from_pretrained(
            self._meta.source, trust_remote_code=True).text_config
        onnx_dir = download_onnx_variant(_ONNX_SOURCE, variant)
        self._onnx = OnnxFlorence2(onnx_dir, variant,
                                   tcfg.decoder_start_token_id, tcfg.eos_token_id)
        self._runtime = "onnx"

    def _load_torch(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        self._model = AutoModelForCausalLM.from_pretrained(
            self._meta.source, torch_dtype=torch.float32,
            trust_remote_code=True).to("cpu").eval()
        self._runtime = "torch"

    def _postprocess(self, ids, image: Any) -> str:
        text = self._processor.batch_decode([ids], skip_special_tokens=False)[0]
        parsed = self._processor.post_process_generation(
            text, task=_TASK, image_size=(image.width, image.height))
        return str(parsed.get(_TASK, text))

    def infer(self, image: Any, prompt: str) -> str:
        if self._runtime == "onnx":
            enc = self._processor(text=_TASK, images=image, return_tensors="np")
            ids = self._onnx.generate(enc, max_new_tokens=_MAX_NEW_TOKENS)
            return self._postprocess(ids, image)
        import torch

        inputs = self._processor(text=_TASK, images=image, return_tensors="pt")
        with torch.no_grad():
            # Greedy (num_beams=1) to keep the latency comparison fair with the
            # other adapters, which all decode greedily.
            ids = self._model.generate(input_ids=inputs["input_ids"],
                                       pixel_values=inputs["pixel_values"],
                                       max_new_tokens=_MAX_NEW_TOKENS, num_beams=1)
        return self._postprocess(ids[0], image)
