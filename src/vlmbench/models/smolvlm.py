from __future__ import annotations

from typing import Any

from .base import ModelMeta

_META = ModelMeta(name="smolvlm-256m", params_b=0.256, license="Apache-2.0",
                  source="HuggingFaceTB/SmolVLM-256M-Instruct",
                  supported_backends=("fp32", "onnx-int8"))


class SmolVLMAdapter:
    def __init__(self, meta: ModelMeta = _META) -> None:
        self._meta = meta
        self._model = None
        self._processor = None
        self._runtime = "torch"

    @property
    def meta(self) -> ModelMeta:
        return self._meta

    def load(self, backend: str, dtype: str) -> None:
        import torch
        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self._meta.source)
        if backend == "onnx-int8":
            from optimum.onnxruntime import ORTModelForVision2Seq
            from .._paths import quant_dir  # helper returning a cache path
            from ..backends.onnx_export import export_smolvlm_onnx_int8

            out = quant_dir(self._meta.name, "onnx-int8")
            if not any(out.glob("*.onnx")):
                export_smolvlm_onnx_int8(self._meta.source, out)
            self._model = ORTModelForVision2Seq.from_pretrained(out)
            self._runtime = "onnx"
            return
        try:
            # transformers >= 5 renamed the auto class.
            from transformers import AutoModelForImageTextToText as _AutoVLM
        except ImportError:  # pragma: no cover - older transformers
            from transformers import AutoModelForVision2Seq as _AutoVLM
        try:
            # transformers >= 5 renamed the dtype kwarg to `dtype`.
            model = _AutoVLM.from_pretrained(self._meta.source, dtype=torch.float32)
        except TypeError:  # transformers 4.x uses `torch_dtype`
            model = _AutoVLM.from_pretrained(self._meta.source, torch_dtype=torch.float32)
        self._model = model.to("cpu").eval()
        self._runtime = "torch"

    def infer(self, image: Any, prompt: str) -> str:
        import torch

        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=text, images=[image], return_tensors="pt")
        with torch.no_grad():
            ids = self._model.generate(**inputs, max_new_tokens=64)
        return self._processor.batch_decode(ids, skip_special_tokens=True)[0]
