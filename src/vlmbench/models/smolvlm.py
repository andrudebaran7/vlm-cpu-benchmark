from __future__ import annotations

from typing import Any

from .base import ModelMeta

_META = ModelMeta(name="smolvlm-256m", params_b=0.256, license="Apache-2.0",
                  source="HuggingFaceTB/SmolVLM-256M-Instruct",
                  supported_backends=("fp32", "onnx-int8"))

_MAX_NEW_TOKENS = 64


def _chat_text(processor, prompt: str) -> str:
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt}]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True)


class SmolVLMAdapter:
    def __init__(self, meta: ModelMeta = _META) -> None:
        self._meta = meta
        self._model = None
        self._processor = None
        self._runtime = "torch"
        self._onnx = None
        self._eos: tuple[int, ...] = ()

    @property
    def meta(self) -> ModelMeta:
        return self._meta

    def load(self, backend: str, dtype: str) -> None:
        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self._meta.source)
        if backend == "onnx-int8":
            self._load_onnx("int8")
            return
        self._load_torch()

    def _load_onnx(self, variant: str) -> None:
        # SmolVLM/Idefics3 cannot be exported or loaded by optimum; instead we
        # drive the model's own pre-quantized ONNX graphs directly. See
        # docs/known-issues.md (I13) and backends/onnx_smolvlm.py.
        from transformers import AutoConfig

        from ..backends.onnx_smolvlm import OnnxSmolVLM, download_onnx_variant

        cfg = AutoConfig.from_pretrained(self._meta.source)
        onnx_dir = download_onnx_variant(self._meta.source, variant)
        self._onnx = OnnxSmolVLM(onnx_dir, variant, cfg.image_token_id)
        self._eos = _eos_ids(self._processor)
        self._runtime = "onnx"

    def _load_torch(self) -> None:
        import torch

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
        if self._runtime == "onnx":
            return self._infer_onnx(image, prompt)
        return self._infer_torch(image, prompt)

    def _infer_onnx(self, image: Any, prompt: str) -> str:
        text = _chat_text(self._processor, prompt)
        enc = self._processor(text=text, images=[image], return_tensors="np")
        ids = self._onnx.generate(enc, max_new_tokens=_MAX_NEW_TOKENS,
                                  eos_token_ids=self._eos)
        return self._processor.tokenizer.decode(ids, skip_special_tokens=True).strip()

    def _infer_torch(self, image: Any, prompt: str) -> str:
        import torch

        text = _chat_text(self._processor, prompt)
        inputs = self._processor(text=text, images=[image], return_tensors="pt")
        with torch.no_grad():
            ids = self._model.generate(**inputs, max_new_tokens=_MAX_NEW_TOKENS)
        # Slice off the prompt tokens so only the generated answer is decoded;
        # otherwise the echoed chat prompt pollutes the answer and destroys the
        # ANLS score (see docs/known-issues.md I15).
        gen = ids[:, inputs["input_ids"].shape[1]:]
        return self._processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


def _eos_ids(processor) -> tuple[int, ...]:
    tok = processor.tokenizer
    ids = {tok.eos_token_id}
    special = tok.convert_tokens_to_ids("<end_of_utterance>")
    if special is not None and special != tok.unk_token_id:
        ids.add(special)
    return tuple(i for i in ids if i is not None)
