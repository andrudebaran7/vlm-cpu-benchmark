from __future__ import annotations

from typing import Any

from .base import ModelMeta

# NOTE: GGUF backends are intentionally not advertised: no GGUF inference
# path is implemented for moondream2 (its load() below only builds the fp32
# transformers model), so declaring them would silently mislabel fp32
# results as quantized. See docs/known-issues.md (I14).
_META = ModelMeta(name="moondream2", params_b=1.86, license="Apache-2.0",
                  source="vikhyatk/moondream2",
                  supported_backends=("fp32",))


class Moondream2Adapter:
    def __init__(self, meta: ModelMeta = _META) -> None:
        self._meta = meta
        self._model = None
        self._tokenizer = None

    @property
    def meta(self) -> ModelMeta:
        return self._meta

    def load(self, backend: str, dtype: str) -> None:
        if backend != "fp32":
            raise NotImplementedError(
                f"moondream2 has no {backend!r} backend implemented; only fp32.")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._model = AutoModelForCausalLM.from_pretrained(
            self._meta.source, trust_remote_code=True).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(self._meta.source)

    def infer(self, image: Any, prompt: str) -> str:
        # moondream2 exposes a high-level query API in its remote code.
        answer = self._model.query(image, prompt)
        return answer["answer"] if isinstance(answer, dict) else str(answer)
