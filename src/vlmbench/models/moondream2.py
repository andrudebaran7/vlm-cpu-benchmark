from __future__ import annotations

from typing import Any

from .base import ModelMeta

_META = ModelMeta(name="moondream2", params_b=1.86, license="Apache-2.0",
                  source="vikhyatk/moondream2",
                  supported_backends=("fp32", "gguf-q4", "gguf-q8"))


class Moondream2Adapter:
    def __init__(self, meta: ModelMeta = _META) -> None:
        self._meta = meta
        self._model = None
        self._tokenizer = None

    @property
    def meta(self) -> ModelMeta:
        return self._meta

    def load(self, backend: str, dtype: str) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._model = AutoModelForCausalLM.from_pretrained(
            self._meta.source, trust_remote_code=True).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(self._meta.source)

    def infer(self, image: Any, prompt: str) -> str:
        # moondream2 exposes a high-level query API in its remote code.
        answer = self._model.query(image, prompt)
        return answer["answer"] if isinstance(answer, dict) else str(answer)
