from __future__ import annotations

from typing import Any, Callable

from .base import ModelMeta


class FakeVLModel:
    def __init__(self, meta: ModelMeta,
                 responder: Callable[[Any, str], str]) -> None:
        self._meta = meta
        self._responder = responder
        self.loaded_backend: str | None = None
        self.loaded_dtype: str | None = None

    @property
    def meta(self) -> ModelMeta:
        return self._meta

    def load(self, backend: str, dtype: str) -> None:
        self.loaded_backend = backend
        self.loaded_dtype = dtype

    def infer(self, image: Any, prompt: str) -> str:
        return self._responder(image, prompt)
