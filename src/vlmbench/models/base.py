from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ModelMeta:
    name: str
    params_b: float
    license: str
    source: str
    supported_backends: tuple[str, ...]


@runtime_checkable
class VLModel(Protocol):
    @property
    def meta(self) -> ModelMeta: ...

    def load(self, backend: str, dtype: str) -> None: ...

    def infer(self, image: Any, prompt: str) -> str: ...
