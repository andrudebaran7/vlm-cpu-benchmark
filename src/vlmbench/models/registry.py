from __future__ import annotations

from .base import VLModel
from .florence2 import Florence2Adapter
from .internvl2_5 import InternVL2_5Adapter
from .moondream2 import Moondream2Adapter
from .smolvlm import SmolVLMAdapter

_BUILDERS = {
    "smolvlm-256m": SmolVLMAdapter,
    "moondream2": Moondream2Adapter,
    "florence2-base": Florence2Adapter,
    "internvl2_5-2b": InternVL2_5Adapter,
}


def build_model(name: str) -> VLModel:
    if name not in _BUILDERS:
        raise ValueError(f"unknown model: {name!r}; known: {sorted(_BUILDERS)}")
    return _BUILDERS[name]()


def known_models() -> tuple[str, ...]:
    return tuple(_BUILDERS)
