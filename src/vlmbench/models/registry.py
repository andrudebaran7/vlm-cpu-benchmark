from __future__ import annotations

from .base import VLModel
from .florence2 import Florence2Adapter
from .internvl2_5 import InternVL2_5Adapter
from .moondream2 import Moondream2Adapter
from .smolvlm import SmolVLMAdapter


def _lazy(module: str, fn: str):
    def make():
        import importlib
        mod = importlib.import_module(f".detectors.{module}", __package__)
        return getattr(mod, fn)()
    return make


_BUILDERS = {
    "smolvlm-256m": SmolVLMAdapter,
    "moondream2": Moondream2Adapter,
    "florence2-base": Florence2Adapter,
    "internvl2_5-2b": InternVL2_5Adapter,
    "yolo11n": _lazy("ultralytics_detectors", "build_yolo11n"),
    "rt-detr": _lazy("ultralytics_detectors", "build_rtdetr"),
    "yolo-world": _lazy("ultralytics_detectors", "build_yolo_world"),
}


def build_model(name: str) -> VLModel:
    if name not in _BUILDERS:
        raise ValueError(f"unknown model: {name!r}; known: {sorted(_BUILDERS)}")
    return _BUILDERS[name]()


def known_models() -> tuple[str, ...]:
    return tuple(_BUILDERS)
