"""yolo11n, rt-detr (fixed COCO vocabulary) and yolo-world (open-vocab)."""
from __future__ import annotations

from typing import Any

from ..base import ModelMeta
from .base_detector import target_class

_THRESHOLD = 0.25
_AGPL = "AGPL-3.0"


class FixedVocabYolo:
    """COCO-vocabulary detector (YOLO11n / RT-DETR). Answers 'no' for any class
    outside its trained vocabulary -- a recorded capability limit."""

    def __init__(self, meta: ModelMeta, weights: str, loader) -> None:
        self._meta = meta
        self._weights = weights
        self._loader = loader
        self._model = None
        self._names: dict[int, str] = {}
        self._threshold = _THRESHOLD

    @property
    def meta(self) -> ModelMeta:
        return self._meta

    def load(self, backend: str, dtype: str) -> None:
        self._model = self._loader(self._weights)
        self._names = {i: n.lower() for i, n in self._model.names.items()}

    def infer(self, image: Any, prompt: str) -> str:
        cls = target_class(prompt)
        if cls not in self._names.values():
            return "no"
        res = self._model(image, verbose=False, conf=self._threshold)
        for r in res:
            for c in r.boxes.cls.tolist():
                if self._names.get(int(c)) == cls:
                    return "yes"
        return "no"


class OpenVocabWorld:
    """YOLO-World: set the queried class as the open-vocabulary text prompt."""

    def __init__(self, meta: ModelMeta, weights: str) -> None:
        self._meta = meta
        self._weights = weights
        self._model = None
        self._threshold = _THRESHOLD

    @property
    def meta(self) -> ModelMeta:
        return self._meta

    def load(self, backend: str, dtype: str) -> None:
        from ultralytics import YOLO
        self._model = YOLO(self._weights)

    def infer(self, image: Any, prompt: str) -> str:
        cls = target_class(prompt)
        self._model.set_classes([cls])
        res = self._model(image, verbose=False, conf=self._threshold)
        return "yes" if any(len(r.boxes) for r in res) else "no"


def build_yolo11n() -> FixedVocabYolo:
    from ultralytics import YOLO
    meta = ModelMeta("yolo11n", 0.0026, _AGPL, "ultralytics/yolo11n", ("fp32",))
    return FixedVocabYolo(meta, "yolo11n.pt", lambda w: YOLO(w))


def build_rtdetr() -> FixedVocabYolo:
    from ultralytics import RTDETR
    meta = ModelMeta("rt-detr", 0.032, _AGPL, "ultralytics/rtdetr-l", ("fp32",))
    return FixedVocabYolo(meta, "rtdetr-l.pt", lambda w: RTDETR(w))


def build_yolo_world() -> OpenVocabWorld:
    meta = ModelMeta("yolo-world", 0.013, _AGPL, "ultralytics/yolov8s-world", ("fp32",))
    return OpenVocabWorld(meta, "yolov8s-world.pt")
