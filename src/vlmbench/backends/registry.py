from __future__ import annotations

from ..models.base import ModelMeta

KNOWN_BACKENDS: tuple[str, ...] = (
    "fp32", "fp16", "onnx-int8", "openvino-int8", "gguf-q4", "gguf-q8",
)


def validate_backend(backend: str) -> None:
    if backend not in KNOWN_BACKENDS:
        raise ValueError(f"unknown backend: {backend!r}; known: {KNOWN_BACKENDS}")


def backend_supports(meta: ModelMeta, backend: str) -> bool:
    return backend in meta.supported_backends
