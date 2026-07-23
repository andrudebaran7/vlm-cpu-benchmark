from __future__ import annotations

import importlib.metadata
import os
import platform
from typing import Any

_TRACKED = ["torch", "transformers", "onnxruntime", "openvino"]


def _version_or_none(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _torch_threads() -> int | None:
    try:
        import torch

        return torch.get_num_threads()
    except Exception:
        return None


def collect_environment() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count() or 1,
        "torch_threads": _torch_threads(),
        "package_versions": {name: _version_or_none(name) for name in _TRACKED},
    }
