from __future__ import annotations

import os
from pathlib import Path


def _cache_root() -> Path:
    return Path(os.environ.get("VLMBENCH_CACHE", ".vlmbench_cache"))


def quant_dir(model_name: str, backend: str) -> Path:
    path = _cache_root() / model_name / backend
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir(*parts: str) -> Path:
    path = _cache_root().joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
