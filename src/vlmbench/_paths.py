from __future__ import annotations

import os
from pathlib import Path


def quant_dir(model_name: str, backend: str) -> Path:
    root = Path(os.environ.get("VLMBENCH_CACHE", ".vlmbench_cache"))
    path = root / model_name / backend
    path.mkdir(parents=True, exist_ok=True)
    return path
