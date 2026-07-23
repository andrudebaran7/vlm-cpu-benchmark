from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .backends.registry import validate_backend
from .models.registry import known_models
from .tasks.registry import known_tasks


@dataclass
class BenchConfig:
    models: list[str]
    backends: list[str]
    tasks: list[str]
    warmup: int
    repeats: int
    subsample_n: int
    seed: int


def load_config(path) -> BenchConfig:
    data = yaml.safe_load(Path(path).read_text())
    for field in ["models", "backends", "tasks"]:
        value = data.get(field)
        if not value:
            raise ValueError(f"config field {field!r} must be a non-empty list")
        if not isinstance(value, list):
            raise ValueError(f"config field {field!r} must be a list, got {type(value).__name__}")
    for backend in data["backends"]:
        validate_backend(backend)
    known_m = set(known_models())
    for model in data["models"]:
        if model not in known_m:
            raise ValueError(f"unknown model: {model!r}; known: {sorted(known_m)}")
    known_t = set(known_tasks())
    for task in data["tasks"]:
        if task not in known_t:
            raise ValueError(f"unknown task: {task!r}; known: {sorted(known_t)}")
    return BenchConfig(
        models=list(data["models"]),
        backends=list(data["backends"]),
        tasks=list(data["tasks"]),
        warmup=int(data.get("warmup", 1)),
        repeats=int(data.get("repeats", 3)),
        subsample_n=int(data.get("subsample_n", 100)),
        seed=int(data.get("seed", 0)),
    )
