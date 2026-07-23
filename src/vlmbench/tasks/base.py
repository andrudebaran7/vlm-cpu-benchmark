from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable

from ..models.base import VLModel


@dataclass
class Example:
    image: Any
    prompt: str
    answers: list[str]


@dataclass
class TaskSpec:
    name: str
    metric: Callable[[str, list[str]], float]


def subsample(examples: list[Example], n: int, seed: int) -> list[Example]:
    if n >= len(examples):
        return list(examples)
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(examples)), n))
    return [examples[i] for i in idx]


def run_task(model: VLModel, examples: list[Example],
             spec: TaskSpec) -> tuple[float, list[str]]:
    preds: list[str] = []
    scores: list[float] = []
    for ex in examples:
        pred = model.infer(ex.image, ex.prompt)
        preds.append(pred)
        scores.append(spec.metric(pred, ex.answers))
    mean = sum(scores) / len(scores) if scores else 0.0
    return mean, preds
