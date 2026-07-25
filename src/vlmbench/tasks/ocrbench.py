"""OCRBench task: OCR-centric visual question answering.

Source: ``echo840/OCRBench`` on the Hugging Face Hub --- the 1000-example
OCRBench suite (Liu et al., arXiv:2305.07895) spanning text recognition,
scene-text VQA, document VQA, key-information extraction, and handwritten-math
recognition. Each row is an image + question + list of accepted answers. The
metric is containment (see :func:`vlmbench.tasks.metrics.containment`), which
mirrors OCRBench's official ``answer in prediction`` scoring.

As with DocVQA, the pool is capped and shuffled deterministically so memory
stays bounded regardless of ``subsample_n``; the orchestrator subsamples
further from the returned pool.
"""
from __future__ import annotations

from typing import Any, Iterable

from .base import Example, TaskSpec
from .metrics import containment

_SOURCE = "echo840/OCRBench"
_SPLIT = "test"
_POOL_CAP = 200
_POOL_SEED = 20260722


def _row_to_example(row: dict[str, Any]) -> Example:
    answers = row.get("answer")
    if isinstance(answers, str):
        answers = [answers]
    elif not answers:
        answers = []
    image = row["image"]
    if hasattr(image, "convert"):  # PIL image
        image = image.convert("RGB")
    return Example(image=image, prompt=row["question"],
                   answers=[str(a) for a in answers])


def _rows_to_examples(rows: Iterable[dict[str, Any]]) -> list[Example]:
    return [_row_to_example(r) for r in rows]


def load_ocrbench(cap: int = _POOL_CAP,
                  seed: int = _POOL_SEED) -> "tuple[list[Example], TaskSpec]":
    from datasets import load_dataset

    ds = load_dataset(_SOURCE, split=_SPLIT)
    if cap and cap < len(ds):
        ds = ds.shuffle(seed=seed).select(range(cap))
    return _rows_to_examples(ds), TaskSpec(name="ocrbench", metric=containment)
