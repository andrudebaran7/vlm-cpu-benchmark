"""DocVQA task: real document visual-question-answering examples.

Source: ``nielsr/docvqa_1200_examples`` on the Hugging Face Hub — an
ungated 1200-example subset of DocVQA (image + question + answers),
convenient for CPU-budget subsampling. The metric is ANLS, the standard
DocVQA metric (see :func:`vlmbench.tasks.metrics.anls`).

The pool is capped and shuffled deterministically here so that memory
stays bounded regardless of the run's ``subsample_n``; the orchestrator
then subsamples further from the returned pool.
"""
from __future__ import annotations

from typing import Any, Iterable

from .base import Example, TaskSpec
from .metrics import anls

_SOURCE = "nielsr/docvqa_1200_examples"
_SPLIT = "test"
_POOL_CAP = 200
_POOL_SEED = 20260722


def _question_text(query: Any) -> str:
    """The ``query`` field may be a plain string or a multilingual dict."""
    if isinstance(query, dict):
        return query.get("en") or next(iter(query.values()))
    return query


def _row_to_example(row: dict[str, Any]) -> Example:
    """Map one dataset row to an :class:`Example`."""
    answers = row.get("answers")
    if not answers:
        single = row.get("answer")
        answers = [single] if single else []
    image = row["image"]
    if hasattr(image, "convert"):  # PIL image
        image = image.convert("RGB")
    return Example(image=image, prompt=_question_text(row["query"]),
                   answers=[str(a) for a in answers])


def _rows_to_examples(rows: Iterable[dict[str, Any]]) -> list[Example]:
    return [_row_to_example(r) for r in rows]


def load_docvqa(cap: int = _POOL_CAP,
                seed: int = _POOL_SEED) -> "tuple[list[Example], TaskSpec]":
    from datasets import load_dataset

    ds = load_dataset(_SOURCE, split=_SPLIT)
    if cap and cap < len(ds):
        ds = ds.shuffle(seed=seed).select(range(cap))
    return _rows_to_examples(ds), TaskSpec(name="docvqa", metric=anls)
