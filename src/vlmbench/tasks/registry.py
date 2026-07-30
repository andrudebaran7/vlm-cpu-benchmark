from __future__ import annotations

from .base import Example, TaskSpec
from .metrics import anls

KNOWN_TASKS: tuple[str, ...] = ("sample", "docvqa", "ocrbench", "presence-coco")


def known_tasks() -> tuple[str, ...]:
    return KNOWN_TASKS


def build_task(name: str) -> "tuple[list[Example], TaskSpec]":
    if name == "sample":
        from PIL import Image
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        examples = [Example(image=img, prompt="What color?", answers=["gray"])]
        return examples, TaskSpec(name="sample", metric=anls)
    if name == "docvqa":
        from .docvqa import load_docvqa
        return load_docvqa()
    if name == "ocrbench":
        from .ocrbench import load_ocrbench
        return load_ocrbench()
    if name == "presence-coco":
        from .presence import load_presence
        return load_presence("coco")
    raise ValueError(f"unknown task: {name!r}; known: {sorted(KNOWN_TASKS)}")
