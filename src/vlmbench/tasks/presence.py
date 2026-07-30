"""Binary object-presence task: "Is there a {class} in this image? yes/no".

This task is shared by detectors and VLMs, providing a fair comparison
on the same harness. Scope A implements COCO variant only (open-vocab is deferred).

The metric is exact_match: a simple yes/no check.
"""
from __future__ import annotations

import random
from typing import Any, Iterable

from .base import Example, TaskSpec
from .metrics import exact_match

PROMPT_TEMPLATE = "Is there a {cls} in this image? Answer yes or no."
COCO_CLASSES = ("person", "car", "dog", "chair", "bottle")

_POOL_CAP = 200
_POOL_SEED = 20260722


def build_presence(
    rows: Iterable[tuple[Any, set[str]]],
    classes: tuple[str, ...],
    seed: int,
) -> list[Example]:
    """Build balanced presence examples from (image, present_classes) rows.

    For each class, collect yes-images (class in present set) and no-images,
    downsample the larger group to balance them, emit one Example per selected
    image asking about that class, then shuffle.

    Args:
        rows: Iterable of (image, present_class_names) tuples.
        classes: Classes to ask about.
        seed: Random seed for downsampling and shuffling.

    Returns:
        List of Examples, shuffled.
    """
    rng = random.Random(seed)
    examples: list[Example] = []

    # Convert rows to list to allow multiple iterations
    rows_list = list(rows)

    for cls in classes:
        yes_images = []
        no_images = []

        # Separate images by class presence
        for image, present_classes in rows_list:
            if cls in present_classes:
                yes_images.append(image)
            else:
                no_images.append(image)

        # Balance by downsampling the larger group
        max_count = min(len(yes_images), len(no_images))
        if max_count > 0:
            yes_images = rng.sample(yes_images, max_count)
            no_images = rng.sample(no_images, max_count)

            # Create examples
            for img in yes_images:
                if hasattr(img, "convert"):
                    img = img.convert("RGB")
                examples.append(
                    Example(
                        image=img,
                        prompt=PROMPT_TEMPLATE.format(cls=cls),
                        answers=["yes"],
                    )
                )

            for img in no_images:
                if hasattr(img, "convert"):
                    img = img.convert("RGB")
                examples.append(
                    Example(
                        image=img,
                        prompt=PROMPT_TEMPLATE.format(cls=cls),
                        answers=["no"],
                    )
                )

    # Shuffle and return
    rng.shuffle(examples)
    return examples


def load_presence(
    vocab: str,
    cap: int = _POOL_CAP,
    seed: int = _POOL_SEED,
) -> "tuple[list[Example], TaskSpec]":
    """Load presence task examples.

    For vocab="coco": Load COCO dataset, shuffle, cap, and build presence examples.
    For vocab="openvocab": Deferred (LVIS unavailable).

    Args:
        vocab: Either "coco" or "openvocab".
        cap: Pool size cap before building examples.
        seed: Random seed.

    Returns:
        Tuple of (examples, TaskSpec).

    Raises:
        NotImplementedError: If vocab is not "coco".
    """
    if vocab == "openvocab":
        raise NotImplementedError(
            "presence-openvocab deferred: LVIS unavailable (see plan scope adjustment)"
        )

    if vocab != "coco":
        raise ValueError(f"unknown vocab: {vocab!r}; known: coco")

    from datasets import load_dataset

    # Load COCO dataset and cap at 4x pool cap (for more downsampling headroom)
    ds = load_dataset("detection-datasets/coco", split="val")
    ds = ds.shuffle(seed=seed).select(range(min(cap * 4, len(ds))))

    # Extract (image, present_class_names) from dataset
    id_to_name = ds.features["objects"].feature["category"].names
    rows = [
        (r["image"], {id_to_name[cat_id] for cat_id in r["objects"]["category"]})
        for r in ds
    ]

    # Build presence examples
    examples = build_presence(rows, COCO_CLASSES, seed)

    return examples, TaskSpec(name="presence-coco", metric=exact_match)
