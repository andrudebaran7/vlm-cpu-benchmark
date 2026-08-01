"""Binary object-presence task: "Is there a {class} in this image? yes/no".

This task is shared by detectors and VLMs, providing a fair comparison
on the same harness. Two variants: coco (COCO-80 classes, ground truth
COCO) and openvocab (out-of-COCO apparel classes, ground truth Fashionpedia).

The metric is yesno_match: it extracts the first yes/no token from a
free-form answer, so VLM replies like "Yes." score correctly against the
detectors' exact "yes"/"no".
"""
from __future__ import annotations

import random
from typing import Any, Iterable

from .base import Example, TaskSpec
from .metrics import yesno_match

PROMPT_TEMPLATE = "Is there a {cls} in this image? Answer yes or no."
COCO_CLASSES = ("person", "car", "dog", "chair", "bottle")
# Open-vocabulary variant: fine-grained apparel classes outside the COCO-80,
# from Fashionpedia. Fixed-vocabulary detectors (yolo11n, rt-detr) cannot name
# these; only the open-vocab YOLO-World and the VLMs can.
OPENVOCAB_CLASSES = ("dress", "skirt", "jacket", "hat", "shoe")

# Streaming dataset + target classes per variant.
_VARIANTS = {
    "coco": ("detection-datasets/coco", "val", COCO_CLASSES),
    "openvocab": ("detection-datasets/fashionpedia", "val", OPENVOCAB_CLASSES),
}

_POOL_CAP = 200
_POOL_SEED = 20260722

# Cache built examples by (vocab, cap, seed) so every model in a run sees the
# identical example set (fair comparison) and COCO is streamed only once.
_CACHE: dict[tuple[str, int, int], tuple] = {}


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
    For vocab="openvocab": out-of-COCO apparel classes from Fashionpedia.

    Args:
        vocab: Either "coco" or "openvocab".
        cap: Pool size cap before building examples.
        seed: Random seed.

    Returns:
        Tuple of (examples, TaskSpec).

    Raises:
        ValueError: If vocab is not a known variant.
    """
    if vocab not in _VARIANTS:
        raise ValueError(f"unknown vocab: {vocab!r}; known: {sorted(_VARIANTS)}")

    key = (vocab, cap, seed)
    if key in _CACHE:
        return _CACHE[key]

    from datasets import load_dataset

    dataset_id, split, classes = _VARIANTS[vocab]
    # Stream the detection dataset (materializing a full split is tens of GiB;
    # streaming only fetches the examples we consume). Take ~4x the cap for
    # downsampling headroom.
    ds = load_dataset(dataset_id, split=split, streaming=True)
    id_to_name = ds.features["objects"]["category"].feature.names
    pool = ds.shuffle(seed=seed, buffer_size=max(cap * 4, 500)).take(cap * 4)
    rows = [
        (r["image"], {id_to_name[cat_id] for cat_id in r["objects"]["category"]})
        for r in pool
    ]

    examples = build_presence(rows, classes, seed)
    result = (examples, TaskSpec(name=f"presence-{vocab}", metric=yesno_match))
    _CACHE[key] = result
    return result
