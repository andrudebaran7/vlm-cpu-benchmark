# Detector Baseline — Shared Presence Benchmark: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Benchmark four object detectors (yolo11n, rfdetr-nano, yolo-world, rt-detr) against our four VLMs on a shared binary **presence** task, on this machine, through the existing harness — yielding comparable accuracy + latency + memory + energy on two variants (COCO classes, open-vocab via LVIS).

**Architecture:** Detectors and VLMs answer "Is there a {class} in this image? Answer yes or no." A detector adapter runs detection and returns "yes"/"no" (parsing the target class from the fixed-format prompt — no framework signature change). A presence-task builder produces balanced examples with yes/no ground truth. The unchanged orchestrator profiles every cell (latency/RSS/energy) and scores with `exact_match`.

**Tech Stack:** Python, `ultralytics` (yolo11n, yolo-world, rt-detr), `rfdetr`, HF `datasets` (COCO + LVIS), the existing `vlmbench` harness (orchestrator, profiling, records, stats).

## Global Constraints

- **Same hardware.** All measurements on this machine (AMD Ryzen 5 5500U); the sibling repo's i5 numbers are NOT reused.
- **Shared task, one prompt template, verbatim:** `Is there a {cls} in this image? Answer yes or no.` Detectors parse `{cls}` from this template; VLMs answer it. No change to the `VLModel.infer(image, prompt)` signature or to any VLM adapter.
- **Two variants:** `presence-coco` (targets ∈ COCO-80, ground truth COCO) and `presence-openvocab` (targets ∈ LVIS-minus-COCO, ground truth LVIS). Balanced ~50/50 yes/no via seeded sampling.
- **Five target classes per variant.** COCO set: `person, car, dog, chair, bottle`. Open-vocav set: chosen from LVIS-not-COCO in the spike (candidates: `traffic_cone, whiteboard, sunglasses, thermos, dumbbell`).
- **N=120 per cell, seed 20260722** (matches the paper's VLM runs; ~24 examples/class). Aggregate accuracy is reported with a bootstrap CI (`vlmbench.report.stats.bootstrap_ci`).
- **Metric:** `exact_match` on normalized yes/no.
- **Detectors are AGPL-3.0 (ultralytics)** — record in `ModelMeta.license`; ties to the paper's licensing discussion.
- **Confidence threshold 0.25** for a detection to count as presence (standard default).
- Run Python via `.venv/bin/python` / `.venv/bin/python -m pytest`.

---

### Task 1: Feasibility spike (gated — do first)

De-risks data and detector install before building. Confirms COCO + LVIS presence is computable and all four detectors run on CPU, and records their latencies. Keepable check script.

**Files:**
- Create: `scripts/check_detectors.py`

**Interfaces:**
- Produces: a runnable check; prints per-detector `(answer, latency_ms, peak_rss_mb)` on one image and confirms COCO + LVIS per-image class presence for a sample.

- [ ] **Step 1: Install detector deps**

Run: `uv pip install ultralytics rfdetr`
Expected: installs without error (CPU torch already present from `[models]`).

- [ ] **Step 2: Write the spike script**

Create `scripts/check_detectors.py`:

```python
"""Feasibility spike for the detector baseline. Verifies the four detectors
run on CPU and that COCO/LVIS per-image class presence is computable.
Run: .venv/bin/python scripts/check_detectors.py
"""
from __future__ import annotations

import time

from PIL import Image

from vlmbench.profiling.memory import sample_peak_rss_mb


def _time(fn):
    t0 = time.perf_counter()
    out, peak = sample_peak_rss_mb(fn)
    return out, (time.perf_counter() - t0) * 1000.0, peak


def check_detectors() -> None:
    from ultralytics import YOLO, RTDETR

    img = Image.new("RGB", (640, 640), (120, 120, 120))
    runners = {
        "yolo11n": lambda: YOLO("yolo11n.pt")(img, verbose=False),
        "yolo-world": lambda: YOLO("yolov8s-world.pt")(img, verbose=False),
        "rt-detr": lambda: RTDETR("rtdetr-l.pt")(img, verbose=False),
    }
    for name, fn in runners.items():
        _, ms, peak = _time(fn)
        print(f"{name:12} ran on CPU: {ms:.0f} ms, peak {peak:.0f} MB")
    try:
        from rfdetr import RFDETRNano
        m = RFDETRNano()
        _, ms, peak = _time(lambda: m.predict(img))
        print(f"{'rfdetr-nano':12} ran on CPU: {ms:.0f} ms, peak {peak:.0f} MB")
    except Exception as exc:  # record, do not fix here
        print(f"rfdetr-nano FAILED to run: {type(exc).__name__}: {exc}")


def check_data() -> None:
    from datasets import load_dataset
    # COCO: confirm we can read an image + its object categories.
    coco = load_dataset("detection-datasets/coco", split="val", streaming=True)
    ex = next(iter(coco))
    print("COCO example keys:", list(ex.keys()))
    # LVIS: confirm availability + categories outside COCO-80.
    try:
        lvis = load_dataset("lvis", split="validation", streaming=True)
        lex = next(iter(lvis))
        print("LVIS example keys:", list(lex.keys()))
    except Exception as exc:
        print(f"LVIS load FAILED: {type(exc).__name__}: {exc} "
              f"(fallback: curated open-vocab set, or descope open-vocab)")


if __name__ == "__main__":
    check_detectors()
    check_data()
```

- [ ] **Step 3: Run the spike**

Run: `.venv/bin/python scripts/check_detectors.py`
Expected: each detector prints a CPU latency; COCO example keys print with an object/category field; LVIS either prints keys or a clear fallback message.

- [ ] **Step 4: Decision gate — record findings, adjust downstream**

Write the observed detector latencies and the confirmed COCO/LVIS field names into the report. Then:
- If a detector cannot run on CPU, drop it (note it) — Task 3 builds only the runnable ones.
- Confirm the exact dataset ids + the field giving per-image category names/ids (Task 2 uses these). If `detection-datasets/coco` lacks category *names*, resolve ids via its category map.
- If LVIS is unavailable/too heavy, either use a curated open-vocab set with a documented ground-truth source or descope `presence-openvocab` to a follow-up (note in Task 2).

- [ ] **Step 5: Commit**

```bash
git add scripts/check_detectors.py
git commit -m "spike: detector CPU + COCO/LVIS presence feasibility check"
```

---

### Task 2: Presence task and dataset builder

Builds balanced presence examples with yes/no ground truth for both variants, scored by `exact_match`. Uses the dataset ids/fields confirmed in Task 1.

**Files:**
- Create: `src/vlmbench/tasks/presence.py`
- Modify: `src/vlmbench/tasks/registry.py`
- Test: `tests/tasks/test_presence.py`

**Interfaces:**
- Consumes: `Example`, `TaskSpec`, `exact_match`.
- Produces:
  - `PROMPT_TEMPLATE = "Is there a {cls} in this image? Answer yes or no."`
  - `COCO_CLASSES: tuple[str, ...]`, `OPENVOCAB_CLASSES: tuple[str, ...]`.
  - `build_presence(rows, classes, seed) -> list[Example]` where `rows` is an iterable of `(image, present_class_names: set[str])`; produces balanced yes/no examples (each Example prompt asks about one target class; the answer is `["yes"]` if that class ∈ the image's present set else `["no"]`).
  - `load_presence(vocab: str, cap: int = 200, seed: int = 20260722) -> tuple[list[Example], TaskSpec]`.
  - Registered task names `presence-coco`, `presence-openvocab`.

- [ ] **Step 1: Write the failing test (network-free, fake rows)**

Create `tests/tasks/test_presence.py`:

```python
from vlmbench.tasks.presence import (
    PROMPT_TEMPLATE, build_presence, COCO_CLASSES,
)
from vlmbench.tasks.registry import known_tasks


class _Img:
    def convert(self, mode):
        return self


def test_presence_registered():
    assert "presence-coco" in known_tasks()
    assert "presence-openvocab" in known_tasks()


def test_build_presence_labels_yes_no_from_present_set():
    rows = [(_Img(), {"person"}), (_Img(), {"car", "dog"})]
    ex = build_presence(rows, classes=("person", "car"), seed=1)
    # Each example asks about one class; answer is yes iff present in that image.
    labels = {(e.prompt, e.answers[0]) for e in ex}
    assert (PROMPT_TEMPLATE.format(cls="person"), "yes") in labels  # img0 has person
    assert (PROMPT_TEMPLATE.format(cls="car"), "no") in labels      # img0 lacks car
    assert (PROMPT_TEMPLATE.format(cls="car"), "yes") in labels     # img1 has car


def test_build_presence_is_balanced():
    # 4 images, class 'person' present in exactly 2 of them.
    rows = [(_Img(), {"person"}), (_Img(), {"person"}),
            (_Img(), set()), (_Img(), set())]
    ex = build_presence(rows, classes=("person",), seed=1)
    yes = sum(e.answers[0] == "yes" for e in ex)
    no = sum(e.answers[0] == "no" for e in ex)
    assert yes == no == 2  # balanced
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/tasks/test_presence.py -v`
Expected: FAIL (`ModuleNotFoundError: vlmbench.tasks.presence`).

- [ ] **Step 3: Write the implementation**

Create `src/vlmbench/tasks/presence.py`. (Adjust `_load_rows` dataset ids/fields to those confirmed in Task 1.)

```python
"""Binary object-presence task shared by detectors and VLMs.

Every example asks, with a fixed template, whether one target class is present
in an image; the answer is yes/no from annotations. Two variants: `coco`
(targets in the COCO-80, ground truth COCO) and `openvocab` (targets outside
COCO, ground truth LVIS). Balanced ~50/50 yes/no by seeded downsampling of the
majority label per class.
"""
from __future__ import annotations

import random
from typing import Any, Iterable

from .base import Example, TaskSpec
from .metrics import exact_match

PROMPT_TEMPLATE = "Is there a {cls} in this image? Answer yes or no."
COCO_CLASSES = ("person", "car", "dog", "chair", "bottle")
OPENVOCAB_CLASSES = ("traffic cone", "whiteboard", "sunglasses", "thermos",
                     "dumbbell")
_SEED = 20260722


def build_presence(rows: Iterable[tuple[Any, set[str]]],
                   classes: tuple[str, ...], seed: int = _SEED) -> list[Example]:
    """Balanced yes/no presence examples. For each class, collect its yes and
    no images across ``rows``, then downsample the larger group so the class is
    ~50/50."""
    rows = list(rows)
    rng = random.Random(seed)
    out: list[Example] = []
    for cls in classes:
        yes_imgs = [im for im, present in rows if cls in present]
        no_imgs = [im for im, present in rows if cls not in present]
        k = min(len(yes_imgs), len(no_imgs))
        if k == 0:
            continue
        for im in rng.sample(yes_imgs, k):
            out.append(_example(im, cls, "yes"))
        for im in rng.sample(no_imgs, k):
            out.append(_example(im, cls, "no"))
    rng.shuffle(out)
    return out


def _example(image, cls: str, answer: str) -> Example:
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    return Example(image=image, prompt=PROMPT_TEMPLATE.format(cls=cls),
                   answers=[answer])


def _load_rows(vocab: str, pool_cap: int, seed: int):
    """Yield (image, present_class_names) from the confirmed dataset. COCO for
    'coco'; LVIS for 'openvocab'. Field names per the Task 1 spike."""
    from datasets import load_dataset

    if vocab == "coco":
        ds = load_dataset("detection-datasets/coco", split="val")
        names = ds.features["objects"].feature["category"].names  # id->name
        ds = ds.shuffle(seed=seed).select(range(min(pool_cap, len(ds))))
        for r in ds:
            present = {names[c] for c in r["objects"]["category"]}
            yield r["image"], present
    else:
        ds = load_dataset("lvis", split="validation")
        # LVIS category names resolved per the spike; keep only LVIS-not-COCO.
        for image, present in _lvis_rows(ds, pool_cap, seed):
            yield image, present


def _lvis_rows(ds, pool_cap: int, seed: int):
    ds = ds.shuffle(seed=seed).select(range(min(pool_cap, len(ds))))
    for r in ds:
        yield r["image"], set(r.get("category_names", []))


def load_presence(vocab: str, cap: int = 200,
                  seed: int = _SEED) -> tuple[list[Example], TaskSpec]:
    classes = COCO_CLASSES if vocab == "coco" else OPENVOCAB_CLASSES
    rows = _load_rows(vocab, pool_cap=cap * 4, seed=seed)
    examples = build_presence(rows, classes, seed=seed)
    return examples, TaskSpec(name=f"presence-{vocab}", metric=exact_match)
```

- [ ] **Step 4: Register the tasks**

In `src/vlmbench/tasks/registry.py`, add to `KNOWN_TASKS` and `build_task`:

```python
KNOWN_TASKS: tuple[str, ...] = ("sample", "docvqa", "ocrbench",
                                "presence-coco", "presence-openvocab")
```
and in `build_task`:
```python
    if name in ("presence-coco", "presence-openvocab"):
        from .presence import load_presence
        return load_presence(name.split("-", 1)[1])
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/tasks/test_presence.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Full fast suite**

Run: `.venv/bin/python -m pytest -q -m 'not slow'`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/vlmbench/tasks/presence.py src/vlmbench/tasks/registry.py tests/tasks/test_presence.py
git commit -m "feat: binary object-presence task (coco + openvocab variants)"
```

---

### Task 3: Detector adapters

Wrap the runnable detectors (from Task 1) so `infer(image, prompt)` returns "yes"/"no", parsing the target class from the fixed prompt template. Fixed-vocab detectors return "no" for a class they cannot name.

**Files:**
- Create: `src/vlmbench/models/detectors/__init__.py`, `.../base_detector.py`, `.../ultralytics_detectors.py`, `.../rfdetr_nano.py`
- Modify: `src/vlmbench/models/registry.py`, `pyproject.toml`
- Test: `tests/models/test_detectors.py`

**Interfaces:**
- Consumes: `ModelMeta`, the `VLModel` protocol (`meta`, `load`, `infer`).
- Produces: builder keys `yolo11n`, `yolo-world`, `rt-detr`, `rfdetr-nano`; a helper `target_class(prompt) -> str`.

- [ ] **Step 1: Write the failing test (network-free, mocked detector)**

Create `tests/models/test_detectors.py`:

```python
from vlmbench.models.detectors.base_detector import target_class


def test_target_class_parses_the_prompt_template():
    assert target_class("Is there a person in this image? Answer yes or no.") == "person"
    assert target_class("Is there a traffic cone in this image? Answer yes or no.") == "traffic cone"


def test_fixed_vocab_answers_no_for_unknown_class():
    from vlmbench.models.detectors.ultralytics_detectors import FixedVocabYolo

    det = FixedVocabYolo.__new__(FixedVocabYolo)   # bypass model load
    det._names = {0: "person", 2: "car"}
    det._threshold = 0.25
    det._model = None  # not called for an unknown class
    # 'dog' is not in this model's vocabulary -> "no" without running detection.
    assert det.infer(object(), "Is there a dog in this image? Answer yes or no.") == "no"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/models/test_detectors.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Write the base helper**

Create `src/vlmbench/models/detectors/__init__.py` (empty) and `src/vlmbench/models/detectors/base_detector.py`:

```python
"""Shared helpers for detector adapters that answer the presence question."""
from __future__ import annotations

import re

_CLS_RE = re.compile(r"is there an? (.+?) in this image", re.IGNORECASE)


def target_class(prompt: str) -> str:
    """Extract the queried class from the fixed presence prompt template."""
    m = _CLS_RE.search(prompt)
    return (m.group(1) if m else prompt).strip().lower()
```

- [ ] **Step 4: Write the ultralytics adapters**

Create `src/vlmbench/models/detectors/ultralytics_detectors.py`:

```python
"""yolo11n, rt-detr (fixed COCO vocabulary) and yolo-world (open-vocab)."""
from __future__ import annotations

from typing import Any

from ..base import ModelMeta
from .base_detector import target_class

_THRESHOLD = 0.25


class FixedVocabYolo:
    """COCO-vocabulary detector (YOLO11n / RT-DETR). Answers 'no' for any class
    outside its trained vocabulary, which is a recorded capability limit."""

    def __init__(self, meta: ModelMeta, weights: str, loader) -> None:
        self._meta = meta
        self._weights = weights
        self._loader = loader     # callable(weights) -> ultralytics model
        self._model = None
        self._names: dict[int, str] = {}
        self._threshold = _THRESHOLD

    @property
    def meta(self) -> ModelMeta:
        return self._meta

    def load(self, backend: str, dtype: str) -> None:
        self._model = self._loader(self._weights)
        self._names = {i: n.lower() for i, n in self._model.names.items()}

    def infer(self, image: Any, prompt: str) -> str:
        cls = target_class(prompt)
        if cls not in self._names.values():
            return "no"
        res = self._model(image, verbose=False, conf=self._threshold)
        for r in res:
            for c in r.boxes.cls.tolist():
                if self._names.get(int(c)) == cls:
                    return "yes"
        return "no"


class OpenVocabWorld:
    """YOLO-World: set the queried class as the open-vocabulary text prompt."""

    def __init__(self, meta: ModelMeta, weights: str) -> None:
        self._meta = meta
        self._weights = weights
        self._model = None
        self._threshold = _THRESHOLD

    @property
    def meta(self) -> ModelMeta:
        return self._meta

    def load(self, backend: str, dtype: str) -> None:
        from ultralytics import YOLO
        self._model = YOLO(self._weights)

    def infer(self, image: Any, prompt: str) -> str:
        cls = target_class(prompt)
        self._model.set_classes([cls])
        res = self._model(image, verbose=False, conf=self._threshold)
        return "yes" if any(len(r.boxes) for r in res) else "no"


_AGPL = "AGPL-3.0"


def build_yolo11n() -> FixedVocabYolo:
    from ultralytics import YOLO
    meta = ModelMeta("yolo11n", 0.0026, _AGPL, "ultralytics/yolo11n",
                     ("fp32",))
    return FixedVocabYolo(meta, "yolo11n.pt", lambda w: YOLO(w))


def build_rtdetr() -> FixedVocabYolo:
    from ultralytics import RTDETR
    meta = ModelMeta("rt-detr", 0.032, _AGPL, "ultralytics/rtdetr-l",
                     ("fp32",))
    return FixedVocabYolo(meta, "rtdetr-l.pt", lambda w: RTDETR(w))


def build_yolo_world() -> OpenVocabWorld:
    meta = ModelMeta("yolo-world", 0.013, _AGPL,
                     "ultralytics/yolov8s-world", ("fp32",))
    return OpenVocabWorld(meta, "yolov8s-world.pt")
```

- [ ] **Step 5: Write the rfdetr adapter**

Create `src/vlmbench/models/detectors/rfdetr_nano.py`:

```python
"""RF-DETR-nano (fixed COCO vocabulary) via the rfdetr package."""
from __future__ import annotations

from typing import Any

from ..base import ModelMeta
from .base_detector import target_class

_THRESHOLD = 0.25


class RFDETRNanoAdapter:
    def __init__(self, meta: ModelMeta) -> None:
        self._meta = meta
        self._model = None
        self._names: set[str] = set()
        self._threshold = _THRESHOLD

    @property
    def meta(self) -> ModelMeta:
        return self._meta

    def load(self, backend: str, dtype: str) -> None:
        from rfdetr import RFDETRNano
        from rfdetr.util.coco_classes import COCO_CLASSES  # id->name map
        self._model = RFDETRNano()
        self._names = {n.lower() for n in COCO_CLASSES.values()}

    def infer(self, image: Any, prompt: str) -> str:
        cls = target_class(prompt)
        if cls not in self._names:
            return "no"
        det = self._model.predict(image, threshold=self._threshold)
        labels = {str(l).lower() for l in getattr(det, "class_names", [])}
        return "yes" if cls in labels else "no"


def build_rfdetr_nano() -> RFDETRNanoAdapter:
    meta = ModelMeta("rfdetr-nano", 0.0, "Apache-2.0",
                     "roboflow/rf-detr-nano", ("fp32",))
    return RFDETRNanoAdapter(meta)
```

_Note: the exact rfdetr predict/return API is confirmed in Task 1; adjust the `class_names` access to the observed shape._

- [ ] **Step 6: Register detectors + add the extra**

In `src/vlmbench/models/registry.py`, add the four builders to `_BUILDERS`:

```python
    "yolo11n": _lazy("ultralytics_detectors", "build_yolo11n"),
    "yolo-world": _lazy("ultralytics_detectors", "build_yolo_world"),
    "rt-detr": _lazy("ultralytics_detectors", "build_rtdetr"),
    "rfdetr-nano": _lazy("rfdetr_nano", "build_rfdetr_nano"),
```
using a small lazy helper so importing the registry does not require the detector deps:
```python
def _lazy(module: str, fn: str):
    def make():
        import importlib
        mod = importlib.import_module(f".detectors.{module}", __package__)
        return getattr(mod, fn)()
    return make
```
(Existing VLM builders stay as class references; `build_model` calls the value — for VLMs `Cls()`, for detectors `make()`. Ensure `build_model` calls the value: `return _BUILDERS[name]()`.)

In `pyproject.toml`, add:
```toml
detectors = ["ultralytics>=8.3", "rfdetr>=1.0"]
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/models/test_detectors.py -v`
Expected: PASS (2 tests).

- [ ] **Step 8: Full fast suite**

Run: `.venv/bin/python -m pytest -q -m 'not slow'`
Expected: all pass (registry import must not require detector deps).

- [ ] **Step 9: Commit**

```bash
git add src/vlmbench/models/detectors/ src/vlmbench/models/registry.py pyproject.toml tests/models/test_detectors.py
git commit -m "feat: detector adapters (yolo11n, yolo-world, rt-detr, rfdetr-nano) answering presence"
```

---

### Task 4: Configs and end-to-end smoke run

Validates the whole pipeline: one detector + one VLM through the orchestrator on a tiny presence run, producing an `ok` cell with an accuracy and latency for each.

**Files:**
- Create: `configs/presence_coco.yaml`, `configs/presence_openvocab.yaml`, `configs/presence_smoke.yaml`

**Interfaces:** none (configs + a manual run).

- [ ] **Step 1: Write the configs**

`configs/presence_coco.yaml`:
```yaml
models: [yolo11n, rt-detr, rfdetr-nano, yolo-world, smolvlm-256m, florence2-base, internvl2_5-2b, moondream2]
backends: [fp32]
tasks: [presence-coco]
warmup: 1
repeats: 3
subsample_n: 120
seed: 20260722
```
`configs/presence_openvocab.yaml`: same but `tasks: [presence-openvocab]`.
`configs/presence_smoke.yaml`:
```yaml
models: [yolo11n, smolvlm-256m]
backends: [fp32]
tasks: [presence-coco]
warmup: 1
repeats: 2
subsample_n: 4
seed: 20260722
```

- [ ] **Step 2: Run the smoke end-to-end**

Run: `.venv/bin/python scripts/run_benchmark.py --config configs/presence_smoke.yaml --out results_presence_smoke`
Expected: 2 cells written; both `status=ok` with a `metric_value` in [0,1] and an `infer_ms_mean`. Confirm `yolo11n` answers and is far faster than `smolvlm-256m`.

- [ ] **Step 3: Sanity-check the smoke output**

Run:
```bash
.venv/bin/python -c "import json; [print(json.loads(l)['model'], json.loads(l)['status'], json.loads(l)['metric_value'], round(json.loads(l)['infer_ms_mean'])) for l in open('results_presence_smoke/results.jsonl')]"
```
Expected: `yolo11n ok <score> <small ms>` and `smolvlm-256m ok <score> <larger ms>`.

- [ ] **Step 4: Commit (gitignore the smoke output)**

```bash
echo "results_presence_smoke/" >> .gitignore
git add configs/presence_coco.yaml configs/presence_openvocab.yaml configs/presence_smoke.yaml .gitignore
git commit -m "chore: presence benchmark configs + end-to-end smoke"
```

---

### Task 5: Full runs and figures/tables (execution)

Run the two full presence variants and generate per-task tables + figure. Long (Moondream2 dominates); background and resumable.

- [ ] **Step 1: Launch presence-coco (background)**

Run: `.venv/bin/python scripts/run_benchmark.py --config configs/presence_coco.yaml --out results_presence_coco`
(8 cells × N=120; detectors fast, Moondream2 the bottleneck.)

- [ ] **Step 2: Launch presence-openvocab (background, after or alongside)**

Run: `.venv/bin/python scripts/run_benchmark.py --config configs/presence_openvocab.yaml --out results_presence_openvocab`

- [ ] **Step 3: Patch energy for both**

Run `scripts/measure_energy.py` on each results file (as for the DocVQA/OCRBench runs).

- [ ] **Step 4: Generate per-task tables + figure**

Merge the two results files (or run `make_figures.py` per file) to produce `results_table_presence-coco.tex`, `results_table_presence-openvocab.tex`, and the presence trade-off figure into `paper_artifacts/`.

- [ ] **Step 5: Commit configs/results as appropriate (results gitignored).**

---

### Task 6: Paper integration (execution/writing)

Add the detector baseline to the paper: two presence tables, a findings paragraph, and tie the AGPL licensing point to the existing licensing discussion.

- [ ] **Step 1** Sync artifacts (`make sync`) and add two presence tables (COCO, open-vocab) mirroring the existing per-task table style.
- [ ] **Step 2** Write the findings prose: detectors dominate COCO efficiency (orders of magnitude cheaper on this machine); fixed-vocab detectors fail on open-vocab (answer "no"), where only YOLO-World and the VLMs compete; state the honest nuance (VLMs buy open-vocab generality at an efficiency cost).
- [ ] **Step 3** Extend the licensing subsection: detectors are AGPL-3.0, VLMs mostly Apache/MIT — a real adoption-cost axis.
- [ ] **Step 4** Update abstract/intro to note the detector baseline is now measured (this removes the review's #1 criticism); rebuild, verify 0 undefined refs.
- [ ] **Step 5** Commit + push both repos.

---

## Self-Review

**Spec coverage:** four detectors + VLMs on shared presence (Tasks 3–5); two variants coco/openvocab (Task 2); same-hardware measurement (Task 5 runs here); efficiency + accuracy via the harness (Tasks 4–5); gated feasibility spike (Task 1); two tables + prose + licensing (Task 6). ✓
**Placeholder scan:** dataset field names and the rfdetr return shape are explicitly deferred to the Task 1 spike and noted inline (genuine external unknowns the spike resolves), not vague TODOs. The open-vocab class set is concrete with a spike-confirmation note. ✓
**Type consistency:** `infer(image, prompt) -> str` unchanged across VLM and detector adapters; `target_class(prompt) -> str`; `build_presence(rows, classes, seed) -> list[Example]`; `load_presence(vocab, cap, seed) -> (examples, TaskSpec)`; registry values are all zero-arg callables invoked by `build_model[name]()`. ✓
