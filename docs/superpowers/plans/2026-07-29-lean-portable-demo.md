# Lean Portable Streamlit Demo — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A torch-free Streamlit demo that runs SmolVLM-256M (VQA) and Florence-2-base (OCR) in INT8 ONNX on CPU, showing the answer with latency and peak RAM, small enough for constrained hosts (Raspberry Pi; Streamlit Community Cloud borderline).

**Architecture:** Reuse the existing `OnnxSmolVLM` / `OnnxFlorence2` backends through the standard adapters' `onnx-int8` path (which imports no torch). A thin `demo/lean_infer.py` wrapper exposes `LeanVLM.infer(image, prompt) -> (answer, latency_ms, peak_rss_mb)`; `demo/streamlit_app.py` is a thin view over it. A torch-free `requirements-streamlit.txt` pins the deploy stack.

**Tech Stack:** Python, onnxruntime, transformers (no torch), Pillow, numpy, streamlit, the `vlmbench` package (base install, no `[models]` extra).

## Global Constraints

- **Torch-free:** no file under `demo/` may import `torch`, and `demo/requirements-streamlit.txt` must not list torch. The onnx-int8 adapter path never imports torch; keep it that way.
- **INT8 ONNX only**, exactly two models: `smolvlm-256m` and `florence2-base` (`backend="onnx-int8"`).
- **Reuse, don't reimplement:** inference goes through `build_model(key).load(backend="onnx-int8", ...)` + `.infer(image, prompt)`; peak RSS via `vlmbench.profiling.memory.sample_peak_rss_mb`.
- **Honest footprint:** report the *measured* peak RSS; state the deploy target from that number (Pi/local safe; free Streamlit tier is ~690 MB–2.7 GB, borderline).
- Run all Python via the repo venv: `.venv/bin/python` and `.venv/bin/pytest` (or `.venv/bin/python -m pytest`).

---

### Task 1: Feasibility spike — verify torch-free inference and measure RSS

De-risks the whole design: confirms the onnx-int8 adapter path runs both models without importing torch, and records the real peak RSS. Produces a keepable smoke/portability check.

**Files:**
- Create: `demo/check_lean.py`

**Interfaces:**
- Consumes: `vlmbench.models.registry.build_model`, `vlmbench.profiling.memory.sample_peak_rss_mb`.
- Produces: nothing importable; a runnable check that prints per-model `answer`, `peak_rss_mb`, and a torch-free assertion.

- [ ] **Step 1: Write the check script**

Create `demo/check_lean.py`:

```python
"""Portability spike: run the two demo models on the torch-free onnx-int8
path, assert no torch import, and print peak RSS. Run manually:

    .venv/bin/python demo/check_lean.py
"""
from __future__ import annotations

import sys

from PIL import Image

from vlmbench.models.registry import build_model
from vlmbench.profiling.memory import sample_peak_rss_mb

_CASES = [
    ("smolvlm-256m", "What is in this image?"),
    ("florence2-base", ""),
]


def main() -> None:
    img = Image.new("RGB", (512, 512), (120, 120, 120))
    for key, prompt in _CASES:
        model = build_model(key)
        model.load(backend="onnx-int8", dtype="int8")
        answer, peak_mb = sample_peak_rss_mb(lambda: model.infer(img, prompt))
        torch_loaded = "torch" in sys.modules
        print(f"{key:16} peak_rss={peak_mb:.0f} MB  torch_imported={torch_loaded}  "
              f"answer={answer!r:.60}")
        assert not torch_loaded, f"{key}: torch was imported on the onnx path!"
    print("OK: both models ran torch-free.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the spike**

Run: `.venv/bin/python demo/check_lean.py`
Expected: both lines print an `answer`, `torch_imported=False`, and a `peak_rss` number; final line `OK: both models ran torch-free.`

Record the two `peak_rss` numbers — they set the honest deploy target in Task 4.

- [ ] **Step 3: Decision gate**

- If `torch_imported=True` for a model, its `AutoProcessor` pulls torch: note it; Task 2 must add a manual PIL+numpy preprocessing path for that model (fallback from the spec). Do not proceed to Task 3 for that model until torch-free.
- If both are `False`, proceed as planned.

- [ ] **Step 4: Commit**

```bash
git add demo/check_lean.py
git commit -m "demo: add torch-free portability spike for the lean demo"
```

---

### Task 2: Lean inference wrapper (`demo/lean_infer.py`)

The small, testable surface the UI calls. Holds the demo's model list and a `LeanVLM` that loads one model on the onnx-int8 path and times/measures one inference.

**Files:**
- Create: `demo/lean_infer.py`
- Create: `demo/__init__.py` (empty; makes `demo` importable as a package for tests)
- Test: `tests/demo/__init__.py` (empty), `tests/demo/test_lean_infer.py`

**Interfaces:**
- Consumes: `build_model`, `sample_peak_rss_mb` (as in Task 1).
- Produces:
  - `DemoModel(key: str, label: str, needs_prompt: bool, default_prompt: str)` (frozen dataclass).
  - `DEMO_MODELS: tuple[DemoModel, ...]`.
  - `demo_model(key: str) -> DemoModel` (raises `KeyError` on unknown key).
  - `LeanVLM(key: str)` with `.infer(image, prompt: str) -> tuple[str, float, float]` returning `(answer, latency_ms, peak_rss_mb)`.

- [ ] **Step 1: Write the failing test**

Create `tests/demo/__init__.py` (empty) and `demo/__init__.py` (empty). Then create `tests/demo/test_lean_infer.py`:

```python
import pytest

from demo.lean_infer import DEMO_MODELS, demo_model


def test_demo_models_are_the_two_lean_int8_models():
    assert [m.key for m in DEMO_MODELS] == ["smolvlm-256m", "florence2-base"]


def test_smolvlm_needs_a_prompt_florence_does_not():
    assert demo_model("smolvlm-256m").needs_prompt is True
    assert demo_model("florence2-base").needs_prompt is False
    assert demo_model("smolvlm-256m").default_prompt  # non-empty


def test_demo_model_unknown_key_raises():
    with pytest.raises(KeyError):
        demo_model("moondream2")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/demo/test_lean_infer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'demo.lean_infer'` (or import error).

- [ ] **Step 3: Write the implementation**

Create `demo/lean_infer.py`:

```python
"""Torch-free inference wrapper for the lean/portable demo.

Runs the two small models through the standard adapters' ``onnx-int8`` path,
which imports no torch (verified by demo/check_lean.py). Kept tiny so the
Streamlit app is a thin view over it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from vlmbench.models.registry import build_model
from vlmbench.profiling.memory import sample_peak_rss_mb


@dataclass(frozen=True)
class DemoModel:
    key: str
    label: str
    needs_prompt: bool
    default_prompt: str


DEMO_MODELS: tuple[DemoModel, ...] = (
    DemoModel("smolvlm-256m", "SmolVLM-256M — ask a question about the image",
              True, "What is in this image?"),
    DemoModel("florence2-base", "Florence-2 — read the text (OCR)",
              False, ""),
)


def demo_model(key: str) -> DemoModel:
    for m in DEMO_MODELS:
        if m.key == key:
            return m
    raise KeyError(key)


class LeanVLM:
    """Loads one model on the torch-free onnx-int8 path and runs inference."""

    def __init__(self, key: str) -> None:
        demo_model(key)  # validate
        self._model = build_model(key)
        self._model.load(backend="onnx-int8", dtype="int8")

    def infer(self, image, prompt: str) -> tuple[str, float, float]:
        start = time.perf_counter()
        answer, peak_mb = sample_peak_rss_mb(
            lambda: self._model.infer(image, prompt))
        latency_ms = (time.perf_counter() - start) * 1000.0
        return str(answer).strip(), latency_ms, peak_mb
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/demo/test_lean_infer.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Verify the whole fast suite still passes**

Run: `.venv/bin/python -m pytest -q -m 'not slow'`
Expected: all pass (previous count + 3).

- [ ] **Step 6: Commit**

```bash
git add demo/__init__.py demo/lean_infer.py tests/demo/__init__.py tests/demo/test_lean_infer.py
git commit -m "demo: lean torch-free inference wrapper (LeanVLM)"
```

---

### Task 3: Streamlit app and lean requirements

The UI plus the torch-free deploy stack. The app is a thin view over `LeanVLM`; a smoke test guards that no demo file imports torch.

**Files:**
- Create: `demo/streamlit_app.py`
- Create: `demo/requirements-streamlit.txt`
- Test: `tests/demo/test_no_torch.py`

**Interfaces:**
- Consumes: `demo.lean_infer.DEMO_MODELS`, `demo.lean_infer.demo_model`, `demo.lean_infer.LeanVLM`.
- Produces: a runnable Streamlit entrypoint; no importable API.

- [ ] **Step 1: Write the failing guard test**

Create `tests/demo/test_no_torch.py`:

```python
import ast
import pathlib

_DEMO = pathlib.Path(__file__).resolve().parents[2] / "demo"


def _imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_no_demo_file_imports_torch():
    offenders = {p.name for p in _DEMO.glob("*.py") if "torch" in _imports(p)}
    assert offenders == set(), f"demo files import torch: {offenders}"


def test_streamlit_requirements_have_no_torch():
    req = (_DEMO / "requirements-streamlit.txt").read_text().lower()
    assert "torch" not in req
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/demo/test_no_torch.py -v`
Expected: FAIL — `requirements-streamlit.txt` does not exist yet (FileNotFoundError) / or streamlit_app not present.

- [ ] **Step 3: Write the lean requirements**

Create `demo/requirements-streamlit.txt`:

```text
# Lean, torch-free stack for the portable Streamlit demo.
# INT8 ONNX inference only; no PyTorch, no CUDA.
onnxruntime>=1.18
transformers>=4.44,<5
tokenizers>=0.19
sentencepiece>=0.2
protobuf>=4.0
huggingface-hub>=0.23
pillow>=10.0
numpy>=1.24
streamlit>=1.30
# The vlmbench package itself (base install, WITHOUT the [models]/torch extra).
# When deploying a mirror/checkout, this installs src/vlmbench from the repo root.
.
```

- [ ] **Step 4: Write the Streamlit app**

Create `demo/streamlit_app.py`:

```python
"""Lean, portable VLM demo: SmolVLM-256M (VQA) and Florence-2 (OCR) on CPU,
INT8 ONNX, no PyTorch. Run: streamlit run demo/streamlit_app.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from lean_infer import DEMO_MODELS, LeanVLM, demo_model  # noqa: E402


@st.cache_resource(show_spinner="Loading model (first run downloads ~250 MB)…")
def _load(key: str) -> LeanVLM:
    return LeanVLM(key)


st.set_page_config(page_title="Small VLMs on CPU", page_icon="\U0001f9e9")
st.title("Small VLMs on CPU — lean & portable")
st.caption("SmolVLM-256M and Florence-2 run here in INT8 ONNX — no GPU, no "
           "PyTorch. Small enough for a Raspberry Pi.")

label_to_key = {m.label: m.key for m in DEMO_MODELS}
label = st.radio("Model", list(label_to_key))
key = label_to_key[label]
model = demo_model(key)

uploaded = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp"])
prompt = st.text_input("Question", model.default_prompt) if model.needs_prompt else ""

if uploaded is not None and st.button("Run", type="primary"):
    image = Image.open(uploaded).convert("RGB")
    st.image(image, use_container_width=True)
    with st.spinner("Running on CPU…"):
        answer, latency_ms, peak_mb = _load(key).infer(image, prompt)
    st.subheader("Answer")
    st.write(answer or "_(empty)_")
    c1, c2 = st.columns(2)
    c1.metric("Latency", f"{latency_ms / 1000:.1f} s")
    c2.metric("Peak RAM", f"{peak_mb / 1024:.2f} GB")
    st.caption("Peak RAM is the whole-process resident set — the portability "
               "number. No PyTorch is loaded.")
```

- [ ] **Step 5: Run the guard test to verify it passes**

Run: `.venv/bin/python -m pytest tests/demo/test_no_torch.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Verify the app parses and its imports resolve (no execution)**

Run: `.venv/bin/python -c "import ast; ast.parse(open('demo/streamlit_app.py').read()); print('parse ok')"`
Expected: `parse ok`

- [ ] **Step 7: Commit**

```bash
git add demo/streamlit_app.py demo/requirements-streamlit.txt tests/demo/test_no_torch.py
git commit -m "demo: Streamlit app + lean torch-free requirements"
```

---

### Task 4: README and deploy documentation

Document how to run the lean demo, deploy it, and the measured footprint (from Task 1). Update the existing `demo/README.md` (currently Gradio-only).

**Files:**
- Modify: `demo/README.md`

**Interfaces:** none (docs).

- [ ] **Step 1: Append the lean-demo section to the README**

Add to `demo/README.md` (keep the existing Gradio section; add below it). Fill `<SMOLVLM_RSS>` / `<FLORENCE_RSS>` with the numbers recorded in Task 1, Step 2:

```markdown
## Lean, portable Streamlit demo (torch-free)

A second demo runs the two smallest models in **INT8 ONNX with no PyTorch**,
so it fits constrained hardware. It exposes SmolVLM-256M (ask a question) and
Florence-2-base (read the text / OCR), showing the answer with latency and
peak RAM.

### Run locally

```bash
python -m pip install -r demo/requirements-streamlit.txt   # no torch
streamlit run demo/streamlit_app.py
```

First run downloads ~0.25 GB of INT8 ONNX weights per model from the Hub.

### Measured footprint

Peak resident memory on this machine (torch-free onnx-int8 path):
SmolVLM-256M ≈ `<SMOLVLM_RSS>` MB, Florence-2 ≈ `<FLORENCE_RSS>` MB —
versus 6–12 GB through the full PyTorch stack in the benchmark. The weights
are ~250 MB each; the rest is runtime.

### Where it runs

- **Local / Raspberry Pi 4/5 (8 GB):** comfortably — the recommended target
  (ARM builds of onnxruntime work; inference is slower than x86 but fits).
- **Streamlit Community Cloud:** the free tier allocates a *variable* ~690 MB–
  2.7 GB per app. It fits when the measured footprint is under ~700–900 MB and
  the app has headroom; under contention it may be throttled. Point the app at
  `demo/streamlit_app.py` with `demo/requirements-streamlit.txt`.
- **Not included:** Moondream2 and InternVL2.5-2B — no INT8 artifact and
  ~4 GB of weights, so they stay in the CLI benchmark, not the lean demo.
```

- [ ] **Step 2: Commit**

```bash
git add demo/README.md
git commit -m "docs: document the lean/portable Streamlit demo and its footprint"
```

---

## Self-Review

**Spec coverage:**
- Torch-free onnx path + measure RSS → Task 1 (spike) + Task 2 (wrapper). ✓
- Two models selectable (SmolVLM VQA, Florence OCR) → `DEMO_MODELS` (Task 2), UI radio (Task 3). ✓
- `demo/lean_infer.py`, `demo/streamlit_app.py`, `demo/requirements-streamlit.txt`, README → Tasks 2–4. ✓
- Out of scope (moondream/internvl, GPU, benchmark UI) → excluded; noted in README. ✓
- Honest deploy target from measured number → Task 1 records it, Task 4 states it. ✓
- Feasibility spike gated before UI → Task 1 with a decision gate before Task 3. ✓
- Manual-preproc fallback → called out in Task 1 Step 3 gate (only if a processor pulls torch).

**Placeholder scan:** `<SMOLVLM_RSS>` / `<FLORENCE_RSS>` are the one intentional fill-in, sourced from a concrete measurement step (Task 1 Step 2) — acceptable.

**Type consistency:** `LeanVLM.infer -> (str, float, float)` used consistently; `demo_model` raises `KeyError`; `DEMO_MODELS` keys match the two backends declared onnx-int8-capable (`smolvlm-256m`, `florence2-base`).
