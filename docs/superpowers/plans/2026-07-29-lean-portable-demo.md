# Lean Portable Streamlit Demo — Implementation Plan (SmolVLM-only, refined)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A torch-free Streamlit demo that runs **SmolVLM-256M** (visual question answering) in INT8 ONNX on CPU: upload one image, ask a question, get the answer with latency and peak RAM. Memory-disciplined so it runs on modest hardware (Raspberry Pi 8 GB, Oracle Cloud Always-Free ARM 12 GB, HF PRO CPU Space 16 GB, or local).

**Architecture:** Reuse the existing `OnnxSmolVLM` backend through the SmolVLM adapter's `onnx-int8` path (verified torch-free: Task 1 spike ran it with no torch import). A thin `demo/lean_infer.py` wrapper loads the model once, resizes the input, runs one inference, and releases memory back to the OS. `demo/streamlit_app.py` is a thin single-image UI over it.

**Tech Stack:** Python, onnxruntime (inference-only — no autograd, no torch), transformers processor (numpy), Pillow, numpy, streamlit; the `vlmbench` package (base install, no `[models]`/torch extra).

## Global Constraints

- **Torch-free:** no file under `demo/` may import `torch`; `demo/requirements-streamlit.txt` must not list torch (and must not list `timm`/`torchvision`, which pull torch). The onnx-int8 SmolVLM path is torch-free; keep it that way. (ONNX Runtime is inference-only, so there is no gradient state to disable — "no grad" is inherent.)
- **One model:** SmolVLM-256M only, `backend="onnx-int8"`. Florence-2 is out of scope here (its trust_remote_code processor hard-requires torch).
- **Memory discipline (the point of this demo):**
  - Load the model **once** and cache it (`@st.cache_resource`) — never per interaction.
  - **Resize** each uploaded image to at most 512×512 before inference (bounds the input; note it does not lower the ~2.4 GB peak, which is set by SmolVLM's fixed 17-sub-image vision encoding).
  - After each inference, **release**: drop references and call `gc.collect()` then glibc `malloc_trim(0)` to return freed memory to the OS. Keeps steady-state RSS low between runs.
  - **No history:** one image at a time, no accumulation of past images/answers.
- **Honest footprint:** peak RSS per inference is ~2.4 GB (measured, torch-free). Fits Pi 8 GB / Oracle A1 12 GB / HF PRO 16 GB / local; NOT the ~1 GB Streamlit Community Cloud or 512 MB Render free tiers.
- **Reuse, don't reimplement** inference: `build_model("smolvlm-256m").load(backend="onnx-int8", ...)` + `.infer(image, prompt)`; peak RSS via `vlmbench.profiling.memory.sample_peak_rss_mb`.
- Run all Python via `.venv/bin/python` / `.venv/bin/python -m pytest`.

## Status of Task 1 (already done)

Task 1 (feasibility spike, `demo/check_lean.py`, commit `9012c91`) is COMPLETE and confirmed SmolVLM runs torch-free on the onnx-int8 path at ~2.4 GB peak (with `enable_cpu_mem_arena=False`, added to `OnnxSmolVLM`). Do not redo it. Florence was found blocked and dropped from scope.

---

### Task 2: Lean inference wrapper (`demo/lean_infer.py`)

The small, testable surface the UI calls: load SmolVLM once on the torch-free onnx-int8 path, resize the image, run one inference, release memory.

**Files:**
- Create: `demo/lean_infer.py`
- Create: `demo/__init__.py` (empty), `tests/demo/__init__.py` (empty)
- Test: `tests/demo/test_lean_infer.py`

**Interfaces:**
- Consumes: `vlmbench.models.registry.build_model`, `vlmbench.profiling.memory.sample_peak_rss_mb`.
- Produces:
  - `MODEL_KEY: str = "smolvlm-256m"`, `MAX_SIDE: int = 512`.
  - `resize_max_side(image, max_side=MAX_SIDE) -> PIL.Image.Image` — returns the image scaled so its longest side is ≤ `max_side` (no upscaling), preserving aspect ratio.
  - `release_memory() -> None` — `gc.collect()` then best-effort glibc `malloc_trim(0)` (must not raise on non-glibc platforms).
  - `LeanVLM()` with `.infer(image, prompt: str) -> tuple[str, float, float]` returning `(answer, latency_ms, peak_rss_mb)`; resizes the image, runs one inference, releases memory afterwards.

- [ ] **Step 1: Write the failing tests**

Create `demo/__init__.py` and `tests/demo/__init__.py` (both empty). Then `tests/demo/test_lean_infer.py`:

```python
from PIL import Image

from demo.lean_infer import MAX_SIDE, MODEL_KEY, release_memory, resize_max_side


def test_model_key_is_smolvlm():
    assert MODEL_KEY == "smolvlm-256m"


def test_resize_scales_long_side_down_preserving_aspect():
    img = Image.new("RGB", (2048, 1024))
    out = resize_max_side(img, max_side=512)
    assert max(out.size) == 512
    assert out.size == (512, 256)  # aspect preserved


def test_resize_does_not_upscale_small_images():
    img = Image.new("RGB", (300, 200))
    out = resize_max_side(img, max_side=512)
    assert out.size == (300, 200)


def test_release_memory_is_safe_to_call():
    release_memory()  # must not raise on any platform
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/demo/test_lean_infer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'demo.lean_infer'`.

- [ ] **Step 3: Write the implementation**

Create `demo/lean_infer.py`:

```python
"""Torch-free, memory-disciplined inference for the lean SmolVLM demo.

Runs SmolVLM-256M through the standard adapter's ``onnx-int8`` path (ONNX
Runtime, no torch, no autograd). Loads the model once; the caller resizes the
image and releases memory after each inference.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import gc
import time

from vlmbench.models.registry import build_model
from vlmbench.profiling.memory import sample_peak_rss_mb

MODEL_KEY = "smolvlm-256m"
MAX_SIDE = 512


def resize_max_side(image, max_side: int = MAX_SIDE):
    """Scale ``image`` down so its longest side is <= ``max_side`` (never up)."""
    w, h = image.size
    longest = max(w, h)
    if longest <= max_side:
        return image
    scale = max_side / longest
    return image.resize((round(w * scale), round(h * scale)))


def release_memory() -> None:
    """Return freed heap back to the OS (glibc); no-op elsewhere."""
    gc.collect()
    libc_name = ctypes.util.find_library("c")
    if not libc_name:
        return
    try:
        libc = ctypes.CDLL(libc_name)
        if hasattr(libc, "malloc_trim"):
            libc.malloc_trim(0)
    except OSError:
        pass


class LeanVLM:
    """Loads SmolVLM-256M once on the torch-free onnx-int8 path."""

    def __init__(self) -> None:
        self._model = build_model(MODEL_KEY)
        self._model.load(backend="onnx-int8", dtype="int8")

    def infer(self, image, prompt: str) -> tuple[str, float, float]:
        image = resize_max_side(image)
        start = time.perf_counter()
        answer, peak_mb = sample_peak_rss_mb(
            lambda: self._model.infer(image, prompt))
        latency_ms = (time.perf_counter() - start) * 1000.0
        release_memory()
        return str(answer).strip(), latency_ms, peak_mb
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/demo/test_lean_infer.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Run the fast suite**

Run: `.venv/bin/python -m pytest -q -m 'not slow'`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add demo/__init__.py demo/lean_infer.py tests/demo/__init__.py tests/demo/test_lean_infer.py
git commit -m "demo: lean torch-free SmolVLM wrapper (resize, load-once, release memory)"
```

---

### Task 3: Streamlit app and lean requirements

The single-image UI plus the torch-free deploy stack. A smoke test guards that no demo file imports torch.

**Files:**
- Create: `demo/streamlit_app.py`, `demo/requirements-streamlit.txt`
- Test: `tests/demo/test_no_torch.py`

**Interfaces:**
- Consumes: `demo.lean_infer.LeanVLM`.

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


def test_streamlit_requirements_exclude_torch_stack():
    req = (_DEMO / "requirements-streamlit.txt").read_text().lower()
    for banned in ("torch", "timm", "torchvision"):
        assert banned not in req, f"{banned} must not be in the lean requirements"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/demo/test_no_torch.py -v`
Expected: FAIL — `requirements-streamlit.txt` does not exist (FileNotFoundError).

- [ ] **Step 3: Write the lean requirements**

Create `demo/requirements-streamlit.txt`:

```text
# Lean, torch-free stack for the portable SmolVLM demo (INT8 ONNX, CPU).
# Do NOT add torch / timm / torchvision — they pull PyTorch and blow the footprint.
onnxruntime>=1.18
transformers>=4.44,<5
tokenizers>=0.19
sentencepiece>=0.2
protobuf>=4.0
huggingface-hub>=0.23
pillow>=10.0
numpy>=1.24
streamlit>=1.30
# The vlmbench package (base install, WITHOUT the [models]/torch extra).
.
```

- [ ] **Step 4: Write the Streamlit app**

Create `demo/streamlit_app.py`:

```python
"""Lean, portable SmolVLM-256M demo: ask a question about one image, on CPU,
INT8 ONNX, no PyTorch. Run: streamlit run demo/streamlit_app.py

One image at a time, no history. The model loads once and is cached; memory
is released after each analysis.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from lean_infer import LeanVLM  # noqa: E402


@st.cache_resource(show_spinner="Loading SmolVLM (first run downloads ~250 MB)…")
def _model() -> LeanVLM:
    return LeanVLM()


st.set_page_config(page_title="SmolVLM on CPU", page_icon="\U0001f9e9")
st.title("SmolVLM-256M on CPU — lean & portable")
st.caption("Ask a question about an image. Runs in INT8 ONNX — no GPU, no "
           "PyTorch. Small enough for a Raspberry Pi.")

uploaded = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp"],
                            accept_multiple_files=False)
prompt = st.text_input("Question", "What is in this image?")

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, use_container_width=True)
    if st.button("Analyze", type="primary"):
        with st.spinner("Running on CPU…"):
            answer, latency_ms, peak_mb = _model().infer(image, prompt)
        st.subheader("Answer")
        st.write(answer or "_(empty)_")
        c1, c2 = st.columns(2)
        c1.metric("Latency", f"{latency_ms / 1000:.1f} s")
        c2.metric("Peak RAM", f"{peak_mb / 1024:.2f} GB")
        st.caption("Peak RAM is the whole-process resident set. No PyTorch is "
                   "loaded; memory is released after each analysis.")
```

- [ ] **Step 5: Run the guard test to verify it passes**

Run: `.venv/bin/python -m pytest tests/demo/test_no_torch.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Verify the app parses**

Run: `.venv/bin/python -c "import ast; ast.parse(open('demo/streamlit_app.py').read()); print('parse ok')"`
Expected: `parse ok`

- [ ] **Step 7: Commit**

```bash
git add demo/streamlit_app.py demo/requirements-streamlit.txt tests/demo/test_no_torch.py
git commit -m "demo: single-image Streamlit app + lean torch-free requirements"
```

---

### Task 4: README and deploy documentation

Document running and deploying the lean demo, with the honest footprint and the free/paid host options.

**Files:**
- Modify: `demo/README.md`

- [ ] **Step 1: Append the lean-demo section**

Add to `demo/README.md` (keep the existing Gradio section; add below it). Use the measured peak from Task 1 (~2.4 GB) — if a more exact number is on record, use it:

```markdown
## Lean, portable Streamlit demo (SmolVLM, torch-free)

A single-image demo of **SmolVLM-256M** running in **INT8 ONNX with no
PyTorch**. Upload one image, ask a question, get the answer with latency and
peak RAM. One image at a time, no history; the model loads once and memory is
released after each analysis.

### Run locally

```bash
python -m pip install -r demo/requirements-streamlit.txt   # no torch
streamlit run demo/streamlit_app.py
```

First run downloads ~0.25 GB of INT8 ONNX weights from the Hub.

### Measured footprint

Peak resident memory per inference on this machine (torch-free onnx-int8):
**~2.4 GB** — versus 6–12 GB through the full PyTorch stack. The weights are
~250 MB; the rest is the vision encoder's activation memory (SmolVLM splits
each image into 17 sub-images). Resizing the input bounds it but does not
lower this peak; memory is returned to the OS (`malloc_trim`) after each run.

### Where it runs (fits ~2.4 GB)

- **Local / Raspberry Pi 4/5 (8 GB):** the recommended target — ARM builds of
  onnxruntime work; slower than x86 but fits.
- **Oracle Cloud Always-Free (Ampere A1, 12 GB ARM):** a free public host that
  fits comfortably and reinforces the ARM-portability story (VM setup + Oracle
  A1 capacity permitting).
- **Hugging Face Spaces (CPU Basic, 16 GB):** fits, but creating a compute
  Space now requires a PRO plan.
- **Too small (do not use):** Streamlit Community Cloud (~1 GB) and Render free
  (512 MB).
- **Not included:** Moondream2 / InternVL2.5-2B (no INT8 artifact, ~4 GB
  weights) and Florence-2 (its remote processor hard-requires torch).
```

- [ ] **Step 2: Commit**

```bash
git add demo/README.md
git commit -m "docs: document the lean SmolVLM Streamlit demo, footprint, and hosts"
```

---

## Self-Review

**Spec coverage:** SmolVLM-only torch-free onnx path (Task 1 done + Task 2); resize/load-once/release/no-history (Task 2 + Task 3 constraints); single-image UI (Task 3); requirements + no-torch guard (Task 3); honest footprint + hosts (Task 4). ✓
**Placeholder scan:** none (the ~2.4 GB number is the measured Task 1 value). ✓
**Type consistency:** `LeanVLM.infer -> (str, float, float)`; `resize_max_side`/`release_memory` signatures match tests. ✓
