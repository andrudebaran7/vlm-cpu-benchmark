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
