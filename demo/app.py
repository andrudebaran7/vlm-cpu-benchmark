"""Gradio demo: pick a model, upload an image, see the answer + latency/RAM."""
from __future__ import annotations

import time

import gradio as gr

from vlmbench.models.registry import build_model
from vlmbench.profiling.memory import sample_peak_rss_mb

_MODELS = ["smolvlm-256m", "moondream2", "florence2-base"]
_CACHE: dict[str, object] = {}


def _get(name):
    if name not in _CACHE:
        model = build_model(name)
        model.load(backend="fp32", dtype="float32")
        _CACHE[name] = model
    return _CACHE[name]


def run(model_name, image, prompt):
    model = _get(model_name)
    start = time.perf_counter()
    answer, peak = sample_peak_rss_mb(lambda: model.infer(image, prompt))
    elapsed_ms = (time.perf_counter() - start) * 1000
    return answer, f"{elapsed_ms:.0f} ms", f"{peak:.0f} MB"


with gr.Blocks() as demo:
    gr.Markdown("# Small VLMs on CPU — live efficiency demo")
    with gr.Row():
        model_dd = gr.Dropdown(_MODELS, value=_MODELS[0], label="model")
        prompt_tb = gr.Textbox("Describe the image.", label="prompt")
    image_in = gr.Image(type="pil", label="image")
    run_btn = gr.Button("Run")
    answer_out = gr.Textbox(label="answer")
    latency_out = gr.Textbox(label="latency")
    ram_out = gr.Textbox(label="peak RAM")
    run_btn.click(run, [model_dd, image_in, prompt_tb],
                  [answer_out, latency_out, ram_out])

if __name__ == "__main__":
    demo.launch()
