from __future__ import annotations

import os

from typing import Any

from .base import ModelMeta

# InternLM2's tokenizer.model contains a single degenerate piece whose string
# is a literal NUL character. SentencePiece 0.2.x refuses to load any model
# with a NUL in a piece ("piece must not include null character"), which
# transformers 4.x then swallows into a bool ``False`` return from
# ``AutoTokenizer.from_pretrained`` (it assumes the file is non-SPM and there
# is no fast/tiktoken fallback). The model crashes downstream inside its own
# ``chat()``. The proper NUL byte is already representable via the BYTE token
# ``<0x00>``, so we sanitize the offending piece and load from a patched copy.
_NUL = "\x00"
_NUL_REPLACEMENT = "␀"  # ␀ SYMBOL FOR NULL (printable, no NUL byte)


def _sanitize_proto(proto: Any) -> int:
    """Replace NUL characters in tokenizer pieces in place; return count fixed."""
    fixed = 0
    for piece in proto.pieces:
        if _NUL in piece.piece:
            piece.piece = piece.piece.replace(_NUL, _NUL_REPLACEMENT)
            fixed += 1
    return fixed


def _patched_tokenizer_dir(source: str) -> str:
    """Return a directory to load the tokenizer from.

    If the model's ``tokenizer.model`` has no NUL-containing piece, the
    original snapshot directory is returned unchanged. Otherwise a patched
    copy (sanitized ``tokenizer.model`` plus symlinks to the other files) is
    built once under the vlmbench cache and reused on subsequent runs.
    """
    from pathlib import Path

    from huggingface_hub import snapshot_download

    from sentencepiece import sentencepiece_model_pb2 as spm_pb2

    from .._paths import cache_dir

    try:
        snap = Path(snapshot_download(source))
    except Exception:  # offline: use whatever is already cached
        snap = Path(snapshot_download(source, local_files_only=True))

    model_file = snap / "tokenizer.model"
    if not model_file.exists():
        return str(snap)

    proto = spm_pb2.ModelProto()
    proto.ParseFromString(model_file.read_bytes())
    if not any(_NUL in p.piece for p in proto.pieces):
        return str(snap)  # SentencePiece can load this as-is

    out = cache_dir("internlm2_tokenizer_fix", snap.name)
    patched_model = out / "tokenizer.model"
    if not patched_model.exists():
        _sanitize_proto(proto)
        for name in os.listdir(snap):
            if name == "tokenizer.model":
                continue
            link = out / name
            if not link.exists():
                link.symlink_to(os.path.realpath(snap / name))
        patched_model.write_bytes(proto.SerializeToString())
    return str(out)

_META = ModelMeta(
    name="internvl2_5-2b",
    params_b=2.0,
    license="MIT (project); verify base-LLM license of the checkpoint",
    source="OpenGVLab/InternVL2_5-2B",
    supported_backends=("fp32",),
)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class InternVL2_5Adapter:
    def __init__(self, meta: ModelMeta = _META) -> None:
        self._meta = meta
        self._model = None
        self._tokenizer = None

    @property
    def meta(self) -> ModelMeta:
        return self._meta

    def load(self, backend: str, dtype: str) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._model = AutoModel.from_pretrained(
            self._meta.source,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(
            _patched_tokenizer_dir(self._meta.source),
            trust_remote_code=True, use_fast=False,
        )

    def _pixel_values(self, image: Any):
        import numpy as np
        import torch

        img = image.convert("RGB").resize((448, 448))
        arr = np.asarray(img).astype("float32") / 255.0  # H, W, C
        mean = np.array(_IMAGENET_MEAN, dtype="float32")
        std = np.array(_IMAGENET_STD, dtype="float32")
        arr = (arr - mean) / std
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1, C, H, W

    def infer(self, image: Any, prompt: str) -> str:
        pixel_values = self._pixel_values(image)
        question = f"<image>\n{prompt}"
        response = self._model.chat(
            self._tokenizer,
            pixel_values,
            question,
            generation_config=dict(max_new_tokens=64, do_sample=False),
        )
        return str(response)
