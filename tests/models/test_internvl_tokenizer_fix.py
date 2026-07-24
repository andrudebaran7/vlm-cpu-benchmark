"""Unit tests for the InternLM2 tokenizer NUL-piece sanitization."""
from __future__ import annotations

import pytest

from vlmbench.models.internvl2_5 import _NUL_REPLACEMENT, _sanitize_proto

spm_pb2 = pytest.importorskip("sentencepiece.sentencepiece_model_pb2")


def _proto_with_pieces(strings):
    proto = spm_pb2.ModelProto()
    for s in strings:
        p = proto.pieces.add()
        p.piece = s
        p.score = 0.0
    return proto


def test_sanitize_replaces_nul_and_reports_count():
    proto = _proto_with_pieces(["hello", "\x00", "world"])
    fixed = _sanitize_proto(proto)
    assert fixed == 1
    assert [p.piece for p in proto.pieces] == ["hello", _NUL_REPLACEMENT, "world"]
    # A sanitized proto must serialize (the whole point: SentencePiece 0.2.x
    # rejects NUL, but the bytes themselves round-trip fine here).
    assert proto.SerializeToString()


def test_sanitize_is_noop_without_nul():
    proto = _proto_with_pieces(["a", "\x01", "b"])  # 0x01 is allowed
    assert _sanitize_proto(proto) == 0
    assert [p.piece for p in proto.pieces] == ["a", "\x01", "b"]
