import pytest
from vlmbench.backends.registry import (KNOWN_BACKENDS, backend_supports,
                                        validate_backend)
from vlmbench.models.base import ModelMeta


def test_supports_reflects_meta():
    meta = ModelMeta(name="m", params_b=0.5, license="MIT", source="x",
                     supported_backends=("fp32", "onnx-int8"))
    assert backend_supports(meta, "fp32") is True
    assert backend_supports(meta, "gguf-q4") is False


def test_validate_backend_rejects_unknown():
    assert "fp32" in KNOWN_BACKENDS
    with pytest.raises(ValueError):
        validate_backend("magic")
    validate_backend("onnx-int8")  # no raise
