import pytest

from vlmbench.models.registry import build_model


@pytest.mark.parametrize("name,source", [
    ("smolvlm-256m", "HuggingFaceTB/SmolVLM-256M-Instruct"),
    ("moondream2", "vikhyatk/moondream2"),
    ("florence2-base", "microsoft/Florence-2-base"),
    ("internvl2_5-2b", "OpenGVLab/InternVL2_5-2B"),
])
def test_build_model_returns_adapter_with_meta(name, source):
    model = build_model(name)
    assert model.meta.name == name
    assert model.meta.source == source
    assert isinstance(model.meta.supported_backends, tuple)


def test_build_model_rejects_unknown():
    with pytest.raises(ValueError):
        build_model("nope")
