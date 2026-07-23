import pytest
from PIL import Image

from vlmbench.models.registry import build_model

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("name", ["smolvlm-256m", "florence2-base"])
def test_adapter_load_and_infer_smoke(name):
    model = build_model(name)
    model.load(backend="fp32", dtype="float32")
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    out = model.infer(image=img, prompt="What color is the image?")
    assert isinstance(out, str) and len(out) >= 0
