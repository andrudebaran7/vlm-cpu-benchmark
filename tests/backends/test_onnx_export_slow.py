import pytest
from PIL import Image

from vlmbench.models.smolvlm import SmolVLMAdapter

pytestmark = pytest.mark.slow


def test_smolvlm_onnx_int8_infer():
    model = SmolVLMAdapter()
    model.load(backend="onnx-int8", dtype="int8")
    img = Image.new("RGB", (64, 64), color=(10, 200, 10))
    out = model.infer(image=img, prompt="Describe the image.")
    assert isinstance(out, str)
