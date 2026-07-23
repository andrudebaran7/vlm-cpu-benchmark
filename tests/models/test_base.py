from vlmbench.models.base import ModelMeta
from vlmbench.models.fake import FakeVLModel


def test_fake_model_loads_and_infers():
    meta = ModelMeta(name="fake-tiny", params_b=0.1, license="Apache-2.0",
                     source="fake/tiny", supported_backends=("fp32", "onnx-int8"))
    model = FakeVLModel(meta=meta, responder=lambda image, prompt: f"echo:{prompt}")
    model.load(backend="fp32", dtype="float32")
    assert model.meta.name == "fake-tiny"
    assert model.loaded_backend == "fp32"
    assert model.infer(image=None, prompt="hi") == "echo:hi"
