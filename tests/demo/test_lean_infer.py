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


def test_infer_releases_memory_even_when_inference_raises(monkeypatch):
    import demo.lean_infer as li

    called = {"released": False}
    monkeypatch.setattr(li, "release_memory", lambda: called.__setitem__("released", True))

    class _Boom:
        def infer(self, image, prompt):
            raise RuntimeError("boom")

    obj = li.LeanVLM.__new__(li.LeanVLM)
    obj._model = _Boom()
    with __import__("pytest").raises(RuntimeError):
        obj.infer(Image.new("RGB", (10, 10)), "q")
    assert called["released"] is True
