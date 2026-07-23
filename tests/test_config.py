import pytest
from vlmbench.config import BenchConfig, load_config


def test_load_valid_config(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "models: [smolvlm-256m]\n"
        "backends: [fp32, onnx-int8]\n"
        "tasks: [sample]\n"
        "warmup: 1\nrepeats: 3\nsubsample_n: 50\nseed: 13\n"
    )
    cfg = load_config(p)
    assert isinstance(cfg, BenchConfig)
    assert cfg.backends == ["fp32", "onnx-int8"]
    assert cfg.seed == 13


def test_rejects_unknown_backend(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("models: [smolvlm-256m]\nbackends: [magic]\ntasks: [sample]\n"
                 "warmup: 0\nrepeats: 1\nsubsample_n: 1\nseed: 1\n")
    with pytest.raises(ValueError):
        load_config(p)


def test_rejects_unknown_model(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("models: [nope]\nbackends: [fp32]\ntasks: [sample]\n"
                 "warmup: 0\nrepeats: 1\nsubsample_n: 1\nseed: 1\n")
    with pytest.raises(ValueError):
        load_config(p)


def test_rejects_unknown_task(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("models: [smolvlm-256m]\nbackends: [fp32]\ntasks: [nope]\n"
                 "warmup: 0\nrepeats: 1\nsubsample_n: 1\nseed: 1\n")
    with pytest.raises(ValueError):
        load_config(p)


def test_rejects_scalar_field(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("models: [smolvlm-256m]\nbackends: [fp32]\ntasks: sample\n"
                 "warmup: 0\nrepeats: 1\nsubsample_n: 1\nseed: 1\n")
    with pytest.raises(ValueError):
        load_config(p)
