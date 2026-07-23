from vlmbench._paths import quant_dir


def test_quant_dir_creates_isolated_path(tmp_path, monkeypatch):
    monkeypatch.setenv("VLMBENCH_CACHE", str(tmp_path / "cache"))
    path = quant_dir("smolvlm-256m", "onnx-int8")
    assert path.exists() and path.is_dir()
    assert path.name == "onnx-int8"
    assert path.parent.name == "smolvlm-256m"
