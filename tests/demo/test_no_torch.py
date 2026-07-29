import ast
import pathlib

_DEMO = pathlib.Path(__file__).resolve().parents[2] / "demo"


def _imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_no_demo_file_imports_torch():
    offenders = {p.name for p in _DEMO.glob("*.py") if "torch" in _imports(p)}
    assert offenders == set(), f"demo files import torch: {offenders}"


def test_streamlit_requirements_exclude_torch_stack():
    req = (_DEMO / "requirements-streamlit.txt").read_text().lower()
    for banned in ("torch", "timm", "torchvision"):
        assert banned not in req, f"{banned} must not be in the lean requirements"
