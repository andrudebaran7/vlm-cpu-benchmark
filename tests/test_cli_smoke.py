import subprocess
import sys


def test_run_benchmark_help():
    out = subprocess.run(
        [sys.executable, "scripts/run_benchmark.py", "--help"],
        capture_output=True, text=True)
    assert out.returncode == 0
    assert "--config" in out.stdout


def test_make_figures_help():
    out = subprocess.run(
        [sys.executable, "scripts/make_figures.py", "--help"],
        capture_output=True, text=True)
    assert out.returncode == 0
    assert "--results" in out.stdout
