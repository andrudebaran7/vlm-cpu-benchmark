"""Feasibility spike for the detector baseline. Verifies the four detectors
run on CPU and that COCO/LVIS per-image class presence is computable.
Run: .venv/bin/python scripts/check_detectors.py
"""
from __future__ import annotations

import time

from PIL import Image

from vlmbench.profiling.memory import sample_peak_rss_mb


def _time(fn):
    t0 = time.perf_counter()
    out, peak = sample_peak_rss_mb(fn)
    return out, (time.perf_counter() - t0) * 1000.0, peak


def check_detectors() -> None:
    from ultralytics import YOLO, RTDETR

    img = Image.new("RGB", (640, 640), (120, 120, 120))
    runners = {
        "yolo11n": lambda: YOLO("yolo11n.pt")(img, verbose=False),
        "yolo-world": lambda: YOLO("yolov8s-world.pt")(img, verbose=False),
        "rt-detr": lambda: RTDETR("rtdetr-l.pt")(img, verbose=False),
    }
    for name, fn in runners.items():
        try:
            _, ms, peak = _time(fn)
            print(f"{name:12} ran on CPU: {ms:.0f} ms, peak {peak:.0f} MB")
        except Exception as exc:
            print(f"{name:12} FAILED: {type(exc).__name__}: {exc}")
    try:
        from rfdetr import RFDETRNano
        m = RFDETRNano()
        _, ms, peak = _time(lambda: m.predict(img))
        print(f"{'rfdetr-nano':12} ran on CPU: {ms:.0f} ms, peak {peak:.0f} MB")
    except Exception as exc:
        print(f"{'rfdetr-nano':12} FAILED: {type(exc).__name__}: {exc}")


def check_data() -> None:
    from datasets import load_dataset

    for name in ("detection-datasets/coco",):
        try:
            ds = load_dataset(name, split="val", streaming=True)
            ex = next(iter(ds))
            print(f"COCO ({name}) keys:", list(ex.keys()))
            if "objects" in ex:
                print("  objects subkeys:", list(ex["objects"].keys()))
        except Exception as exc:
            print(f"COCO ({name}) FAILED: {type(exc).__name__}: {exc}")
    for name in ("lvis", "winvoker/lvis"):
        try:
            ds = load_dataset(name, split="validation", streaming=True)
            ex = next(iter(ds))
            print(f"LVIS ({name}) keys:", list(ex.keys()))
            break
        except Exception as exc:
            print(f"LVIS ({name}) FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    check_detectors()
    check_data()
