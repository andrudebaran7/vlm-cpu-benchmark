from __future__ import annotations

from pathlib import Path

import pandas as pd

from .records import CellResult

_COLUMNS = [
    "model", "backend", "task", "status", "metric_name", "metric_value",
    "infer_ms_mean", "infer_ms_p95", "peak_rss_mb", "energy_j", "error",
]


def to_dataframe(results: list[CellResult]) -> pd.DataFrame:
    rows = [r.to_dict() for r in results]
    return pd.DataFrame(rows, columns=_COLUMNS)


def write_csv(results: list[CellResult], path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_dataframe(results).to_csv(path, index=False)
    return path
