from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path


class CellStatus(str, Enum):
    OK = "ok"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    OOM = "oom"


@dataclass
class CellResult:
    model: str
    backend: str
    task: str
    status: CellStatus
    metric_name: str | None
    metric_value: float | None
    infer_ms_mean: float | None
    infer_ms_p95: float | None
    peak_rss_mb: float | None
    energy_j: float | None
    error: str | None

    def key(self) -> tuple[str, str, str]:
        return (self.model, self.backend, self.task)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "CellResult":
        data = dict(data)
        data["status"] = CellStatus(data["status"])
        return cls(**data)


class JsonlStore:
    def __init__(self, path) -> None:
        self.path = Path(path)

    def append(self, result: CellResult) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(result.to_dict()) + "\n")

    def load(self) -> list[CellResult]:
        if not self.path.exists():
            return []
        out = []
        with self.path.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    out.append(CellResult.from_dict(json.loads(line)))
        return out

    def completed_keys(self) -> set[tuple[str, str, str]]:
        return {r.key() for r in self.load()}
