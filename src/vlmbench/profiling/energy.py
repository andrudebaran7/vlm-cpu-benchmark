from __future__ import annotations

import glob
import os
from typing import Callable, TypeVar

T = TypeVar("T")

_POWERCAP_GLOB = "/sys/class/powercap/*/energy_uj"


def _is_top_level_domain(energy_path: str) -> bool:
    """True for a top-level RAPL package (e.g. ``intel-rapl:0``), False for a
    nested sub-domain (e.g. ``intel-rapl:0:0`` = core). Nested sub-domains are
    already accounted for in their parent package's energy, so summing them
    would double-count."""
    domain = os.path.basename(os.path.dirname(energy_path))  # e.g. intel-rapl:0
    return domain.count(":") <= 1


def _discover_powercap_files() -> list[str]:
    files = glob.glob(_POWERCAP_GLOB)
    return sorted(p for p in files if _is_top_level_domain(p))


def _default_reader() -> int | None:
    """Sum cumulative energy across all powercap domains, or None if unreadable."""
    files = _discover_powercap_files()
    if not files:
        return None
    total = 0
    read_any = False
    for path in files:
        try:
            with open(path) as handle:
                total += int(handle.read().strip())
                read_any = True
        except (OSError, ValueError):
            continue
    return total if read_any else None


def read_energy_uj(reader: Callable[[], int | None] | None = _default_reader) -> int | None:
    if reader is None:
        return None
    try:
        return reader()
    except Exception:
        return None


def measure_energy_j(
    fn: Callable[[], T],
    *,
    reader: Callable[[], int] | None = _default_reader,
) -> tuple[T, float | None]:
    before = read_energy_uj(reader) if reader is not None else None
    result = fn()
    after = read_energy_uj(reader) if reader is not None else None
    if before is None or after is None or after < before:
        return result, None
    return result, (after - before) / 1_000_000.0
