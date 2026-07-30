"""Shared helpers for detector adapters that answer the presence question."""
from __future__ import annotations

import re

_CLS_RE = re.compile(r"is there an? (.+?) in this image", re.IGNORECASE)


def target_class(prompt: str) -> str:
    """Extract the queried class from the fixed presence prompt template."""
    m = _CLS_RE.search(prompt)
    return (m.group(1) if m else prompt).strip().lower()
