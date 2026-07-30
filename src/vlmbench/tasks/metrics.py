from __future__ import annotations

import re


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


_YESNO_RE = re.compile(r"\b(yes|no)\b")


def yesno_match(pred: str, golds: list[str]) -> float:
    """Binary yes/no scoring for presence-style questions. Extracts the first
    yes/no token from a free-form answer (VLMs reply e.g. ``"Yes."`` or ``"No,
    there is no person."``; detectors reply exactly ``"yes"``/``"no"``) and
    compares it to the gold yes/no. An answer with no yes/no token scores 0."""
    m = _YESNO_RE.search(_normalize(pred))
    predicted = m.group(1) if m else None
    gold = _normalize(golds[0]) if golds else None
    return 1.0 if predicted is not None and predicted == gold else 0.0


def exact_match(pred: str, golds: list[str]) -> float:
    p = _normalize(pred)
    return 1.0 if any(p == _normalize(g) for g in golds) else 0.0


def containment(pred: str, golds: list[str]) -> float:
    """OCRBench-style scoring: correct if any gold answer appears as a
    substring of the prediction (after lower-casing and whitespace
    normalization). This mirrors the official OCRBench check
    (``answer in prediction``) for its general categories."""
    p = _normalize(pred)
    return 1.0 if any(_normalize(g) in p for g in golds if g) else 0.0


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def anls(pred: str, golds: list[str], threshold: float = 0.5) -> float:
    p = _normalize(pred)
    best = 0.0
    for g in golds:
        gg = _normalize(g)
        denom = max(len(p), len(gg)) or 1
        sim = 1.0 - _levenshtein(p, gg) / denom
        best = max(best, sim)
    return best if best >= threshold else 0.0
