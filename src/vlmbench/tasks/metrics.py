from __future__ import annotations


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def exact_match(pred: str, golds: list[str]) -> float:
    p = _normalize(pred)
    return 1.0 if any(p == _normalize(g) for g in golds) else 0.0


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
