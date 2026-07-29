"""Bootstrap confidence intervals for per-example task scores.

Accuracy metrics here are means of bounded per-example scores (ANLS in
[0,1], containment in {0,1}), so a nonparametric percentile bootstrap over
the per-example scores gives an honest CI on the reported mean without
distributional assumptions. Pure stdlib (no numpy) so it runs anywhere the
lean stack does.
"""
from __future__ import annotations

import random


def bootstrap_ci(scores, n_boot: int = 10000, ci: float = 0.95,
                 seed: int = 20260722) -> tuple[float, float] | None:
    """Percentile bootstrap CI for the mean of ``scores``.

    Returns ``(low, high)`` at the given central ``ci`` level, or ``None`` if
    ``scores`` is empty. Deterministic given ``seed``.
    """
    scores = list(scores)
    n = len(scores)
    if n == 0:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        total = 0.0
        for _ in range(n):
            total += scores[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    alpha = (1.0 - ci) / 2.0
    lo = means[max(0, int(alpha * n_boot))]
    hi = means[min(n_boot - 1, int((1.0 - alpha) * n_boot))]
    return lo, hi
