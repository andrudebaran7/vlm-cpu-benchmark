from vlmbench.report.stats import bootstrap_ci


def test_ci_brackets_the_mean():
    scores = [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0]  # mean 0.625
    lo, hi = bootstrap_ci(scores, n_boot=2000, seed=1)
    assert lo <= 0.625 <= hi
    assert 0.0 <= lo < hi <= 1.0


def test_zero_variance_gives_degenerate_interval():
    lo, hi = bootstrap_ci([0.5, 0.5, 0.5, 0.5], n_boot=500, seed=1)
    assert lo == hi == 0.5


def test_deterministic_given_seed():
    s = [0.1, 0.9, 0.4, 0.7, 0.2, 0.8]
    assert bootstrap_ci(s, n_boot=1000, seed=42) == bootstrap_ci(s, n_boot=1000, seed=42)


def test_empty_scores_returns_none():
    assert bootstrap_ci([], n_boot=100, seed=1) is None


def test_wider_interval_for_smaller_sample():
    import random
    rng = random.Random(0)
    pop = [rng.random() for _ in range(1000)]
    small = pop[:20]
    large = pop[:500]
    ws = bootstrap_ci(small, n_boot=3000, seed=7)
    wl = bootstrap_ci(large, n_boot=3000, seed=7)
    assert (ws[1] - ws[0]) > (wl[1] - wl[0])  # smaller N => wider CI
