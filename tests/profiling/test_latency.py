from vlmbench.profiling.latency import profile_latency, LatencyStats


def make_fake_clock(values):
    it = iter(values)
    return lambda: next(it)


def test_excludes_warmup_and_computes_ms_stats():
    # 1 warmup call (2 clock reads) then 3 measured calls of 10, 20, 30 ms.
    seconds = [0.0, 0.5,           # warmup: ignored
               0.0, 0.010,          # 10 ms
               0.0, 0.020,          # 20 ms
               0.0, 0.030]          # 30 ms
    stats = profile_latency(lambda: None, warmup=1, repeats=3,
                            clock=make_fake_clock(seconds))
    assert isinstance(stats, LatencyStats)
    assert stats.n == 3
    assert round(stats.mean_ms, 3) == 20.0
    assert round(stats.median_ms, 3) == 20.0
    assert stats.p95_ms >= stats.median_ms
    assert stats.std_ms > 0
