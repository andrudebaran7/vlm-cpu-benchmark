from vlmbench.profiling.memory import sample_peak_rss_mb


def test_returns_result_and_peak_from_injected_sampler():
    readings = iter([100.0, 250.0, 180.0, 250.0, 120.0])

    def fake_sampler():
        try:
            return next(readings)
        except StopIteration:
            return 120.0

    def work():
        # Give the sampler thread time to poll several readings.
        total = 0
        for _ in range(200000):
            total += 1
        return total

    result, peak = sample_peak_rss_mb(work, interval_s=0.0, sampler=fake_sampler)
    assert result == 200000
    assert peak == 250.0
