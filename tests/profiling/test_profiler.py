from vlmbench.profiling.profiler import StageCallables, profile_cell


def make_fake_clock(values):
    it = iter(values)
    return lambda: next(it)


def test_profiles_each_stage_and_captures_output():
    # Deterministic clock feeds latency profiling for pre/infer/post (warmup=0,
    # repeats=1 each -> 2 reads per stage), then peak-RAM/energy pass needs none.
    seconds = [0.0, 0.001,   # pre  1 ms
               0.0, 0.005,   # infer 5 ms
               0.0, 0.002]   # post 2 ms
    stages = StageCallables(
        pre=lambda: {"pixel": 1},
        infer=lambda x: {"logits": x["pixel"] + 1},
        post=lambda y: f"answer:{y['logits']}",
    )
    profile = profile_cell(
        stages, warmup=0, repeats=1, interval_s=0.0,
        clock=make_fake_clock(seconds),
        sampler=lambda: 42.0,
        energy_reader=iter([0, 1_000_000]).__next__,
    )
    assert profile.output == "answer:2"
    assert round(profile.infer.mean_ms, 3) == 5.0
    assert profile.peak_rss_mb == 42.0
    assert profile.energy_j == 1.0
