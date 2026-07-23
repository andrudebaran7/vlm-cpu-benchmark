from vlmbench.profiling.energy import measure_energy_j


def test_computes_joules_from_reader_delta():
    readings = iter([1_000_000, 3_500_000])  # microjoules before/after
    result, joules = measure_energy_j(lambda: "done",
                                      reader=lambda: next(readings))
    assert result == "done"
    assert joules == 2.5  # (3.5e6 - 1.0e6) uJ = 2.5 J


def test_returns_none_when_reader_unavailable():
    result, joules = measure_energy_j(lambda: 42, reader=None)
    assert result == 42
    assert joules is None
