from vlmbench.profiling.environ import collect_environment


def test_environment_has_expected_keys_and_types():
    env = collect_environment()
    assert isinstance(env["python_version"], str)
    assert isinstance(env["platform"], str)
    assert isinstance(env["cpu_count"], int) and env["cpu_count"] >= 1
    assert "torch_threads" in env
    assert isinstance(env["package_versions"], dict)
    assert "torch" in env["package_versions"]
