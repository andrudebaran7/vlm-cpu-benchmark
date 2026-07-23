import vlmbench


def test_package_imports_and_has_version():
    assert isinstance(vlmbench.__version__, str)
    assert vlmbench.__version__
