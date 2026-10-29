"""Smoke test - kiểm tra package import OK."""


def test_package_import():
    import src

    assert src.__version__
