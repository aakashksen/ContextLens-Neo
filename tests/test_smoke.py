import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_package_imports():
    import contextlens

    assert contextlens.__name__ == "contextlens"


def test_real_marker_excluded_by_default():
    config = tomllib.loads((REPO / "pyproject.toml").read_text())
    pytest_opts = config["tool"]["pytest"]["ini_options"]
    assert "-m 'not real'" in pytest_opts["addopts"]
    assert any(m.startswith("real") for m in pytest_opts["markers"])
