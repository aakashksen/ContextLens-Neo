import pytest



def pytest_collection_modifyitems(config, items):
    """GPU tests need a GPU turn, so they are collected only when -m names gpu."""
    if "gpu" in (config.getoption("-m") or "").replace("not gpu", ""):
        return
    kept = [i for i in items if "gpu" not in i.keywords]
    config.hook.pytest_deselected(items=[i for i in items if i not in kept])
    items[:] = kept


@pytest.fixture(autouse=True)
def _cpu_unless_real(request, monkeypatch):
    """Offline tests run on the CPU whether or not a GPU exists, through the setting the product reads."""
    if "real" not in request.keywords and "gpu" not in request.keywords:
        monkeypatch.setenv("CONTEXTLENS_DEVICE", "cpu")
