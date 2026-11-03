
def pytest_collection_modifyitems(config, items):
    """GPU tests need a GPU turn, so they are collected only when -m names gpu."""
    if "gpu" in (config.getoption("-m") or "").replace("not gpu", ""):
        return
    kept = [i for i in items if "gpu" not in i.keywords]
    config.hook.pytest_deselected(items=[i for i in items if i not in kept])
    items[:] = kept
