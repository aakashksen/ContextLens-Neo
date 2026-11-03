"""tools/run_everything.py: the right Python, a backend that dies at once is reported, and no process outlives it."""

import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import run_everything  # noqa: E402

PARENT = """
import subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
open(sys.argv[1], "w").write(str(child.pid))
time.sleep(60)
"""


def start_tree(tmp_path):
    pidfile = tmp_path / "child.pid"
    parent = run_everything.spawn([sys.executable, "-c", PARENT, str(pidfile)])
    end = time.monotonic() + 15
    while not pidfile.exists() or not pidfile.read_text():
        assert time.monotonic() < end, "the child never started"
        time.sleep(0.05)
    return parent, int(pidfile.read_text())


def test_kill_tree_kills_children_and_grandchildren(tmp_path):
    parent, child = start_tree(tmp_path)
    assert psutil.pid_exists(child)
    run_everything.kill_tree(parent)
    assert parent.poll() is not None
    assert not psutil.pid_exists(child)


def test_stop_kills_everything_started_even_when_start_fails(tmp_path):
    parent, child = start_tree(tmp_path)
    started = [parent]
    with pytest.raises(RuntimeError), run_everything.stopping(started):
        raise RuntimeError("something failed after starting")
    assert not psutil.pid_exists(child) and parent.poll() is not None


def test_backend_python_is_the_venv_with_the_project_installed():
    py = Path(run_everything.backend_python())
    out = subprocess.run([str(py), "-c", "import langgraph, contextlens"], capture_output=True, text=True,
                         env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})  # fmt: skip
    assert out.returncode == 0, out.stderr


def test_a_backend_that_exits_at_once_is_reported_not_waited_for():
    proc = run_everything.spawn([sys.executable, "-c", "raise SystemExit(3)"])
    t0 = time.monotonic()
    with pytest.raises(SystemExit) as err:
        run_everything.wait_for_backend(proc, "the-python", seconds=60)
    assert time.monotonic() - t0 < 10
    assert "exited with code 3" in str(err.value) and "the-python" in str(err.value)


@pytest.mark.real
def test_vite_does_not_outlive_the_script_on_any_path():
    """The real dev server (needs Node and clients/web/node_modules): port 5173 is free again after each stop."""
    import shutil

    npm = shutil.which("npm")
    assert npm and (ROOT / "clients" / "web" / "node_modules").exists()
    assert not run_everything.listening(5173), "port 5173 is already in use"
    for path in ("error", "normal"):
        procs = []
        try:
            with run_everything.stopping(procs):
                procs.append(run_everything.spawn([npm, "--prefix", "clients/web", "run", "dev", "--", "--host", "127.0.0.1",
                                                   "--port", "5173", "--strictPort"], cwd=ROOT))  # fmt: skip
                run_everything.wait_for(5173, "vite", 60)
                if path == "error":
                    raise RuntimeError("the backend died")
        except RuntimeError:
            pass
        end = time.monotonic() + 10
        while run_everything.listening(5173) and time.monotonic() < end:
            time.sleep(0.2)
        assert not run_everything.listening(5173), f"vite is still serving after the {path} path"
