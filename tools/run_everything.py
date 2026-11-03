"""Start the whole stack with one command: python tools/run_everything.py

Postgres and the CPU speech service (docker compose, skipped for a port that already answers), the backend on
127.0.0.1:8000, and the web client on 127.0.0.1:5173. Ollama is not started here: start it yourself with
qwen3.5:4b-q4_K_M and nomic-embed-text pulled. The backend loads both models at startup; this script waits for
that. Ctrl-C, an error, or a process that exits stops everything this script started, whole process trees;
volumes are kept.
"""
import contextlib
import importlib.util
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OLLAMA = "http://127.0.0.1:11434/api/tags"  # lists models; loads nothing
READY = "http://127.0.0.1:8000/ready"
SERVICES = {"db": 5432, "speech": 50051}
WINDOWS = os.name == "nt"


def listening(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait_for(port: int, what: str, seconds: int = 120) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if listening(port):
            return
        time.sleep(1)
    raise SystemExit(f"{what} did not start listening on port {port} in {seconds} s")


def backend_python() -> str:
    """The project's own virtual environment: the system Python has no langgraph."""
    for rel in (".venv/Scripts/python.exe", ".venv/bin/python"):
        if (ROOT / rel).exists():
            return str(ROOT / rel)
    if importlib.util.find_spec("langgraph") is not None:
        return sys.executable
    raise SystemExit("No .venv found in the project and this Python lacks langgraph; create the venv first (README).")


def spawn(cmd: list[str], **kw) -> subprocess.Popen:
    """Own process group (Windows) or session (POSIX), so Ctrl-C reaches only this script and kill_tree finds
    the whole tree intact."""
    if WINDOWS:
        return subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP, **kw)
    return subprocess.Popen(cmd, start_new_session=True, **kw)


def kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is None or WINDOWS:  # on Windows a dead parent may still have children; taskkill finds none then
        if WINDOWS:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
        else:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(10)


@contextlib.contextmanager
def stopping(procs: list[subprocess.Popen]):
    """Whatever happens inside, every process in `procs` (a list that may still grow) is killed with its tree."""
    try:
        yield
    finally:
        for p in procs:
            kill_tree(p)


def wait_for_backend(proc: subprocess.Popen, python: str, seconds: int = 300) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if proc.poll() is not None:
            raise SystemExit(
                f"The backend exited with code {proc.returncode} right after starting (see its error above). "
                f"Run `{python} -m contextlens.api.production` with PYTHONPATH=src to reproduce it."
            )
        if listening(8000):
            return
        time.sleep(0.5)
    raise SystemExit(f"the backend did not start listening on port 8000 in {seconds} s")


def wait_until_warm(proc: subprocess.Popen, seconds: int = 900) -> None:
    """The backend refuses connections until Ollama has loaded both models; its /ready says why it is not."""
    end, last = time.monotonic() + seconds, {}
    while time.monotonic() < end:
        if proc.poll() is not None:
            raise SystemExit(f"The backend exited with code {proc.returncode} while the models were loading.")
        try:
            with urllib.request.urlopen(READY, timeout=3) as r:
                print("Models loaded (ms):", json.load(r)["cold_start_ms"], flush=True)
                return
        except urllib.error.HTTPError as err:
            last = json.load(err)
        except OSError:
            pass
        time.sleep(2)
    raise SystemExit(f"The models did not finish loading in {seconds} s; last report: {last}. Is Ollama running?")


def main() -> int:
    npm = shutil.which("npm")
    if npm is None:
        return print("npm was not found; install Node.js to run the web client") or 1
    python = backend_python()
    try:
        urllib.request.urlopen(OLLAMA, timeout=3).close()
    except OSError:
        print("Warning: Ollama does not answer on 127.0.0.1:11434; the backend cannot load its models without it.", flush=True)
    started_compose = [name for name, port in SERVICES.items() if not listening(port)]
    procs: list[subprocess.Popen] = []
    try:
        with stopping(procs):
            if started_compose:
                subprocess.run(["docker", "compose", "up", "-d", *started_compose], cwd=ROOT, check=True)
            for name, port in SERVICES.items():
                wait_for(port, name)
            backend = spawn([python, "-m", "contextlens.api.production"], cwd=ROOT,
                            env={**os.environ, "PYTHONPATH": str(ROOT / "src")})  # fmt: skip
            procs.append(backend)
            procs.append(spawn([npm, "--prefix", "clients/web", "run", "dev", "--", "--host", "127.0.0.1"], cwd=ROOT))
            wait_for_backend(backend, python)
            wait_until_warm(backend)
            print("\nReady: open http://127.0.0.1:5173/?server=ws://127.0.0.1:8000  (Ctrl-C to stop)\n", flush=True)
            while all(p.poll() is None for p in procs):
                time.sleep(1)
            print("A process exited; stopping.")
            return 1
    except KeyboardInterrupt:
        return 0
    finally:
        if started_compose:
            subprocess.run(["docker", "compose", "stop", *started_compose], cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
