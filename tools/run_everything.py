"""Start the whole stack with one command: python tools/run_everything.py

Postgres and the CPU speech service (docker compose, skipped for a port that already answers), the backend on
127.0.0.1:8000, and the web client on 127.0.0.1:5173. Ollama is not started here: start it yourself with
qwen3.5:4b-q4_K_M and nomic-embed-text pulled. Ctrl-C stops what this script started; volumes are kept.
"""
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OLLAMA = "http://127.0.0.1:11434/api/tags"  # lists models; loads nothing
SERVICES = {"db": 5432, "speech": 50051}


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


def main() -> int:
    npm = shutil.which("npm")
    if npm is None:
        return print("npm was not found; install Node.js to run the web client") or 1
    try:
        urllib.request.urlopen(OLLAMA, timeout=3).close()
    except OSError:
        print("Warning: Ollama does not answer on 127.0.0.1:11434; transcripts and cards need it.")
    started_compose = [name for name, port in SERVICES.items() if not listening(port)]
    if started_compose:
        subprocess.run(["docker", "compose", "up", "-d", *started_compose], cwd=ROOT, check=True)
    for name, port in SERVICES.items():
        wait_for(port, name)
    procs = [
        subprocess.Popen([sys.executable, "-m", "contextlens.api.production"], cwd=ROOT,
                         env={**os.environ, "PYTHONPATH": str(ROOT / "src")}),
        subprocess.Popen([npm, "--prefix", "clients/web", "run", "dev", "--", "--host", "127.0.0.1"], cwd=ROOT),
    ]
    try:
        wait_for(8000, "the backend", 300)
        print("\nReady: open http://127.0.0.1:5173/?server=ws://127.0.0.1:8000  (Ctrl-C to stop)\n")
        while all(p.poll() is None for p in procs):
            time.sleep(1)
        print("A process exited; stopping.")
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()
        if started_compose:
            subprocess.run(["docker", "compose", "stop", *started_compose], cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
