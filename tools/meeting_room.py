"""The meeting room demo server.

Replay (no model, no speech service; the AMI transcript with its own timings stands in):
    .venv/Scripts/python tools/meeting_room.py
Live (real speech service + diarizer + stream API + local models; needs Ollama, the speech container, HF_TOKEN):
    .venv/Scripts/python tools/meeting_room.py --live
Then open http://127.0.0.1:8790/lab.html (the public room.html is the short scripted demo)

Listens on 127.0.0.1 only. AMI Meeting Corpus, CC BY 4.0, https://groups.inf.ed.ac.uk/ami/corpus/ .
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SITE = ROOT / "site"


def default_raw() -> Path:
    """datasets/raw of this checkout, else of the main checkout this worktree belongs to (datasets are not in git)."""
    here = ROOT / "datasets" / "raw"
    if (here / "ami-audio").is_dir():
        return here
    try:
        common = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        return (ROOT / common).resolve().parent / "datasets" / "raw"
    except (subprocess.CalledProcessError, OSError):
        return here


def build_app(raw: Path, live: bool, port: int, speech_addr: str):
    from fastapi import FastAPI

    from contextlens.room.server import add_room

    if not live:
        app = FastAPI(title="ContextLens Neo meeting room (replay)")
        add_room(app, raw, site=SITE)
        return app

    from contextlens.api import production as P
    from contextlens.api.app import create_app
    from contextlens.memory.store import InMemoryTranscriptStore
    from contextlens.room.rig import make_rig
    from contextlens.speech.client import GrpcSpeech
    from contextlens.speech.diarize import PyannoteBackend, SharedSpeakerModel

    embedder, llms = P.local_embedder(), P.local_llms()
    prod = P.build_services(
        store=InMemoryTranscriptStore(), embedder=embedder, llms=llms, background_llm=llms["memory"],
        speech=GrpcSpeech(speech_addr), make_backend=SharedSpeakerModel(PyannoteBackend), assistant="team_routed",
        warmup=P.local_warmup(llms["memory"], embedder), unload=P.unload_local_models,
    )
    app = create_app(prod.services)  # the real /ws/{conversation_id}, its lifespan, warmup and shutdown
    rig = make_rig(prod.services, llms, f"ws://127.0.0.1:{port}", build=P.build_assistant,
                   checker=lambda: P.cascade_checker(llms["consistency"]))
    add_room(app, raw, rig=rig, site=SITE)
    return app


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--live", action="store_true", help="real speech service, diarizer, stream API and local models")
    ap.add_argument("--speech", default="localhost:50051", help="speech service address (live only)")
    ap.add_argument("--raw", type=Path, default=None, help="datasets/raw folder holding ami/ and ami-audio/")
    a = ap.parse_args()
    raw = a.raw or default_raw()
    print(f"mode: {'live' if a.live else 'replay'}; data: {raw}", flush=True)
    uvicorn.run(build_app(raw, a.live, a.port, a.speech), host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
