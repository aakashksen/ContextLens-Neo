"""Local live demo: serves site/ and a small API that runs the real assistant on local Ollama models.

Run:  .venv/Scripts/python tools/live_demo.py [--port 8790]
Then open http://127.0.0.1:8790/live.html

Same code path as production (TeamPerPress / _SinglePerPress), with the in-memory transcript store and no speech service.
Listens on 127.0.0.1 only. Local models only, no outside calls.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import uuid
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SITE = ROOT / "site"
MAX_TURNS, MAX_TURN_CHARS, MAX_QUESTION_CHARS, MAX_BODY = 40, 600, 300, 64_000
KINDS = ("single", "team")
_LOCK = threading.Lock()  # one local GPU: presses run one at a time


class BadRequest(ValueError):
    pass


def parse(body: object) -> tuple[list[dict], str | None, str]:
    if not isinstance(body, dict):
        raise BadRequest("body must be a JSON object")
    turns, question, kind = body.get("turns"), body.get("question"), body.get("assistant", "team")
    if kind not in KINDS:
        raise BadRequest("assistant must be single or team")
    if not isinstance(turns, list) or not 1 <= len(turns) <= MAX_TURNS:
        raise BadRequest(f"turns must be a list of 1 to {MAX_TURNS}")
    clean = []
    for t in turns:
        speaker, text = (t.get("speaker"), t.get("text")) if isinstance(t, dict) else (None, None)
        if not (isinstance(speaker, str) and isinstance(text, str) and speaker.strip() and text.strip()):
            raise BadRequest("each turn needs a speaker and text")
        clean.append({"speaker": speaker.strip()[:40], "text": text.strip()[:MAX_TURN_CHARS]})
    if question is not None and not isinstance(question, str):
        raise BadRequest("question must be a string or null")
    question = (question or "").strip()[:MAX_QUESTION_CHARS] or None
    return clean, question, kind


def build_assistant(kind: str, store, deadline_s: float | None = None):
    """The production per-press assistants, bound to this press's store."""
    from contextlens.api.production import DEADLINE_S, TeamPerPress, _SinglePerPress, local_embedder, local_llms
    from contextlens.memory.extract import MemoryStore
    from contextlens.retrieval.hybrid import Retriever

    cls = _SinglePerPress if kind == "single" else TeamPerPress
    return cls(store, Retriever(store, local_embedder()), MemoryStore(), local_llms(), deadline_s or DEADLINE_S)


def press(body: object, factory=build_assistant) -> dict:
    from contextlens.memory.store import InMemoryTranscriptStore
    from contextlens.replay.runner import Context
    from contextlens.schemas import ButtonRequest, Turn

    turns, question, kind = parse(body)
    conv = "live-" + uuid.uuid4().hex[:8]
    store = InMemoryTranscriptStore()
    for i, t in enumerate(turns, 1):
        store.add(Turn(id=f"t{i}", conversation_id=conv, seq=i, speaker=t["speaker"], text=t["text"],
                       start_ms=(i - 1) * 4000, end_ms=i * 4000 - 500, source="synthetic"))
    req = ButtonRequest(conversation_id=conv, as_of_seq=len(turns), question=question)
    with _LOCK:
        card = factory(kind, store).answer(req, Context(store=store, as_of_seq=len(turns)))
    return {"assistant": kind, "question": question, "card": card.model_dump(mode="json"),
            "latency_ms": card.latency_ms, "tokens": card.tokens}


def warm() -> None:
    from contextlens.api.production import local_embedder, local_llms

    local_llms()["assistance"].load()
    local_embedder().embed(["warm up"])


class Handler(SimpleHTTPRequestHandler):
    press_fn = staticmethod(press)

    def _json(self, code: int, obj: dict) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/health":
            return self._json(200, {"ok": True, "model": "qwen3.5:4b-q4_K_M", "assistants": list(KINDS)})
        super().do_GET()

    def do_POST(self):
        if self.path != "/api/press":
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > MAX_BODY:
                raise BadRequest("body too large")
            result = self.press_fn(json.loads(self.rfile.read(n) or b"null"))
        except (BadRequest, json.JSONDecodeError) as err:
            return self._json(400, {"error": str(err)})
        except Exception as err:  # the model or Ollama failed: say what kind, not the content
            return self._json(502, {"error": f"the local model failed ({type(err).__name__})"})
        self._json(200, result)

    def log_message(self, *a):
        pass


def make_server(port: int, press_fn=press) -> ThreadingHTTPServer:
    handler = type("H", (Handler,), {"press_fn": staticmethod(press_fn)})
    return ThreadingHTTPServer(("127.0.0.1", port), partial(handler, directory=str(SITE)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8790)
    port = ap.parse_args().port
    print("warming the local models ...", flush=True)
    warm()
    srv = make_server(port)
    print(f"ready: http://127.0.0.1:{port}/live.html", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
