"""The real browser path: headless Edge with a fake microphone that plays tests/fixtures/speech/phrase.wav, the
real web client (Vite dev server), the real backend (uvicorn, the agent team with fake models) and the real
speech service (SPEECH_ADDR, default localhost:50051, CPU). It drives Edge over the DevTools protocol.

It fails if any link is broken: getUserMedia, the AudioContext, the WebSocket, the speech service, the transcript
on the page, the press, the card on the page. Not checked: a human voice in a real room, a physical microphone,
other browsers, a GPU speech service, and the real language models (see test_integration.py)."""

import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
from contextlib import closing
from pathlib import Path

import pytest
import uvicorn
from websockets.sync.client import connect

from contextlens.api import production
from contextlens.api.app import create_app
from contextlens.llm.client import LLMResult
from contextlens.memory.store import InMemoryTranscriptStore
from contextlens.obs.tracing import Telemetry
from contextlens.speech.client import GrpcSpeech

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import run_everything  # noqa: E402

PHRASE = ROOT / "tests" / "fixtures" / "speech" / "phrase.wav"
EDGE_PATHS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]  # fmt: skip


def free_port() -> int:
    with closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Heard(InMemoryTranscriptStore):
    def __init__(self) -> None:
        super().__init__()
        self.seen = []

    def add(self, turn) -> None:
        self.seen.append(turn)
        super().add(turn)


class CitesThursday:
    """Stands in for the language model: cites the turn that the real speech service transcribed."""

    model = "fake"

    def __init__(self, store: Heard) -> None:
        self.store = store

    def complete(self, messages, *, schema=None, **_):
        if schema.__name__ == "_Plan":
            out = {"mode": "recall", "followup": None}
        else:
            turn = next(t for t in self.store.seen if "thursday" in t.text.lower())
            quote = turn.text[turn.text.lower().index("thursday") :][:8]
            out = {"tool": "final", "findings": [{
                "agent": "memory", "kind": "recall", "statement": "The review is on Thursday", "confidence": 0.9,
                "evidence": [{"turn_id": turn.id, "quote": quote, "speaker": "unknown", "start_ms": turn.start_ms}],
            }]}  # fmt: skip
        return LLMResult(text=json.dumps(out), parsed=schema.model_validate(out), model="fake")


class Cdp:
    def __init__(self, url: str) -> None:
        self.ws, self.n = connect(url, max_size=None).__enter__(), 0
        self.session = None

    def call(self, method: str, **params):
        self.n += 1
        msg = {"id": self.n, "method": method, "params": params}
        if self.session:
            msg["sessionId"] = self.session
        self.ws.send(json.dumps(msg))
        while True:
            reply = json.loads(self.ws.recv(timeout=30))
            if reply.get("id") == self.n:
                assert "error" not in reply, reply
                return reply["result"]

    def js(self, expression: str):
        return self.call("Runtime.evaluate", expression=expression, returnByValue=True)["result"].get("value")

    def wait_for(self, expression: str, seconds: float, what: str):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if value := self.js(expression):
                return value
            time.sleep(0.5)
        page = self.js("document.body.innerText")
        raise AssertionError(f"timed out after {seconds} s waiting for {what}; the page says: {page!r}")


@pytest.mark.real
def test_browser_microphone_to_card_through_the_real_client_and_backend():
    edge = shutil.which("msedge") or next((p for p in EDGE_PATHS if os.path.exists(p)), None)
    npm = shutil.which("npm")
    assert edge and npm and (ROOT / "clients" / "web" / "node_modules").exists(), "needs Edge, Node, node_modules"

    store = Heard()
    llm = CitesThursday(store)
    prod = production.build_services(
        store=store, embedder=None, llms={"memory": llm, "consistency": llm, "assistance": llm}, background_llm=None,
        speech=GrpcSpeech(os.environ.get("SPEECH_ADDR", "localhost:50051")), telemetry=Telemetry(),
    )  # fmt: skip
    api_port, web_port = free_port(), free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(prod.services), host="127.0.0.1", port=api_port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    procs: list = []
    profile = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    try:
        with run_everything.stopping(procs):
            procs.append(run_everything.spawn([npm, "--prefix", "clients/web", "run", "dev", "--", "--host", "127.0.0.1",
                                               "--port", str(web_port), "--strictPort"], cwd=ROOT))  # fmt: skip
            run_everything.wait_for(web_port, "vite", 60)
            run_everything.wait_for(api_port, "the backend", 30)
            procs.append(run_everything.spawn([edge, "--headless=new", "--remote-debugging-port=0",
                f"--user-data-dir={profile.name}", "--no-first-run", "--no-default-browser-check",
                "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
                f"--use-file-for-fake-audio-capture={PHRASE}", "--autoplay-policy=no-user-gesture-required",
                "about:blank"]))  # fmt: skip
            port_file = Path(profile.name) / "DevToolsActivePort"
            end = time.monotonic() + 30
            while not (port_file.exists() and len(port_file.read_text().split()) >= 2):
                assert time.monotonic() < end, "Edge did not open its DevTools port"
                time.sleep(0.2)
            dev_port, path = port_file.read_text().split()[:2]
            browser = Cdp(f"ws://127.0.0.1:{dev_port}{path}")
            page = next(t for t in browser.call("Target.getTargets")["targetInfos"] if t["type"] == "page")
            browser.session = browser.call("Target.attachToTarget", targetId=page["targetId"], flatten=True)["sessionId"]
            browser.call("Page.navigate", url=f"http://127.0.0.1:{web_port}/?server=ws://127.0.0.1:{api_port}")

            browser.wait_for("document.body.innerText.includes('Recording')", 30, "the client to connect and start the microphone")
            browser.wait_for("document.querySelector('.transcript li')?.innerText.toLowerCase().includes('thursday')", 90,
                             "the phrase from the fake microphone to appear in the transcript")  # fmt: skip
            browser.js("[...document.querySelectorAll('button')].find(b => b.innerText.includes('What was said')).click()")
            card = browser.wait_for("document.querySelector('article[aria-label=Answer]')?.innerText", 30, "a card")
            assert "thursday" in card.lower() and "Answered" in card, card
            assert not browser.js("document.querySelector('[role=alert]')?.innerText"), "the page shows an error"
    finally:
        server.should_exit = True
        profile.cleanup()
