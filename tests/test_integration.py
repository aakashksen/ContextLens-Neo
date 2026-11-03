"""The real backend wired as in production (agent team, speech client, shared speaker model, telemetry, vision
store), with fakes only for the models, the speech service and the stores. Real-marked tests use the real ones."""

import base64
import json
import os
import queue
import sys
import threading
import time
import uuid
import wave
from pathlib import Path

import jsonschema
import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry
from starlette.websockets import WebSocketDisconnect

from contextlens.api import production
from contextlens.api.app import create_app
from contextlens.api.contract import TurnUpdateMessage, contract_schema, server_adapter
from contextlens.llm.client import FakeLLM
from contextlens.memory.store import InMemoryTranscriptStore
from contextlens.obs import tracing
from contextlens.obs.tracing import Telemetry
from contextlens.replay.runner import Context
from contextlens.schemas import ButtonRequest, Turn
from contextlens.speech.diarize import SharedSpeakerModel
from test_api import ONE_SECOND, FakeSpeech, NoEmbed, Spans, card_for, press, wait_for

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import fake_server  # noqa: E402

SCHEMA = json.loads((ROOT / "docs" / "client-contract.schema.json").read_text("utf-8"))


def finding(agent, turn_id, quote):
    return {"agent": agent, "kind": "recall", "statement": f"{agent} found it", "confidence": 0.9,
            "evidence": [{"turn_id": turn_id, "quote": quote, "speaker": "unknown", "start_ms": 0}]}  # fmt: skip


def team_llms():
    mem = {"tool": "final", "findings": [finding("memory", "c1-t0", "The deadline is Monday")]}
    con = {"tool": "final", "findings": [finding("consistency", "c1-t2", "moved it to Wednesday")]}
    return {"memory": FakeLLM([mem] * 3), "consistency": FakeLLM([con] * 3),
            "assistance": FakeLLM([{"mode": "recall", "followup": None}] * 3)}  # fmt: skip


def make(backend=None, speech=None, registry=None, **kw):
    prod = production.build_services(
        store=InMemoryTranscriptStore(), embedder=NoEmbed(), llms=team_llms(), background_llm=None,
        speech=speech or FakeSpeech(), make_backend=(lambda: backend) if backend else None,
        telemetry=Telemetry(registry=registry or CollectorRegistry()), **kw,
    )  # fmt: skip
    prod.services.speech_timeout_s = 0.3
    return prod, TestClient(create_app(prod.services))


def frames_until(ws, stop, limit=40):
    out = []
    for _ in range(limit):
        out.append(ws.receive_json())
        if stop(out[-1]):
            return out
    raise AssertionError(f"never saw the frame we waited for: {out}")


def is_type(t):
    return lambda m: m["type"] == t


def test_turn_update_is_part_of_the_contract():
    types = {d["properties"]["type"]["const"] for d in SCHEMA["server_to_client"]["$defs"].values()
             if "type" in d.get("properties", {})}  # fmt: skip
    assert "turn_update" in types
    assert contract_schema() == SCHEMA
    msg = server_adapter.validate_json(TurnUpdateMessage(turn=Turn(
        id="c-t0", conversation_id="c", seq=0, speaker="speaker_1", text="x", start_ms=0, end_ms=1, source="asr",
    )).model_dump_json())  # fmt: skip
    assert msg.type == "turn_update"


def test_real_backend_matches_contract():
    prod, client = make(backend=Spans())
    seen = set()

    def check(m):
        jsonschema.validate(m, SCHEMA["server_to_client"])
        seen.add(m["type"])
        return m

    def send(ws, payload):
        jsonschema.validate(payload, SCHEMA["client_to_server"])
        ws.send_json(payload)

    with client.websocket_connect("/ws/c1") as ws:
        assert check(ws.receive_json())["state"] == "recording"
        for _ in range(3):
            send(ws, ONE_SECOND)
        got = [check(m) for m in frames_until(ws, lambda m: m["type"] == "turn_update" and m["turn"]["seq"] == 2)]
        sent = [m["turn"] for m in got if m["type"] == "transcript"]
        assert [t["seq"] for t in sent] == [0, 1, 2]
        send(ws, press(as_of=2, question="What was the deadline?"))
        card = check(frames_until(ws, is_type("card"))[-1])["card"]
        assert card["status"] == "answered"
        assert {e["turn_id"] for f in card["findings"] for e in f["evidence"]} <= {t["id"] for t in sent}
        send(ws, {"type": "pause"})
        assert check(frames_until(ws, is_type("status"))[-1])["state"] == "paused"
        send(ws, {"type": "resume"})
        assert check(frames_until(ws, is_type("status"))[-1])["state"] == "recording"
        ws.send_text("not json")
        assert check(frames_until(ws, is_type("status"))[-1])["state"] == "error"
        send(ws, {"type": "delete"})
        assert check(frames_until(ws, is_type("status"))[-1])["state"] == "deleted"
    assert seen == {"transcript", "turn_update", "card", "status"}


def test_speaker_assigned_later_reaches_the_client_once_per_change():
    prod, client = make(backend=Spans())
    with client.websocket_connect("/ws/c1") as ws:
        ws.receive_json()
        for _ in range(2):
            ws.send_json(ONE_SECOND)
        got = frames_until(ws, lambda m: m["type"] == "turn_update" and m["turn"]["seq"] == 1)
        first = [m for m in got if m["type"] == "transcript"]
        assert {t["turn"]["speaker"] for t in first} == {"unknown"}
        ups = {m["turn"]["seq"]: m["turn"]["speaker"] for m in got if m["type"] == "turn_update"}
        assert ups == {0: "speaker_1", 1: "speaker_2"}
        ws.send_json(ONE_SECOND)
        more = frames_until(ws, lambda m: m["type"] == "turn_update" and m["turn"]["seq"] == 2)
        assert [m["turn"]["seq"] for m in more if m["type"] == "turn_update"] == [2]  # 0 and 1 are not resent


def test_nothing_arrives_after_deleted():
    speech, spans = FakeSpeech(), Spans()
    spans.hold = threading.Event()
    prod, client = make(backend=spans, speech=speech)
    speech.gate = threading.Event()
    with client.websocket_connect("/ws/c1") as ws:
        ws.receive_json()
        ws.send_json(ONE_SECOND)  # its transcript is held inside the speech service
        assert wait_for(lambda: speech.n == 1)
        ws.send_json({"type": "delete"})
        assert ws.receive_json()["state"] == "deleted"
        speech.gate.set()  # the late segment is released after the delete
        spans.hold.set()
        time.sleep(0.5)
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()
    assert prod.services.store.turns("c1", 99) == []


def run_script(ws, last_frame_is_close=True):
    """The same client script against any server; returns every frame, in order."""
    frames = []

    def until(pred):
        frames.extend(frames_until(ws, pred))

    until(is_type("status"))
    for _ in range(5):
        ws.send_json(ONE_SECOND)
    seqs = set()
    while len(seqs) < 5:
        until(is_type("transcript"))
        seqs.add(frames[-1]["turn"]["seq"])
    ws.send_json(press(as_of=4))
    until(is_type("card"))
    ws.send_json({"type": "pause"})
    until(lambda m: m["type"] == "status" and m["state"] == "paused")
    ws.send_json({"type": "resume"})
    until(lambda m: m["type"] == "status" and m["state"] == "recording")
    ws.send_text("not json")
    until(lambda m: m["type"] == "status" and m["state"] == "error")
    ws.send_json({"type": "delete"})
    until(lambda m: m["type"] == "status" and m["state"] == "deleted")
    return frames


def paths(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from paths(v, f"{prefix}.{k}")
    elif isinstance(x, list):
        for v in x:
            yield from paths(v, prefix + "[]")
    else:
        yield prefix, type(x).__name__


def shapes(frames):
    by_type: dict[str, dict[str, set]] = {}
    for m in frames:
        slot = by_type.setdefault(m["type"], {})
        for path, t in paths(m):
            slot.setdefault(path, set()).add(t)
    return by_type


def test_fake_server_and_real_backend_send_same_shapes():
    with TestClient(fake_server.create_app()).websocket_connect("/ws/c1") as ws:
        fake = run_script(ws)
    prod, client = make(backend=Spans())
    with client.websocket_connect("/ws/c1") as ws:
        real = run_script(ws)
    # turn updates: the fake sends them from its fifth turn on, the real backend whenever a speaker is assigned
    assert {m["type"] for m in fake} == {m["type"] for m in real} == {"status", "transcript", "turn_update", "card"}

    def steps(frames):
        return [(m["type"], m.get("state")) for m in frames if m["type"] != "turn_update"]

    assert steps(fake) == steps(real)
    a, b = shapes(fake), shapes(real)
    for kind in a:
        assert a[kind].keys() == b[kind].keys(), kind
        for path in a[kind]:
            ta, tb = a[kind][path] - {"NoneType"}, b[kind][path] - {"NoneType"}
            assert not ta or not tb or ta == tb, (kind, path, ta, tb)


# -- shared speaker model, idle eviction --


def test_one_speaker_model_is_loaded_for_all_conversations():
    loads = []

    def load():
        loads.append(1)
        return Spans()

    shared = SharedSpeakerModel(load)
    got = []
    threads = [threading.Thread(target=lambda: got.append(shared())) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert loads == [1] and len({id(b) for b in got}) == 1


def test_speaker_model_runs_one_conversation_at_a_time_and_a_failed_load_is_not_retried():
    inside, overlap = [0], []

    class Slow:
        def spans(self, pcm):
            inside[0] += 1
            overlap.append(inside[0])
            time.sleep(0.05)
            inside[0] -= 1
            return []

    backend = SharedSpeakerModel(Slow)()
    ts = [threading.Thread(target=backend.spans, args=(b"",)) for _ in range(3)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert max(overlap) == 1

    calls = []

    def broken():
        calls.append(1)
        raise RuntimeError("terms not accepted")

    shared = SharedSpeakerModel(broken)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            shared()
    assert calls == [1]


def test_idle_conversations_are_evicted_and_continue_from_the_store():
    prod, client = make(idle_evict_s=0.05)
    app_state = client.app.state
    with client.websocket_connect("/ws/c1") as ws:
        ws.receive_json()
        ws.send_json(ONE_SECOND)
        assert ws.receive_json()["turn"]["seq"] == 0
        with client.websocket_connect("/ws/c2") as other:  # c1 is busy: connected
            other.receive_json()
            assert set(app_state.conversations) == {"c1", "c2"}
    time.sleep(0.15)
    with client.websocket_connect("/ws/c3") as ws:  # a new connection sweeps
        ws.receive_json()
        assert set(app_state.conversations) == {"c3"}
    with client.websocket_connect("/ws/c1") as ws:
        ws.receive_json()
        ws.send_json(ONE_SECOND)
        assert ws.receive_json()["turn"]["seq"] == 1  # the sequence continues from the store
        ws.send_json(press(as_of=1))
        assert ws.receive_json()["type"] == "card"


# -- telemetry and vision --


class Stub:
    def __init__(self):
        self.seen = []

    def answer(self, req, ctx):
        self.seen.append((id(ctx), dict(ctx.stage_ms)))
        ctx.stage_ms["memory_ms"] = 5
        t = ctx.store.turns(req.conversation_id, ctx.as_of_seq)[-1]
        return card_for(t.id, t.text)


def test_each_press_gets_a_fresh_context():
    prod, client = make()
    prod.traced.inner = stub = Stub()
    with client.websocket_connect("/ws/c1") as ws:
        ws.receive_json()
        ws.send_json(ONE_SECOND)
        ws.receive_json()
        for _ in range(2):
            ws.send_json(press(as_of=0))
            assert ws.receive_json()["type"] == "card"
    assert len({i for i, _ in stub.seen}) == 2 and all(stage == {} for _, stage in stub.seen)


def test_delete_forgets_the_scene_note_and_queued_telemetry():
    reg = CollectorRegistry()
    prod, client = make(registry=reg)
    from contextlens.vision.local import SceneNote

    prod.scenes.put("c1", SceneNote(id="scene:1", conversation_id="c1", text="a whiteboard"))
    prod.scenes.put("c2", SceneNote(id="scene:2", conversation_id="c2", text="a door"))
    release = threading.Event()
    real_record = prod.telemetry.record
    prod.telemetry.record = lambda *a: (release.wait(5), real_record(*a))[1]
    prod.traced.inner = Stub()
    store = prod.services.store
    for c in ("c1", "c2"):
        store.add(Turn(id=f"{c}-t0", conversation_id=c, seq=0, speaker="u", text="hi", start_ms=0, end_ms=1, source="asr"))

    def ask(c):
        prod.traced.answer(ButtonRequest(conversation_id=c, as_of_seq=0), Context(store=store, as_of_seq=0))

    ask("c2")  # the worker picks this one up and waits
    wait_for(lambda: prod.traced._pending)
    time.sleep(0.1)
    ask("c1")
    ask("c1")
    ask("c2")
    with client.websocket_connect("/ws/c1") as ws:
        ws.receive_json()
        ws.send_json({"type": "delete"})
        assert ws.receive_json()["state"] == "deleted"
    assert prod.scenes.latest("c1") is None and prod.scenes.latest("c2") is not None
    release.set()
    prod.traced.flush(5)
    assert reg.get_sample_value("contextlens_agent_seconds_count", {"agent": "memory"}) == 2  # c2's two; c1's dropped


def test_dropped_presses_are_counted_in_a_metric(monkeypatch):
    monkeypatch.setattr(tracing, "MAX_PENDING", 1)
    reg = CollectorRegistry()
    prod, _ = make(registry=reg)
    prod.traced.inner = Stub()
    release = threading.Event()
    prod.telemetry.record = lambda *a: release.wait(5)
    store = prod.services.store
    store.add(Turn(id="c1-t0", conversation_id="c1", seq=0, speaker="u", text="hi", start_ms=0, end_ms=1, source="asr"))
    for _ in range(3):
        prod.traced.answer(ButtonRequest(conversation_id="c1", as_of_seq=0), Context(store=store, as_of_seq=0))
    assert reg.get_sample_value("contextlens_dropped_presses_total") == 2
    release.set()
    prod.traced.flush(5)


def test_shutdown_flushes_telemetry_with_a_timeout():
    prod, client = make()
    calls = []
    prod.traced.flush = lambda timeout_s=5.0: calls.append(timeout_s)
    with client:
        pass
    assert calls == [production.SHUTDOWN_FLUSH_S]


def test_production_uses_local_models_only():
    llms = production.local_llms()
    clients = {id(c): c for c in llms.values()}.values()
    assert {(type(c).__name__, c.model, c.base_url) for c in clients} == {
        ("OllamaClient", "qwen3.5:4b-q4_K_M", "http://127.0.0.1:11434")
    }
    assert production.local_embedder().base_url == "http://127.0.0.1:11434"
    assert "ClaudeCodeClient" not in Path(production.__file__).read_text("utf-8")


# -- real services; run only when asked (-m real) --

PHRASE = ROOT / "tests" / "fixtures" / "speech" / "phrase.wav"


class Frames:
    """Reads a websocket on a thread so a stuck server fails the test instead of hanging it."""

    def __init__(self, ws):
        self.ws, self.q = ws, queue.Queue()
        self.log = []
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        try:
            while True:
                self.q.put(self.ws.receive_json())
        except Exception as exc:
            self.q.put({"type": "closed", "why": type(exc).__name__})

    def wait(self, pred, timeout):
        end = time.monotonic() + timeout
        while True:
            try:
                m = self.q.get(timeout=max(0.01, end - time.monotonic()))
            except queue.Empty:
                seen = [(m["type"], m.get("state")) for m in self.log]
                raise AssertionError(f"timed out after {timeout} s; frames seen so far: {seen}") from None
            self.log.append(m)
            if pred(m):
                return m


def speak(ws):
    with wave.open(str(PHRASE)) as w:
        pcm = w.readframes(w.getnframes())
    pcm += bytes(32000 * 3)  # silence so the last words are finalised
    for i in range(0, len(pcm), 6400):
        ws.send_json({"type": "audio", "data": base64.b64encode(pcm[i : i + 6400]).decode()})


def hashed_embedder():
    class Hash:
        def embed(self, texts):
            return [[(hash(w) % 997) / 997 for w in (t.split() + [""] * 768)[:768]] for t in texts]

    return Hash()


@pytest.mark.real
def test_audio_reaches_a_transcript_through_real_speech_postgres_and_pyannote():
    """Real speech service (SPEECH_ADDR), real Postgres (DATABASE_URL), real pyannote on the CPU; models faked."""
    from contextlens.memory.store import PostgresTranscriptStore
    from contextlens.speech.client import GrpcSpeech
    from contextlens.speech.diarize import PyannoteBackend

    conv = f"it-{uuid.uuid4().hex[:8]}"
    store = PostgresTranscriptStore(production.database_url(), hashed_embedder())
    shared = SharedSpeakerModel(PyannoteBackend)
    shared()  # load now: audio heard before a conversation has its speaker model is not diarized
    mem = {"tool": "final", "findings": []}
    llms = {"memory": FakeLLM([mem]), "consistency": FakeLLM([mem]), "assistance": FakeLLM([{"mode": "recall"}])}
    prod = production.build_services(
        store=store, embedder=hashed_embedder(), llms=llms, background_llm=None,
        speech=GrpcSpeech(os.environ.get("SPEECH_ADDR", "localhost:50051")),
        make_backend=shared, telemetry=Telemetry(registry=CollectorRegistry()),
    )  # fmt: skip
    try:
        with TestClient(create_app(prod.services)) as client, client.websocket_connect(f"/ws/{conv}") as raw:
            fr = Frames(raw)
            assert fr.wait(is_type("status"), 10)["state"] == "recording"
            assert wait_for(lambda: client.app.state.conversations[conv].diarizer.backend is not None, 30)
            speak(raw)
            fr.wait(lambda m: m["type"] == "transcript" and "thursday" in m["turn"]["text"].lower(), 90)
            sent = [m["turn"] for m in fr.log if m["type"] == "transcript"]
            fr.wait(lambda m: m["type"] == "turn_update", 180)  # pyannote on the CPU names the speaker
            assert store.turns(conv, 99)[0].speaker.startswith("speaker_")
            assert sent[0]["speaker"] == "unknown"
            raw.send_json({"type": "delete"})
            fr.wait(lambda m: m["type"] == "status" and m["state"] == "deleted", 30)
    finally:
        store.delete(conv)


@pytest.mark.real
def test_press_end_to_end_returns_a_cited_card():
    """audio -> transcript -> press -> card with everything real and local: the speech service, Postgres,
    Ollama (qwen3.5:4b-q4_K_M and nomic-embed-text), pyannote on the CPU. Needs a GPU turn; see docs/run-everything.md."""
    conv = f"it-{uuid.uuid4().hex[:8]}"
    app = production.create_production_app()
    with TestClient(app) as client, client.websocket_connect(f"/ws/{conv}") as raw:
        fr = Frames(raw)
        assert fr.wait(is_type("status"), 10)["state"] == "recording"
        speak(raw)
        fr.wait(lambda m: m["type"] == "transcript" and "thursday" in m["turn"]["text"].lower(), 90)
        turns = {m["turn"]["id"]: m["turn"] for m in fr.log if m["type"] == "transcript"}
        last = max(t["seq"] for t in turns.values())
        raw.send_json(press(conv=conv, as_of=last, question="When is the budget review?"))
        card = fr.wait(lambda m: m["type"] in ("card", "closed") or (m["type"] == "status" and m["state"] == "error"), 120)
        assert card["type"] == "card", card
        jsonschema.validate(card, SCHEMA["server_to_client"])
        body = card["card"]
        assert body["status"] in ("answered", "partial") and body["findings"]
        for f in body["findings"]:
            for e in f["evidence"]:
                assert e["turn_id"] in turns and e["quote"].lower() in turns[e["turn_id"]]["text"].lower()
        assert "thursday" in json.dumps(body).lower()
        raw.send_json({"type": "delete"})
        fr.wait(lambda m: m["type"] == "status" and m["state"] == "deleted", 30)
