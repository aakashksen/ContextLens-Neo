"""The streaming backend, with a fake speech service, FakeLLM and in-memory stores: no GPU, no network."""

import asyncio
import base64
import builtins
import os
import threading
import time

import pytest
from fastapi.testclient import TestClient

from contextlens.agents.single import SingleAgent
from contextlens.agents.tools import Tools
from contextlens.api.app import create_app
from contextlens.api.stream import Services
from contextlens.llm.client import FakeLLM
from contextlens.memory.extract import MemoryExtractor, MemoryStore
from contextlens.memory.store import InMemoryTranscriptStore
from contextlens.memory.summary import RollingSummary
from contextlens.retrieval.hybrid import Retriever
from contextlens.schemas import Card, EvidenceRef, Finding, Turn
from contextlens.speech.client import (
    GrpcSpeech,
    Segment,
    SpeechUnavailable,
    decode_segment,
    encode_chunk,
)
from contextlens.speech.diarize import Diarizer

PCM = base64.b64encode(b"\x01\x00" * 160).decode()
AUDIO = {"type": "audio", "data": PCM}
SCRIPT = ["The deadline is Monday", "ok", "We moved it to Wednesday", "Who sends the sheet?"]


class NoEmbed:
    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeStream:
    def __init__(self, speech):
        self.speech, self.queue = speech, asyncio.Queue()

    def send(self, pcm):
        if self.speech.fail_send:
            raise SpeechUnavailable("backlog")
        self.speech.received.append(pcm)
        text = SCRIPT[self.speech.n % len(SCRIPT)]
        i, self.speech.n = self.speech.n, self.speech.n + 1
        self.queue.put_nowait(Segment(text=text, start_ms=i * 1000, end_ms=i * 1000 + 900))

    async def segments(self):
        while True:
            seg = await self.queue.get()
            while self.speech.gate is not None and not self.speech.gate.is_set():
                await asyncio.sleep(0.01)
            yield seg

    async def close(self):
        self.speech.closed += 1


class FakeSpeech:
    def __init__(self, mode="up"):
        self.mode, self.n, self.closed, self.received = mode, 0, 0, []
        self.gate: threading.Event | None = None
        self.fail_send = False
        self.opens = 0

    async def open(self):
        self.opens += 1
        if self.mode == "down":
            raise SpeechUnavailable("connection refused")
        if self.mode == "stalled":
            await asyncio.sleep(3600)
        return FakeStream(self)


def turn(seq, text, speaker="Ana", conv="c1"):
    return Turn(id=f"{conv}-t{seq}", conversation_id=conv, seq=seq, speaker=speaker, text=text,
                start_ms=seq * 1000, end_ms=seq * 1000 + 900, source="synthetic")  # fmt: skip


def card_for(turn_id, quote):
    f = Finding(agent="single", kind="recall", statement="s", confidence=0.9,
                evidence=[EvidenceRef(turn_id=turn_id, quote=quote, speaker="unknown", start_ms=0)])  # fmt: skip
    return Card(mode="recall", status="answered", findings=[f])


class Recorder:
    """An assistant that records what it could see and can be held while turns arrive."""

    def __init__(self):
        self.seen: list[tuple[int, list[int]]] = []
        self.hold: threading.Event | None = None
        self.started = threading.Event()

    def __call__(self, conv_id, as_of):
        rec = self

        class A:
            def answer(self, req, ctx):
                t = ctx.store.turns(conv_id, as_of)[-1]
                rec.started.set()
                if rec.hold is not None:
                    assert rec.hold.wait(10)
                rec.seen.append((ctx.as_of_seq, [t.seq for t in ctx.store.turns(conv_id, 10**9)]))
                return card_for(t.id, t.text)

        return A()


def build(speech=None, assistant=None, seed=(), extractor=None, summary_llm=None, backend=None, **kw):
    store = InMemoryTranscriptStore()
    for t in seed:
        store.add(t)
    retriever, memory = Retriever(store, NoEmbed()), MemoryStore()
    svc = Services(
        store=store, retriever=retriever, memory=memory, speech=speech or FakeSpeech(),
        make_assistant=assistant or Recorder(),
        extractor=MemoryExtractor(extractor) if extractor else None,
        make_summary=(lambda: RollingSummary(summary_llm)) if summary_llm else None,
        make_backend=(lambda: backend) if backend else None,
        speech_timeout_s=0.3, **kw,
    )  # fmt: skip
    return TestClient(create_app(svc)), svc


def press(conv="c1", as_of=0, **kw):
    return {"type": "press", "request": {"conversation_id": conv, "as_of_seq": as_of, **kw}}


def wait_for(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def hear(ws, n):
    """Send n audio messages and read the n transcripts."""
    out = []
    for _ in range(n):
        ws.send_json(AUDIO)
        msg = ws.receive_json()
        assert msg["type"] == "transcript", msg
        out.append(msg["turn"])
    return out


def test_press_returns_card():
    store = InMemoryTranscriptStore()
    memory = MemoryStore()
    retriever = Retriever(store, NoEmbed())
    quote = SCRIPT[0]
    llm = FakeLLM([{"tool": "final", "card": card_for("c1-t0", quote).model_dump()}])

    def make_assistant(conv, as_of):
        return SingleAgent(llm, Tools(store, retriever, memory, as_of, conv), "v1")

    svc = Services(store=store, retriever=retriever, memory=memory, make_assistant=make_assistant,
                   speech=FakeSpeech())  # fmt: skip
    with TestClient(create_app(svc)).websocket_connect("/ws/c1") as ws:
        assert ws.receive_json() == {"type": "status", "state": "recording", "detail": None}
        turns = hear(ws, 2)
        assert [t["text"] for t in turns] == SCRIPT[:2]
        assert turns[0]["source"] == "asr" and turns[0]["speaker_confidence"] < 0.5
        ws.send_json(press(as_of=1, question="What was the deadline?"))
        msg = ws.receive_json()
        assert msg["type"] == "card" and msg["card"]["status"] == "answered"
        assert msg["card"]["findings"][0]["evidence"][0]["turn_id"] == "c1-t0"


def test_press_during_ingest_uses_frozen_seq():
    rec = Recorder()
    rec.hold = threading.Event()
    client, svc = build(assistant=rec)
    with client.websocket_connect("/ws/c1") as ws:
        ws.receive_json()
        hear(ws, 2)
        ws.send_json(press(as_of=1))
        assert rec.started.wait(5)
        late = hear(ws, 2)  # turns 2 and 3 arrive while the press is being answered
        assert [t["seq"] for t in late] == [2, 3]
        rec.hold.set()
        assert ws.receive_json()["type"] == "card"
    assert rec.seen == [(1, [0, 1])]
    assert [t.seq for t in svc.store.turns("c1", 99)] == [0, 1, 2, 3]


def test_press_rules_match_the_contract():
    client, _ = build(seed=[turn(0, "hello")])
    with client.websocket_connect("/ws/c1") as ws:
        ws.receive_json()
        for bad in (press(conv="other"), press(as_of=-1), press(as_of=5)):
            ws.send_json(bad)
            msg = ws.receive_json()
            assert msg["type"] == "status" and msg["state"] == "error"
        ws.send_json(press(as_of=0))
        assert ws.receive_json()["type"] == "card"


def test_pause_stops_ingest():
    speech = FakeSpeech()
    client, svc = build(speech=speech, seed=[turn(0, "stored before", conv="c1")])
    with client.websocket_connect("/ws/c1") as ws:
        ws.receive_json()
        assert hear(ws, 1)[0]["seq"] == 1
        speech.gate = threading.Event()  # audio goes out but its text stays in flight
        ws.send_json(AUDIO)
        assert wait_for(lambda: len(speech.received) == 2)
        ws.send_json({"type": "pause"})
        assert ws.receive_json() == {"type": "status", "state": "paused", "detail": None}
        ws.send_json(AUDIO)  # ignored while paused
        time.sleep(0.2)
        assert len(speech.received) == 2
        ws.send_json(press(as_of=1))  # still answered from what is stored
        assert ws.receive_json()["type"] == "card"
        ws.send_json({"type": "resume"})
        assert ws.receive_json()["state"] == "recording"
        speech.gate.set()  # only now does the pre-pause text come back, after resume
        time.sleep(0.3)
        assert [t.seq for t in svc.store.turns("c1", 99)] == [0, 1]
        speech.gate = None
        assert hear(ws, 1)[0]["seq"] == 2  # the in-flight text never came back
    assert len(svc.store.turns("c1", 99)) == 3


def test_delete_removes_everything():
    ext = FakeLLM([{"items": [{"kind": "fact", "subject": "deadline", "statement": "Monday",
                               "source_turn_ids": ["c1-t0"]}]}])  # fmt: skip
    summ = FakeLLM(["The deadline is Monday."])
    client, svc = build(extractor=ext, summary_llm=summ)
    with client.websocket_connect("/ws/c1") as ws:
        ws.receive_json()
        hear(ws, 1)
        assert wait_for(lambda: svc.memory.items("c1", 99))
        svc.retriever.search("c1", "deadline", 0)  # fills the vector cache
        ws.send_json({"type": "delete"})
        assert ws.receive_json() == {"type": "status", "state": "deleted", "detail": None}
        with pytest.raises(Exception):
            ws.receive_json()  # the server closed
    assert svc.store.turns("c1", 99) == [] and svc.memory.items("c1", 99) == []
    assert svc.retriever._cache == {}
    with client.websocket_connect("/ws/c1") as ws:  # starts again at seq 0 with no summary
        ws.receive_json()
        assert hear(ws, 1)[0]["seq"] == 0


def test_delete_drops_late_work():
    speech = FakeSpeech()
    speech.gate = threading.Event()
    release = threading.Event()

    class SlowExtractorLLM(FakeLLM):
        def _raw(self, *a, **k):
            assert release.wait(10)
            return super()._raw(*a, **k)

    ext = SlowExtractorLLM([{"items": [{"kind": "fact", "subject": "deadline", "statement": "Monday",
                                        "source_turn_ids": ["c1-t0"]}]}])  # fmt: skip
    client, svc = build(speech=speech, extractor=ext, seed=[turn(0, "seed")])
    with client.websocket_connect("/ws/c1") as a, client.websocket_connect("/ws/c1") as b:
        a.receive_json(), b.receive_json()
        speech.gate = None
        a.send_json(AUDIO)
        assert a.receive_json()["type"] == "transcript"  # turn 1; memory update now blocked in the thread
        speech.gate = threading.Event()
        a.send_json(AUDIO)  # its text is held in flight
        assert wait_for(lambda: len(speech.received) == 2)
        b.send_json({"type": "delete"})
        assert b.receive_json()["state"] == "deleted"
        speech.gate.set()
        release.set()
        time.sleep(0.4)
        a.send_json(press(as_of=0))
        msg = a.receive_json()  # nothing late arrives before this
        assert msg["type"] == "status" and msg["state"] == "error"
    assert svc.store.turns("c1", 99) == [] and svc.memory.items("c1", 99) == []


def test_card_answered_after_a_delete_is_not_sent():
    rec = Recorder()
    rec.hold = threading.Event()
    client, svc = build(assistant=rec, seed=[turn(0, "hello there")])
    with client.websocket_connect("/ws/c1") as a, client.websocket_connect("/ws/c1") as b:
        a.receive_json(), b.receive_json()
        a.send_json(press(as_of=0))
        assert rec.started.wait(5)
        b.send_json({"type": "delete"})
        assert b.receive_json()["state"] == "deleted"
        rec.hold.set()
        time.sleep(0.3)
        a.send_json(press(as_of=0))
        msg = a.receive_json()  # the old card never arrives
        assert msg["type"] == "status" and "past the last turn" in msg["detail"]


@pytest.mark.parametrize("mode", ["down", "stalled"])
def test_speech_service_down_sends_status_and_keeps_text_clients_working(mode):
    client, svc = build(speech=FakeSpeech(mode), seed=[turn(0, "hello there")])
    t0 = time.monotonic()
    with client.websocket_connect("/ws/c1") as ws:
        ws.receive_json()
        ws.send_json(AUDIO)
        msg = ws.receive_json()
        assert msg["type"] == "status" and msg["state"] == "error"
        assert "speech service unavailable" in msg["detail"]
        assert time.monotonic() - t0 < 3
        ws.send_json(AUDIO)  # no second report while the service stays down
        ws.send_json(press(as_of=0))
        assert ws.receive_json()["type"] == "card"
    with client.websocket_connect("/ws/other") as ws:  # another conversation is unaffected
        assert ws.receive_json()["state"] == "recording"
        ws.send_json({"type": "pause"})
        assert ws.receive_json()["state"] == "paused"
    assert svc.speech.opens == 1


def test_stream_failure_mid_conversation_reports_once_and_recovers():
    speech = FakeSpeech()
    client, svc = build(speech=speech, speech_retry_s=0.0)
    with client.websocket_connect("/ws/c1") as ws:
        ws.receive_json()
        hear(ws, 1)
        speech.fail_send = True
        ws.send_json(AUDIO)
        assert ws.receive_json()["state"] == "error"
        speech.fail_send = False
        assert hear(ws, 1)[0]["seq"] == 1


def test_no_audio_files_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    marker = b"\x7b\x01" * 400
    writes = []
    real_open, real_os_open = builtins.open, os.open

    def spy_open(file, mode="r", *a, **k):
        if any(c in str(mode) for c in "wax+"):
            writes.append(str(file))
        return real_open(file, mode, *a, **k)

    def spy_os_open(path, flags, *a, **k):
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
            writes.append(str(path))
        return real_os_open(path, flags, *a, **k)

    class Backend:
        def label(self, pcm, start_ms, end_ms):
            return "S1", 0.9

    client, svc = build(backend=Backend())
    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(os, "open", spy_os_open)
    with client.websocket_connect("/ws/c1") as ws:
        ws.receive_json()
        for _ in range(3):
            ws.send_json({"type": "audio", "data": base64.b64encode(marker).decode()})
            ws.receive_json()
        ws.send_json({"type": "pause"})
        ws.receive_json()
    monkeypatch.undo()
    assert writes == []
    assert list(tmp_path.rglob("*")) == []
    assert b"".join(svc.speech.received) == marker * 3  # the service got the audio, in memory only


def test_diarizer_assigns_speakers_and_flags_low_confidence():
    seen = []

    class Backend:
        def label(self, pcm, start_ms, end_ms):
            seen.append((len(pcm), start_ms, end_ms))
            return ("A", 0.9) if start_ms == 0 else ("B", 0.3)

    d = Diarizer("c1", Backend())
    d.new_stream()
    d.feed(bytes(32000))  # one second
    t = d.assign([Segment(text=" hi ", start_ms=0, end_ms=500), Segment(text="yo", start_ms=500, end_ms=1000)])
    assert [(x.seq, x.id, x.speaker, x.speaker_confidence, x.text, x.source) for x in t] == [
        (0, "c1-t0", "A", 0.9, "hi", "asr"),
        (1, "c1-t1", "B", 0.3, "yo", "asr"),
    ]
    assert seen == [(16000, 0, 500), (16000, 500, 1000)]
    d.new_stream()  # a new stream restarts its clock; turn times keep running
    assert d.assign([Segment(text="again", start_ms=0, end_ms=100)])[0].start_ms == 1000
    assert Diarizer("c1").assign([Segment(text="x", start_ms=0, end_ms=1)])[0].speaker_confidence == 0.0


def test_wire_codec_matches_the_proto():
    assert encode_chunk(b"ab") == b"\x0a\x02ab"
    seg, final = decode_segment(b"\x0a\x02hi\x10\x05\x18\xe8\x07\x20\x01")
    assert (seg.text, seg.start_ms, seg.end_ms, final) == ("hi", 5, 1000, True)


def test_grpc_client_against_an_in_process_server():
    import grpc

    async def scenario():
        def pass_through(b):
            return b

        handler = grpc.method_handlers_generic_handler(
            "contextlens.speech.Speech",
            {"Transcribe": grpc.stream_stream_rpc_method_handler(
                _seg_stream, request_deserializer=pass_through,
                response_serializer=pass_through)},
        )  # fmt: skip
        server = grpc.aio.server()
        server.add_generic_rpc_handlers((handler,))
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        try:
            stream = await GrpcSpeech(f"127.0.0.1:{port}", 2.0).open()
            stream.send(b"\x00\x00")
            seg = await asyncio.wait_for(stream.segments().__anext__(), 5)
            await stream.close()
            return seg
        finally:
            await server.stop(0)

    async def _seg_stream(it, ctx):
        async for _ in it:
            yield b"\x0a\x03abc\x10\x01\x18\x02\x20\x01"

    seg = asyncio.run(scenario())
    assert (seg.text, seg.start_ms, seg.end_ms) == ("abc", 1, 2)


def test_grpc_client_reports_a_service_that_is_not_there():
    async def scenario():
        with pytest.raises(SpeechUnavailable):
            await GrpcSpeech("127.0.0.1:1", 0.5).open()

    asyncio.run(scenario())


SECOND = base64.b64encode(bytes(32000)).decode()  # one second of silence
ONE_SECOND = {"type": "audio", "data": SECOND}


class Spans:
    """A stand-in for a diarization pipeline: second i of the window is spoken by X when i is even, else Y."""

    def __init__(self):
        self.lengths, self.hold, self.entered = [], None, threading.Event()
        self.only_first = False

    def spans(self, pcm):
        first = not self.lengths
        self.lengths.append(len(pcm))
        self.entered.set()
        if self.hold is not None:
            assert self.hold.wait(10)
        if self.only_first and not first:
            return []
        return [(i * 1000, i * 1000 + 900, "X" if i % 2 == 0 else "Y") for i in range(len(pcm) // 32000)]


def speakers(svc, conv="c1"):
    return [(t.speaker, t.speaker_confidence) for t in svc.store.turns(conv, 10**9)]


def test_speakers_are_assigned_in_the_background_and_sent_as_turn_updates():
    spans = Spans()
    client, svc = build(backend=spans)
    with client.websocket_connect("/ws/c1") as ws:
        ws.receive_json()
        for _ in range(2):
            ws.send_json(ONE_SECOND)
        frames = [ws.receive_json() for _ in range(4)]  # two transcripts and, once the pipeline has run, two updates
        sent = [f["turn"] for f in frames if f["type"] == "transcript"]
        assert [(t["seq"], t["speaker"], t["speaker_confidence"]) for t in sent] == [(0, "unknown", 0.0), (1, "unknown", 0.0)]
        updates = {f["turn"]["seq"]: f["turn"]["speaker"] for f in frames if f["type"] == "turn_update"}
        assert updates == {0: "speaker_1", 1: "speaker_2"}
        assert wait_for(lambda: speakers(svc) == [("speaker_1", 1.0), ("speaker_2", 1.0)])
        ws.send_json(ONE_SECOND)
        msgs = [ws.receive_json(), ws.receive_json()]
        assert sorted((m["type"], m["turn"]["seq"]) for m in msgs) == [("transcript", 2), ("turn_update", 2)]  # no resends
        assert wait_for(lambda: [s for s, _ in speakers(svc)] == ["speaker_1", "speaker_2", "speaker_1"])


def hear_audio(ws, n):
    out = []
    for _ in range(n):
        ws.send_json(ONE_SECOND)
        out.append(ws.receive_json()["turn"])
    return out


def test_delete_during_background_diarization_brings_nothing_back():
    spans = Spans()
    spans.hold, spans.only_first = threading.Event(), True
    client, svc = build(backend=spans)
    with client.websocket_connect("/ws/c1") as ws:
        ws.receive_json()
        hear_audio(ws, 1)
        assert spans.entered.wait(5)
        ws.send_json({"type": "delete"})
        assert ws.receive_json()["state"] == "deleted"
    with client.websocket_connect("/ws/c1") as ws:
        ws.receive_json()
        hear_audio(ws, 1)  # the same turn id as the deleted conversation's first turn
        spans.hold.set()  # the pre-delete run finishes now and must change nothing
        assert wait_for(lambda: len(spans.lengths) >= 2)
        time.sleep(0.3)
        assert speakers(svc) == [("unknown", 0.0)]
        assert max(spans.lengths) <= 32000  # no audio from before the delete


def test_diarization_uses_only_the_retained_window_and_leaves_older_turns_alone():
    d = Diarizer("c1", Spans())
    d.new_stream()
    d.feed(bytes(32000 * 130))  # 130 s; two minutes are kept
    pcm, start = d.window()
    assert (len(pcm), start) == (32000 * 120, 10_000)
    old = Turn(id="c1-t0", conversation_id="c1", seq=0, speaker="unknown", speaker_confidence=0.0, text="a",
               start_ms=5_000, end_ms=6_000, source="asr")  # fmt: skip
    new = old.model_copy(update={"id": "c1-t1", "seq": 1, "start_ms": 20_000, "end_ms": 21_000})
    got = d.speaker_updates([(0, 2_000, "X"), (10_000, 12_000, "Y")], start, start + 120_000, [old, new])  # times in the window
    assert got == [(1, "speaker_1", 1.0)]  # the turn from before the window is not touched


def test_speaker_names_stay_the_same_between_runs():
    d = Diarizer("c1", Spans())
    t0 = Turn(id="c1-t0", conversation_id="c1", seq=0, speaker="unknown", speaker_confidence=0.0, text="a",
              start_ms=0, end_ms=1000, source="asr")  # fmt: skip
    t1 = t0.model_copy(update={"id": "c1-t1", "seq": 1, "start_ms": 2000, "end_ms": 3000})
    first = d.speaker_updates([(0, 1000, "P"), (2000, 3000, "Q")], 0, 3000, [t0, t1])
    assert [(s, n) for s, n, _ in first] == [(0, "speaker_1"), (1, "speaker_2")]
    t0, t1 = (t.model_copy(update={"speaker": n, "speaker_confidence": c}) for t, (_, n, c) in zip((t0, t1), first))
    t2 = t0.model_copy(update={"id": "c1-t2", "seq": 2, "start_ms": 4000, "end_ms": 5000})
    # the pipeline calls the same people by other names this time, and a third person speaks
    second = d.speaker_updates([(0, 1000, "B"), (2000, 3000, "A"), (4000, 5000, "C")], 0, 5000, [t0, t1, t2])
    assert [(s, n) for s, n, _ in second] == [(0, "speaker_1"), (1, "speaker_2"), (2, "speaker_3")]


def test_a_turn_without_a_clear_speaker_stays_unknown_and_low_confidence_is_kept():
    d = Diarizer("c1", Spans())
    t = Turn(id="c1-t0", conversation_id="c1", seq=0, speaker="unknown", speaker_confidence=0.0, text="a",
             start_ms=0, end_ms=1000, source="asr")  # fmt: skip
    assert d.speaker_updates([(5000, 6000, "P")], 0, 1000, [t]) == []  # nobody spoke during it
    [(seq, name, conf)] = d.speaker_updates([(0, 400, "P"), (400, 1000, "Q")], 0, 1000, [t])
    assert (seq, name, conf) == (0, "speaker_1", pytest.approx(0.6))  # the larger share wins, with that share


def test_hf_token_is_read_from_the_environment_or_dotenv_and_never_shown(tmp_path, monkeypatch):
    from contextlens.speech.diarize import hf_token

    secret = "hf_" + "x" * 30
    monkeypatch.delenv("HF_TOKEN", raising=False)
    (tmp_path / ".env").write_text(f"OTHER=1\nHF_TOKEN={secret}\n")
    assert hf_token(tmp_path / ".env") == secret
    monkeypatch.setenv("HF_TOKEN", "from-env")
    assert hf_token(tmp_path / ".env") == "from-env"
    monkeypatch.delenv("HF_TOKEN")
    with pytest.raises(RuntimeError) as e:
        hf_token(tmp_path / "missing.env")
    assert secret not in str(e.value) and "HF_TOKEN" in str(e.value)


def test_a_backend_that_fails_to_load_leaves_the_connection_and_turns_unknown(caplog):
    calls = []

    def broken():
        calls.append(1)
        raise RuntimeError("token hf_secret123 refused")

    client, svc = build()
    svc.make_backend = broken
    with caplog.at_level("WARNING"), client.websocket_connect("/ws/c1") as ws:
        assert ws.receive_json()["state"] == "recording"
        for t in hear_audio(ws, 2):
            assert (t["speaker"], t["speaker_confidence"]) == ("unknown", 0.0)
        assert wait_for(lambda: bool(calls))
        time.sleep(0.2)
    assert calls == [1]  # one attempt, no retry loop
    assert speakers(svc) == [("unknown", 0.0), ("unknown", 0.0)]
    assert "RuntimeError" in caplog.text and "hf_secret123" not in caplog.text


def test_a_slow_backend_load_does_not_hold_up_the_transcript():
    gate = threading.Event()
    spans = Spans()

    def slow():
        assert gate.wait(10)
        return spans

    client, svc = build()
    svc.make_backend = slow
    with client.websocket_connect("/ws/c1") as ws:
        assert ws.receive_json()["state"] == "recording"  # not waiting for the load
        assert hear_audio(ws, 1)[0]["speaker"] == "unknown"
        gate.set()
        time.sleep(0.3)
        hear_audio(ws, 1)
        assert wait_for(lambda: [s for s, _ in speakers(svc)][-1] != "unknown")
        assert spans.lengths[0] <= 32000  # audio heard before the load is not in the window
