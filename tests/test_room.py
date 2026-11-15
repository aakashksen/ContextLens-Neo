"""The meeting room: timing rules, replay, trace, server endpoints and the live session, with fakes only.
No Ollama, no speech service, no GPU, no websocket to a real server."""
import asyncio
import json
import sys
import time
import wave
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from contextlens.api.contract import TranscriptMessage, TurnUpdateMessage
from contextlens.memory.store import InMemoryTranscriptStore
from contextlens.replay.runner import Context, DroppedFinding
from contextlens.room import live as live_mod
from contextlens.room import meetings as M
from contextlens.room import rig as rig_mod
from contextlens.room import server as S
from contextlens.room.live import EventLog, LiveRig, LiveSession
from contextlens.room.replay import ReplayAssistant, as_of_seq, spoken_by
from contextlens.room.trace import build_trace
from contextlens.schemas import ButtonRequest, Card, EvidenceRef, Finding, MemoryItem, Turn

RATE = 16000


def turn(seq, speaker, text, start, end, conv="ES2003a"):
    return Turn(id=f"{conv}:t{seq}", conversation_id=conv, seq=seq, speaker=speaker, text=text, start_ms=start, end_ms=end, source="dataset")


TURNS = [
    turn(0, "A", "We should price the remote at twenty five euros", 0, 3000),
    turn(1, "B", "Production costs cannot exceed twelve euros", 3500, 6000),
    turn(2, "A", "Actually the price should be thirty euros", 7000, 9500),
    turn(3, "C", "Lunch is at noon", 10000, 12000),
]


def write_wav(path: Path, seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(b"\x01\x00" * int(RATE * seconds))


@pytest.fixture
def raw(tmp_path):
    write_wav(tmp_path / "ami-audio" / "ES2003a.wav", 4.0)
    write_wav(tmp_path / "ami-audio" / "notes.wav", 1.0)  # not a meeting id
    (tmp_path / "ami-audio" / "ES2010d.wav").write_bytes(b"not a wav")  # unreadable
    return tmp_path


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(S, "timed_turns", lambda raw, mid: list(TURNS))
    monkeypatch.setattr(S, "summary_items", lambda raw, mid: [("decision", "The remote will sell for thirty euros"), ("commitment", "Nobody mentioned this at all")])


# ---- meetings ----

def test_find_meetings_skips_bad_files_and_finds_video(raw):
    (raw / "ami-video").mkdir()
    (raw / "ami-video" / "ES2003a.mp4").write_bytes(b"x")
    found = M.find_meetings(raw)
    assert list(found) == ["ES2003a"]
    assert found["ES2003a"].duration_ms == 4000
    assert found["ES2003a"].public() == {"id": "ES2003a", "duration_ms": 4000, "video": True}


def test_wav_chunks_start_and_size(raw):
    wav = raw / "ami-audio" / "ES2003a.wav"
    chunks = list(M.wav_chunks(wav, 1000, 500))
    assert len(chunks[0]) == 500 * 32 and sum(map(len, chunks)) == 3000 * 32
    assert list(M.wav_chunks(wav, 99_000, 500)) == []


@pytest.mark.parametrize("header,expected", [
    (None, None), ("bytes=0-9", (0, 9)), ("bytes=5-", (5, 99)), ("bytes=-10", (90, 99)), ("bytes=90-500", (90, 99)),
    ("bytes=1-2,4-5", None), ("garbage", None), ("bytes=a-b", None),
])
def test_parse_range(header, expected):
    assert M.parse_range(header, 100) == expected


def test_parse_range_outside_file():
    with pytest.raises(ValueError):
        M.parse_range("bytes=100-200", 100)


def test_valid_id_blocks_paths():
    assert M.valid_id("ES2003a") and M.valid_id("TS3006b")
    assert not M.valid_id("../x") and not M.valid_id("ES2003a.wav") and not M.valid_id("")


def test_timed_turns_and_summary_from_ami_xml(tmp_path):
    ami = tmp_path / "ami"
    (ami / "words").mkdir(parents=True)
    (ami / "segments").mkdir()
    (ami / "abstractive").mkdir()
    ns = 'xmlns:nite="http://nite.sourceforge.net/"'
    (ami / "words" / "ES2003a.A.words.xml").write_text(
        f'<nite:root {ns}><w nite:id="w0" starttime="1.0" endtime="1.5">Hello</w><w nite:id="w1" starttime="1.5" endtime="2.0">there</w>'
        '<w nite:id="w2" starttime="2.0" endtime="2.0" punc="true">.</w></nite:root>', encoding="utf-8")
    (ami / "segments" / "ES2003a.A.segments.xml").write_text(
        f'<nite:root {ns}><segment nite:id="s1"><nite:child href="ES2003a.A.words.xml#id(w0)..id(w2)"/></segment></nite:root>', encoding="utf-8")
    (ami / "abstractive" / "ES2003a.abssumm.xml").write_text(
        f'<nite:root {ns}><abstract><sentence>Ignored</sentence></abstract><decisions><sentence>The remote will sell.</sentence></decisions>'
        '<actions><sentence>Dave will draw it.</sentence></actions></nite:root>', encoding="utf-8")
    turns = M.timed_turns(tmp_path, "ES2003a")
    assert [(t.speaker, t.text, t.start_ms, t.end_ms) for t in turns] == [("A", "Hello there.", 1000, 2000)]
    assert M.timed_turns(tmp_path, "ES9999a") == []
    assert M.summary_items(tmp_path, "ES2003a") == [("decision", "The remote will sell."), ("commitment", "Dave will draw it.")]
    assert M.summary_items(tmp_path, "ES9999a") == []


def test_align_items_places_at_best_turn_and_drops_unmatched():
    items = [("decision", "The remote will sell for thirty euros"), ("commitment", "Nobody mentioned this at all")]
    placed = M.align_items("ES2003a", items, TURNS)
    assert len(placed) == 1
    assert placed[0].as_of_seq == 2 and placed[0].source_turn_ids == ["ES2003a:t2"] and placed[0].kind == "decision"


# ---- timing rule ----

def test_spoken_by_excludes_a_turn_still_being_spoken():
    assert [t.seq for t in spoken_by(TURNS, 6000)] == [0, 1]
    assert [t.seq for t in spoken_by(TURNS, 5999)] == [0]
    assert as_of_seq(TURNS, 100) == -1
    assert as_of_seq(TURNS, 9500) == 2


def test_replay_assistant_uses_only_the_store_and_abstains_when_nothing_matches():
    store = InMemoryTranscriptStore()
    for t in TURNS[:2]:
        store.add(t)
    ctx = Context(store=store, as_of_seq=1)
    card = ReplayAssistant().answer(ButtonRequest(conversation_id="ES2003a", as_of_seq=1, question="what price for the remote"), ctx)
    assert card.status == "answered"
    assert {e.turn_id for f in card.findings for e in f.evidence} <= {"ES2003a:t0", "ES2003a:t1"}
    assert ctx.stage_ms["search_hits"] >= 1
    nothing = ReplayAssistant().answer(ButtonRequest(conversation_id="ES2003a", as_of_seq=1, question="zebra giraffe"), Context(store=store, as_of_seq=1))
    assert nothing.status == "abstained" and nothing.missing
    empty = ReplayAssistant().answer(ButtonRequest(conversation_id="x", as_of_seq=0), Context(store=InMemoryTranscriptStore(), as_of_seq=0))
    assert empty.status == "abstained"


# ---- trace ----

def finding(agent, statement, tid, label=None):
    return Finding(agent=agent, kind="recall", statement=statement, confidence=0.9, label=label,
                   evidence=[EvidenceRef(turn_id=tid, quote="q", speaker="A", start_ms=0)])


def test_trace_routed_with_conflict_and_drops():
    card = Card(mode="recall", status="answered", latency_ms=4200, tokens=900, missing=["note one"],
                findings=[finding("memory", "price 25", "t0"), finding("consistency", "price moved", "t2", "update")])
    stage = {"ran_memory": 1, "memory_ms": 1500, "memory_tokens": 300, "ran_consistency": 1, "consistency_ms": 1800,
             "search_ms": 90, "search_hits": 5, "checker_ms": 400, "checker_update": 1}
    drops = [DroppedFinding(reason="number", statement="price 99", turn_ids=["t1"])]
    tr = build_trace("team_routed", card, stage, drops, speech_ms=700, speech_note="real speech")
    by = {s["id"]: s for s in tr["stages"]}
    assert by["speech"]["ms"] == 700 and by["search"]["ms"] == 90 and by["search"]["note"] == "5 earlier turns"
    assert by["checker"] == {"id": "checker", "label": "Checker", "ms": 400, "note": "update"}
    assert by["specialists"]["ms"] == 3300 and by["specialists"]["note"] == "memory, consistency"
    assert by["guard"]["note"] == "1 finding(s) dropped"
    ran = {s["name"]: s for s in tr["specialists"]}
    assert ran["memory"]["ran"] and ran["consistency"]["ran"] and not ran["assistance"]["ran"]
    assert ran["memory"]["findings"] == [0] and ran["consistency"]["findings"] == [1]
    assert tr["checker"] == {"ran": True, "label": "update", "failed": False}
    assert tr["drops"][0]["reason"] == "number" and "quotes" in tr["drops"][0]["why"]
    assert tr["abstained"] == {"yes": False, "why": []} and tr["notes"] == ["note one"]


def test_trace_abstained_single_and_unknown_drop_reason():
    card = Card(mode="recall", status="abstained", findings=[], missing=["memory found nothing relevant"])
    tr = build_trace("single_plus", card, {"llm": 2000, "tools": 300, "search_ms": 50, "search_hits": 0, "checker_ms": 10, "checker_failed": 1},
                     [DroppedFinding(reason="weird", statement="s", turn_ids=[])])
    assert tr["abstained"]["yes"] and tr["abstained"]["why"] == ["memory found nothing relevant"]
    assert tr["specialists"][0]["name"] == "single" and tr["specialists"][0]["ms"] == 2300
    assert tr["checker"]["failed"] and tr["checker"]["label"] is None
    assert tr["drops"][0]["why"] == "weird"
    assert {s["id"]: s for s in tr["stages"]}["speech"]["ms"] is None


# ---- replay server ----

@pytest.fixture
def client(raw, patched):
    app = FastAPI()
    S.add_room(app, raw, site=None)
    with TestClient(app) as c:
        yield c


def test_health_meetings_and_replay_payload(client):
    h = client.get("/api/health").json()
    assert h["mode"] == "replay" and h["assistants"] == ["single_plus", "team_routed"] and "CC BY 4.0" in h["attribution"]
    assert [m["id"] for m in client.get("/api/room/meetings").json()["meetings"]] == ["ES2003a"]
    p = client.get("/api/room/meetings/ES2003a/replay").json()
    assert [t["seq"] for t in p["turns"]] == [0, 1, 2, 3] and p["turns"][0]["end_ms"] == 3000
    assert [m["at_ms"] for m in p["memory"]] == [9500] and p["flags"] == []
    assert client.get("/api/room/meetings/ES9999a/replay").status_code == 404
    assert client.get("/api/room/meetings/..%2Fx/replay").status_code == 404


def test_audio_serves_whole_file_and_ranges(client, raw):
    size = (raw / "ami-audio" / "ES2003a.wav").stat().st_size
    full = client.get("/api/room/audio/ES2003a")
    assert full.status_code == 200 and len(full.content) == size and full.headers["accept-ranges"] == "bytes"
    part = client.get("/api/room/audio/ES2003a", headers={"Range": "bytes=10-19"})
    assert part.status_code == 206 and part.content == full.content[10:20]
    assert part.headers["content-range"] == f"bytes 10-19/{size}"
    assert client.get("/api/room/audio/ES2003a", headers={"Range": f"bytes={size + 5}-"}).status_code == 416
    assert client.get("/api/room/audio/ES2010d").status_code == 404
    assert client.get("/api/room/video/ES2003a").status_code == 404


def test_press_uses_only_what_was_said_before(client):
    body = {"meeting": "ES2003a", "assistant": "both", "question": "what price for the remote", "position_ms": 6200}
    r = client.post("/api/room/press", json=body).json()
    assert r["as_of_seq"] == 1 and r["simulated"] is True
    assert [x["assistant"] for x in r["results"]] == ["single_plus", "team_routed"]
    cited = {e["turn_id"] for x in r["results"] for f in x["card"]["findings"] for e in f["evidence"]}
    assert cited and cited <= {"ES2003a:t0", "ES2003a:t1"}  # the turn at 9500 ms does not exist yet
    assert all(x["trace"]["stages"] for x in r["results"])
    later = client.post("/api/room/press", json={**body, "position_ms": 13000}).json()
    assert later["as_of_seq"] == 3


def test_press_before_anything_is_said_and_bad_input(client):
    r = client.post("/api/room/press", json={"meeting": "ES2003a", "position_ms": 100}).json()
    assert r["results"] == [] and r["as_of_seq"] == -1 and r["error"]
    assert client.post("/api/room/press", json={"meeting": "ES2003a", "assistant": "team"}).status_code == 400
    assert client.post("/api/room/press", json={"meeting": "ES2003a", "question": "x" * 301}).status_code == 422
    assert client.post("/api/room/press", json={"meeting": "../x"}).status_code == 404
    assert client.post("/api/room/press", json={"meeting": "ES2003a", "position_ms": -1}).status_code == 422


def test_live_endpoints_are_off_in_replay(client):
    assert client.post("/api/room/live/start", json={"meeting": "ES2003a"}).status_code == 409
    assert client.get("/api/room/live/abc/events").status_code == 409


def test_site_is_served_last(raw, patched, tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    (site / "room.html").write_text("<p>room</p>", encoding="utf-8")
    app = FastAPI()
    S.add_room(app, raw, site=site)
    with TestClient(app) as c:
        assert c.get("/room.html").text == "<p>room</p>" and c.get("/api/health").status_code == 200


# ---- live session ----

class FakeTransport:
    """Stands in for /ws/{conversation_id}: every `every` audio chunks it 'transcribes' one turn into the fake store."""

    def __init__(self, conv, world, every=2):
        self.conv, self.world, self.every = conv, world, every
        self.audio = 0
        self.sent_types = []
        self.queue = asyncio.Queue()
        self.closed = False

    async def send(self, text):
        msg = json.loads(text)
        self.sent_types.append(msg["type"])
        if msg["type"] != "audio":
            return
        self.audio += 1
        if self.audio % self.every == 0:
            n = len(self.world.turns.setdefault(self.conv, []))
            t = turn(n, "SPEAKER_00", f"this is a spoken sentence number {n}", n * 1000, n * 1000 + 800, conv=self.conv)
            self.world.turns[self.conv].append(t)
            await self.queue.put(TranscriptMessage(turn=t).model_dump_json())

    async def messages(self):
        while (m := await self.queue.get()) is not None:
            yield m

    async def close(self):
        self.closed = True
        await self.queue.put(None)


class World:
    def __init__(self):
        self.turns, self.memory, self.transports, self.flag_on, self.answers = {}, {}, [], set(), []

    def rig(self):
        async def connect(conv):
            t = FakeTransport(conv, self)
            self.transports.append(t)
            return t

        def answer(name, conv, as_of, question):
            self.answers.append((name, as_of, question))
            seen = [t for t in self.turns[conv] if t.seq <= as_of]
            f = finding("memory" if name == "team_routed" else "single", f"{name} saw {len(seen)}", seen[-1].id)
            ctx = Context(store=InMemoryTranscriptStore(), as_of_seq=as_of)
            ctx.stage_ms.update({"llm": 100, "tools": 20} if name == "single_plus" else {"ran_memory": 1, "memory_ms": 50})
            return Card(mode="recall", status="answered", findings=[f], latency_ms=170), ctx

        def check(conv, t):
            return {"label": "update", "confidence": 0.8, "earlier_ids": [self.turns[conv][0].id], "checker_ms": 5, "search_ms": 2} if t.seq in self.flag_on else None

        return LiveRig(connect=connect, turns=lambda c: list(self.turns.get(c, [])), memory=lambda c: list(self.memory.get(c, [])),
                       check_flag=check, answer=answer)


async def instant(_s):
    await asyncio.sleep(0)


def make_session(raw, world, **kw):
    m = M.find_meetings(raw)["ES2003a"]
    return LiveSession("s1", m.id, m.wav, m.duration_ms, world.rig(), pace=instant, **kw)


async def settle(cond, tries=200):
    for _ in range(tries):
        if cond():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("did not settle")


def types(s, since=0):
    return [e["type"] for e in s.log.since(since)[0]]


def test_live_session_feeds_audio_and_turns_become_events(raw):
    async def go():
        world = World()
        s = make_session(raw, world)
        await s.play(0)
        await settle(lambda: "ended" in types(s))
        evs = s.log.since(0)[0]
        turns = [e for e in evs if e["type"] == "turn"]
        assert world.transports[0].audio == 8 and len(turns) == 4  # 4 s of audio in 0.5 s chunks
        assert turns[0]["turn"]["speaker"] == "SPEAKER_00" and turns[0]["speech_lag_ms"] is not None and s.last_lag_ms is not None
        assert [e["n"] for e in evs] == list(range(len(evs)))
        await s.close()
        assert world.transports[0].sent_types[-1] == "delete" and world.transports[0].closed
    asyncio.run(go())


def test_live_turn_update_and_status_are_forwarded(raw):
    async def go():
        world = World()
        s = make_session(raw, world)
        await s._open(0)
        t = turn(0, "SPEAKER_01", "a sentence", 0, 500, conv=s._conv)
        await world.transports[0].queue.put(TurnUpdateMessage(turn=t).model_dump_json())
        await world.transports[0].queue.put(json.dumps({"type": "status", "state": "error", "detail": "speech service unavailable"}))
        await world.transports[0].queue.put("not json")  # ignored
        await settle(lambda: "status" in types(s))
        evs = s.log.since(0)[0]
        assert [e["type"] for e in evs if e["type"] != "reset"] == ["turn_update", "status"]
        assert evs[-1]["detail"] == "speech service unavailable"
        await s.close()
    asyncio.run(go())


def test_live_poll_emits_memory_once_and_flags_new_turns(raw):
    async def go():
        world = World()
        s = make_session(raw, world)
        await s.play(0)
        await settle(lambda: "ended" in types(s))
        conv = s._conv
        world.memory[conv] = [MemoryItem(id="m1", conversation_id=conv, kind="decision", subject="price", statement="30 euros",
                                         speaker="A", source_turn_ids=[world.turns[conv][1].id], as_of_seq=1)]
        world.flag_on = {3}
        for _ in range(3):
            await s.poll()
            await asyncio.sleep(0.02)
        evs = s.log.since(0)[0]
        assert [e["item"]["id"] for e in evs if e["type"] == "memory"] == ["m1"]
        flags = [e["flag"] for e in evs if e["type"] == "flag"]
        assert [f["turn_id"] for f in flags] == [world.turns[conv][3].id] and flags[0]["label"] == "update" and flags[0]["earlier_ids"]
        await s.close()
    asyncio.run(go())


def test_live_flag_watcher_skips_backlog_and_short_turns(raw):
    async def go():
        world = World()
        s = make_session(raw, world)
        await s._open(0)
        conv = s._conv
        world.turns[conv] = [turn(i, "A", "one two three four five six", i, i + 1, conv=conv) for i in range(6)]
        world.turns[conv].append(turn(6, "A", "too short", 7, 8, conv=conv))
        checked = []
        orig = s.rig.check_flag
        s.rig.check_flag = lambda c, t: (checked.append(t.seq), orig(c, t))[1]
        for _ in range(4):
            await s.poll()
            await asyncio.sleep(0.02)
        assert 6 not in checked and checked and checked[0] >= 3  # the oldest unchecked turns were skipped, not queued
        await s.close()
    asyncio.run(go())


def test_live_pause_stops_feeding_and_play_resumes_without_reset(raw):
    async def go():
        world = World()
        gate = asyncio.Event()

        async def slow(_s):
            await gate.wait()

        s = make_session(raw, world)
        s.pace = slow
        await s.play(0)
        await settle(lambda: world.transports[0].audio == 1)
        await s.pause()
        fed = s.fed_ms
        await asyncio.sleep(0.02)
        assert world.transports[0].audio == 1 and "reset" in types(s)
        gate.set()
        await s.play(fed)  # resume at the same place: same conversation
        await settle(lambda: "ended" in types(s))
        assert len(world.transports) == 1 and types(s).count("reset") == 1
        await s.close()
    asyncio.run(go())


def test_live_seek_starts_a_fresh_conversation_at_that_position(raw):
    async def go():
        world = World()
        s = make_session(raw, world)
        await s.play(0)
        await settle(lambda: "ended" in types(s))
        first = s._conv
        await s.seek(2000, playing=True)
        await settle(lambda: types(s).count("ended") == 2)
        assert len(world.transports) == 2 and world.transports[0].sent_types[-1] == "delete"
        assert s._conv != first and s.offset_ms == 2000
        new_turns = [e["turn"] for e in s.log.since(0)[0] if e["type"] == "turn"][4:]
        assert new_turns and new_turns[0]["start_ms"] == 2000  # displayed on the meeting's clock
        assert world.transports[1].audio == 4  # 2 s remain
        await s.close()
    asyncio.run(go())


def test_live_play_far_from_the_fed_position_resets(raw):
    async def go():
        world = World()
        s = make_session(raw, world)
        await s.play(0)
        await settle(lambda: "ended" in types(s))
        await s.play(1000)  # the player was seeked while paused: audio no longer matches
        await settle(lambda: len(world.transports) == 2)
        await s.close()
    asyncio.run(go())


def test_live_press_freezes_the_seq_and_builds_traces(raw):
    async def go():
        world = World()
        s = make_session(raw, world)
        assert s.press(["single_plus"], None)["error"]
        await s.play(0)
        await settle(lambda: "ended" in types(s))
        r = await asyncio.to_thread(s.press, ["single_plus", "team_routed"], "price?")
        assert r["as_of_seq"] == 3 and [x["assistant"] for x in r["results"]] == ["single_plus", "team_routed"]
        assert world.answers == [("single_plus", 3, "price?"), ("team_routed", 3, "price?")]
        assert r["results"][0]["trace"]["specialists"][0]["name"] == "single"
        assert r["results"][1]["trace"]["stages"][0]["ms"] == s.last_lag_ms
        await s.close()
    asyncio.run(go())


def test_event_log_since():
    log = EventLog()
    log.add({"type": "a"})
    log.add({"type": "b"})
    assert log.since(1) == ([{"type": "b", "n": 1}], 2) and log.since(5) == ([], 2)


def test_live_endpoints_through_the_server(raw, patched):
    world = World()
    app = FastAPI()
    S.add_room(app, raw, rig=world.rig(), pace=instant, poll_s=0.01)
    with TestClient(app) as c:
        assert c.get("/api/health").json()["mode"] == "live"
        assert c.get("/api/room/meetings/ES2003a/replay").status_code == 409
        sid = c.post("/api/room/live/start", json={"meeting": "ES2003a"}).json()["session"]
        assert c.post(f"/api/room/live/{sid}/control", json={"action": "play", "position_ms": 0}).json() == {"ok": True}
        deadline = time.time() + 5
        while time.time() < deadline:
            r = c.get(f"/api/room/live/{sid}/events", params={"since": 0}).json()
            if any(e["type"] == "ended" for e in r["events"]):
                break
            time.sleep(0.02)
        assert [e["type"] for e in r["events"]].count("turn") == 4
        assert c.get(f"/api/room/live/{sid}/events", params={"since": r["next"]}).json()["events"] == []
        pr = c.post("/api/room/press", json={"meeting": "ES2003a", "assistant": "team_routed", "question": "q", "session": sid}).json()
        assert pr["as_of_seq"] == 3 and pr["results"][0]["assistant"] == "team_routed"
        assert c.post("/api/room/press", json={"meeting": "ES2003a", "session": "nope"}).status_code == 404
        assert c.post(f"/api/room/live/{sid}/control", json={"action": "dance"}).status_code == 400
        assert c.post(f"/api/room/live/{sid}/control", json={"action": "pause"}).json() == {"ok": True}
        assert c.delete(f"/api/room/live/{sid}").json() == {"ok": True}
        assert c.get(f"/api/room/live/{sid}/events").status_code == 404


def test_live_connect_failure_is_reported_not_raised_as_500(raw, patched):
    world = World()
    rig = world.rig()

    async def broken(conv):
        raise ConnectionError("down")

    rig.connect = broken
    app = FastAPI()
    S.add_room(app, raw, rig=rig, pace=instant, poll_s=0.01)
    with TestClient(app) as c:
        sid = c.post("/api/room/live/start", json={"meeting": "ES2003a"}).json()["session"]
        r = c.post(f"/api/room/live/{sid}/control", json={"action": "play"})
        assert r.status_code == 502 and "ConnectionError" in r.json()["error"]
        ev = c.get(f"/api/room/live/{sid}/events").json()["events"]
        assert ev[-1]["type"] == "status" and "stream API" in ev[-1]["detail"]


# ---- rig ----

class FakeSvc:
    def __init__(self):
        self.store = InMemoryTranscriptStore()
        for t in TURNS:
            self.store.add(t)
        from contextlens.memory.extract import MemoryStore
        self.memory, self.retriever = MemoryStore(), object()


class FakeChecker:
    def __init__(self, label):
        self.label = label

    def check(self, statement, earlier):
        return self.label, 0.9


def test_rig_check_flag_flags_only_conflicts():
    svc = FakeSvc()

    class FakeRetriever:
        def search(self, conv, query, as_of, k=8, speaker=None):
            from contextlens.retrieval.hybrid import ScoredTurn
            return [ScoredTurn(turn=t, score=1.0) for t in svc.store.turns(conv, as_of) if t.seq < as_of]

    svc.retriever = FakeRetriever()
    for label, flagged in (("update", True), ("contradiction", True), ("consistent", False), ("insufficient", False)):
        rig = rig_mod.make_rig(svc, {}, "ws://x", build=None, checker=lambda label=label: FakeChecker(label))
        out = rig.check_flag("ES2003a", TURNS[2])
        assert (out is not None) is flagged
        if flagged:
            assert out["label"] == label and "ES2003a:t0" in out["earlier_ids"] and "ES2003a:t2" not in out["earlier_ids"]
    rig = rig_mod.make_rig(svc, {}, "ws://x", build=None, checker=lambda: FakeChecker("update"))
    assert rig.check_flag("ES2003a", TURNS[1].model_copy(update={"id": "other"})) is None


def test_rig_answer_freezes_the_store_and_returns_ctx():
    svc = FakeSvc()
    seen = {}

    class A:
        def answer(self, req, ctx):
            seen["turns"] = [t.seq for t in ctx.store.turns(req.conversation_id, 99)]
            with pytest.raises(PermissionError):
                ctx.store.add(TURNS[0])
            return Card(mode="recall", status="abstained", findings=[], missing=["x"])

    rig = rig_mod.make_rig(svc, {}, "ws://x", build=lambda name, *a: A(), checker=lambda: None)
    card, ctx = rig.answer("team_routed", "ES2003a", 1, "q")
    assert seen["turns"] == [0, 1] and ctx.as_of_seq == 1 and card.status == "abstained"
    assert [t.seq for t in rig.turns("ES2003a")] == [0, 1, 2, 3] and rig.memory("ES2003a") == []


def test_connect_ws_gives_up_with_the_error_type(monkeypatch):
    class Boom(Exception):
        pass

    fake = type(sys)("websockets")

    async def refuse(*a, **k):
        raise Boom("secret detail")

    fake.connect = refuse
    monkeypatch.setitem(sys.modules, "websockets", fake)
    monkeypatch.setattr(rig_mod, "CONNECT_TRIES", 2)

    async def quick(_s):
        return None

    monkeypatch.setattr(rig_mod.asyncio, "sleep", quick)
    with pytest.raises(ConnectionError, match="Boom") as e:
        asyncio.run(rig_mod.connect_ws("ws://127.0.0.1:1", "c"))
    assert "secret" not in str(e.value)


def test_gpu_lock_is_free_between_tests():
    assert live_mod.GPU.acquire(blocking=False)
    live_mod.GPU.release()


# ---- the static page and its scripted data ----

SITE = Path(__file__).resolve().parents[1] / "site"


def test_scripted_data_follows_the_as_of_rule_and_is_current():
    data = json.loads((SITE / "room-data.json").read_text(encoding="utf-8"))
    end = {t["id"]: t["end_ms"] for t in data["turns"]}
    for press in data["presses"]:
        for r in press["results"]:
            cited = {e["turn_id"] for f in r["card"]["findings"] for e in f["evidence"]}
            assert all(end[i] <= press["after_ms"] for i in cited), (press["label"], press["after_ms"])
            assert r["trace"]["stages"] and r["assistant"] in ("single_plus", "team_routed")
    for item in data["memory"] + data["flags"]:
        ids = item.get("source_turn_ids") or [item["turn_id"], *item["earlier_ids"]]
        assert max(end[i] for i in ids) <= item["at_ms"]
    sys.path.insert(0, str(SITE.parent / "tools"))
    import make_room_data

    assert [p["results"] for p in data["presses"]] == [p["results"] for p in json.loads(json.dumps(make_room_data.PRESSES, default=lambda o: o.model_dump(mode="json")))]


def test_room_page_rules():
    css = (SITE / "room.css").read_text(encoding="utf-8")
    js = (SITE / "room.js").read_text(encoding="utf-8")
    html = (SITE / "room.html").read_text(encoding="utf-8")
    assert "transition: all" not in css and "transition:all" not in css
    import re
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b|rgb\(|hsl\(|oklch\(", css), "colours come from tokens.css"
    assert ":focus-visible" in (SITE / "base.css").read_text(encoding="utf-8")
    assert "innerHTML" not in js and "insertAdjacentHTML" not in js and "outerHTML" not in js
    assert "prefers-reduced-motion" in js and "CC BY" in (Path(S.__file__).parent / "meetings.py").read_text(encoding="utf-8")
    assert 'href="room.css"' in html and 'src="room.js"' in html
    assert 'href="room.html"' in (SITE / "live.html").read_text(encoding="utf-8")
