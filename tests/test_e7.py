"""E7, real-time cost: speech and speaker error metrics, the AMI scenario source, the runner, the size check."""
import asyncio
import hashlib
import json
import subprocess
import wave
from pathlib import Path

import pytest
from prometheus_client import CollectorRegistry

from contextlens.api import production
from contextlens.eval.check_run import check, recompute_metrics
from contextlens.eval.metrics import diarization_error_rate, word_error_rate
from contextlens.eval.records import write_local_manifest
from contextlens.experiments import e7_realtime as e7
from contextlens.llm.client import FakeLLM
from contextlens.memory.store import InMemoryTranscriptStore
from contextlens.obs.tracing import Telemetry
from contextlens.replay.runner import Context, Prediction
from contextlens.schemas import (
    ButtonRequest, Card, Conversation, EvidenceRef, Finding, MemoryItem, Scenario, Turn,
)
from contextlens.speech.client import Segment
from contextlens.speech.diarize import BYTES_PER_MS, SAMPLE_RATE

ROOT = Path(__file__).resolve().parents[1]
TURN_MS = 3000
GB = 10**9


def test_wer_on_known_strings():
    assert word_error_rate(["the cat sat on the mat"], ["the cat sat on mat"]) == pytest.approx(1 / 6)  # one deletion
    assert word_error_rate(["a b c d"], ["a x c d e"]) == pytest.approx(2 / 4)  # one substitution, one insertion
    assert word_error_rate(["Hello, World!"], ["hello world"]) == 0.0  # case and punctuation do not count
    assert word_error_rate(["one two"], [""]) == 1.0
    assert word_error_rate(["a b", "c d e f"], ["a b", "c x e"]) == pytest.approx(2 / 6)  # summed over pairs, not averaged
    assert word_error_rate([""], ["anything"]) is None  # no reference words


def test_der_on_known_spans():
    ref = [(0, 10_000, "A"), (10_000, 20_000, "B")]
    assert diarization_error_rate([ref], [[(0, 10_000, "x"), (10_000, 20_000, "y")]]) == 0.0  # names are matched
    assert diarization_error_rate([ref], [[(0, 10_000, "y"), (10_000, 20_000, "x")]]) == 0.0
    assert diarization_error_rate([ref], [[(0, 20_000, "x")]]) == pytest.approx(0.5)  # half confused
    assert diarization_error_rate([ref], [[(0, 10_000, "x")]]) == pytest.approx(0.5)  # half missed
    assert diarization_error_rate([ref], [[(0, 20_000, "unknown")]]) == 1.0  # unknown is no speech
    assert diarization_error_rate([ref], [[]]) == 1.0
    overlap = [(0, 10_000, "A"), (5_000, 10_000, "B")]
    assert diarization_error_rate([overlap], [[(0, 10_000, "x")]]) == pytest.approx(1 / 3)
    assert diarization_error_rate([[]], [[(0, 1_000, "x")]]) is None


def meeting_turns(m: str) -> list[Turn]:
    return [Turn(id=f"{m}:t{i}", conversation_id=m, seq=i, speaker="AA" if i % 2 == 0 else "BB",
                 text=f"turn {i} text about item{i}", start_ms=i * TURN_MS, end_ms=(i + 1) * TURN_MS, source="dataset")
            for i in range(8)]


def meeting_scenarios(m: str) -> list[Scenario]:
    out = []
    for n, as_of in enumerate((5, 7), start=1):
        conv = Conversation(id=m, turns=meeting_turns(m)[: as_of + 1], source="ami", split="dev")
        req = ButtonRequest(conversation_id=m, as_of_seq=as_of, question="what about item1?")
        out.append(Scenario(id=f"{m}-q{n}", conversation=conv, request=req, category="recall",
                            gold_evidence_turn_ids=[f"{m}:t1"], gold_label=None, gold_answer="x", origin="dataset"))
    return out


def test_row_speech_fields_recompute_wer_and_der():
    sc = meeting_scenarios("m1")[0]  # only the turns up to as_of_seq=5 count
    ref = " ".join(t.text for t in meeting_turns("m1")[:6])
    row = Prediction(scenario_id="m1-q1", ranked_turn_ids=["m1:t1"], asr_text=ref.replace("turn 3", "turn"),
                     speaker_spans=[(t.start_ms, t.end_ms, "s1" if t.seq % 2 == 0 else "s2")
                                    for t in meeting_turns("m1")[:6]])
    fresh = recompute_metrics([sc], [row])
    assert fresh["wer"] == pytest.approx(1 / 30)  # one of 30 words dropped
    assert fresh["der"] == 0.0
    bare = recompute_metrics([sc], [Prediction(scenario_id="m1-q1", ranked_turn_ids=["m1:t1"])])
    assert "wer" not in bare and "der" not in bare  # no speech output on the row, nothing to recompute


# -- doubles ---------------------------------------------------------------------------------------------------------

def aligned_script(m: str) -> list[Segment]:
    """Same segmentation as the clean transcript, with one wrong word in each turn of five."""
    return [Segment(text=t.text.replace("text", "tekst"), start_ms=t.start_ms, end_ms=t.end_ms)
            for t in meeting_turns(m)]


def merged_script(m: str) -> list[Segment]:
    """The speech service closes one utterance over two clean turns."""
    ts = meeting_turns(m)
    return [Segment(text=a.text + " " + b.text, start_ms=a.start_ms, end_ms=b.end_ms) for a, b in zip(ts[::2], ts[1::2])]


class ScriptedStream:
    def __init__(self, script: list[Segment]) -> None:
        self.script, self.sent = script, 0

    def send(self, pcm: bytes) -> None:
        self.sent += len(pcm)

    async def segments(self):
        i = 0
        while True:
            while i < len(self.script) and self.script[i].end_ms * BYTES_PER_MS <= self.sent:
                yield self.script[i]
                i += 1
            await asyncio.sleep(0.005)

    async def close(self) -> None:
        pass


class ScriptedSpeech:
    """Each open() gets the next meeting's script."""

    def __init__(self, scripts: list[list[Segment]]) -> None:
        self.scripts = list(scripts)

    async def open(self) -> ScriptedStream:
        return ScriptedStream(self.scripts.pop(0))


class TruthBackend:
    """A speaker model that is right about every second of the audio it is given."""

    def spans(self, pcm: bytes) -> list[tuple[int, int, str]]:
        end = len(pcm) // BYTES_PER_MS
        return [(i * TURN_MS, min((i + 1) * TURN_MS, end), "p" if i % 2 == 0 else "q")
                for i in range(8) if i * TURN_MS < end]


class CiteLast:
    """Cites the newest turn it can see, and reports stage times."""

    def __init__(self, store) -> None:
        self.store = store

    def answer(self, req: ButtonRequest, ctx: Context) -> Card:
        last = self.store.turns(req.conversation_id, ctx.as_of_seq)[-1]
        ev = EvidenceRef(turn_id=last.id, quote=last.text, speaker=last.speaker, start_ms=last.start_ms)
        ctx.stage_ms.update(memory_ms=30, consistency_ms=10, assistance_ms=20, memory_tokens=7)
        return Card(mode="recall", status="answered", tokens=50, findings=[
            Finding(agent="memory", kind="recall", statement=last.text, evidence=[ev], confidence=0.9)])


class FirstTurnExtractor:
    """One memory item per batch, from its first turn, with the speaker that turn has when it is read."""

    def update(self, new_turns: list[Turn], existing: list[MemoryItem]) -> list[MemoryItem]:
        t = new_turns[0]
        return [MemoryItem(id=f"{t.id}-m", conversation_id=t.conversation_id, kind="fact", subject=t.text,
                           statement=t.text, speaker=t.speaker, source_turn_ids=[t.id], as_of_seq=t.seq)]


def make_assistant(store, retriever, memory):
    return CiteLast(store)


class NoEmbed:
    model = "nomic-embed-text"

    def embed(self, texts):
        return [[0.0] for _ in texts]


def write_silence(path: Path, seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(bytes(int(seconds * SAMPLE_RATE) * 2))


# -- speech and one meeting ------------------------------------------------------------------------------------------

def test_transcribe_collects_segments_and_lags():
    pcm = bytes(24 * SAMPLE_RATE * 2)
    segs, lags, wall = asyncio.run(e7.transcribe(ScriptedSpeech([aligned_script("m1")]), pcm, speedup=1000, settle_s=0.05))
    assert [s.text for s in segs] == [s.text for s in aligned_script("m1")]
    assert len(lags) == 8 and all(x >= 0 for x in lags) and wall > 0


def test_transcribe_raises_when_the_service_fails_mid_stream():
    class Failing(ScriptedStream):
        async def segments(self):
            raise e7.SpeechUnavailable("down")
            yield

    class Speech:
        async def open(self):
            return Failing([])

    with pytest.raises(e7.SpeechUnavailable):
        asyncio.run(e7.transcribe(Speech(), bytes(5 * SAMPLE_RATE * 2), speedup=1000, settle_s=0.05))


def test_one_meeting_three_ways():
    scs = meeting_scenarios("m1")
    rows, stats = e7.replay_meeting(
        scs, bytes(24 * SAMPLE_RATE * 2), aligned_script("m1"), backend=TruthBackend(), embedder=NoEmbed(),
        make_assistant=make_assistant, extractor=FirstTurnExtractor(), telemetry=None, step_ms=10_000,
        say=lambda m: None)
    assert set(rows) == set(e7.SYSTEMS) and all(set(r) == {"m1-q1", "m1-q2"} for r in rows.values())
    preds = {n: [r["m1-q1"], r["m1-q2"]] for n, r in rows.items()}
    fresh = recompute_metrics(scs, preds["asr_pipeline"])
    assert fresh["wer"] == pytest.approx(0.2)  # "text" heard as "tekst": one word of five
    assert fresh["der"] == 0.0  # the speaker model was right
    assert "wer" not in recompute_metrics(scs, preds["clean"])
    assert recompute_metrics(scs, preds["asr_oracle_speakers"])["speaker_accuracy"] == 1.0
    # the cited turn is the clean one, and the newest turn at the question
    assert preds["asr_pipeline"][0].ranked_turn_ids == ["m1:t5"] and preds["clean"][1].ranked_turn_ids == ["m1:t7"]
    # memory items were read while their turns were unknown, and keep that label
    assert stats["stale_unknown_items"] > 0 and stats["turns_still_unknown"] == 0


def test_merged_segments_cite_clean_turn_ids():
    scs = meeting_scenarios("m1")
    rows, _ = e7.replay_meeting(
        scs, bytes(24 * SAMPLE_RATE * 2), merged_script("m1"), backend=TruthBackend(), embedder=NoEmbed(),
        make_assistant=make_assistant, extractor=None, telemetry=None, step_ms=10_000, say=lambda m: None)
    assert rows["asr_pipeline"]["m1-q2"].ranked_turn_ids == ["m1:t6"]  # t6 and t7 overlap it equally; the earlier wins
    assert recompute_metrics(scs, [rows["asr_pipeline"][s.id] for s in scs])["der"] > 0  # two speakers in one turn


# -- a whole record through check_run --------------------------------------------------------------------------------

def git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / ".gitignore").write_text("datasets/raw/\n", encoding="utf-8")
    write_local_manifest("ami", {"dev": meeting_scenarios("m1") + meeting_scenarios("m2")}, tmp_path)
    (tmp_path / "datasets/ami-questions").mkdir(parents=True)
    (tmp_path / e7.QUESTIONS_REL).write_text(json.dumps({"labels_reviewed": False, "questions": []}), encoding="utf-8")
    for m in ("m1", "m2"):
        write_silence(tmp_path / e7.AUDIO_REL / f"{m}.wav", 24)
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


SPEEDS: list[float] = []


@pytest.fixture(autouse=True)
def fast_clock(monkeypatch):
    """Records the speed each replay asks for, then runs it fast so the tests do not wait in real time."""
    real = e7.transcribe
    SPEEDS.clear()

    async def quick(speech, pcm, *, speedup=1.0, settle_s=10.0):
        SPEEDS.append(speedup)
        return await real(speech, pcm, speedup=1000.0, settle_s=0.05)

    monkeypatch.setattr(e7, "transcribe", quick)


def run(repo: Path, speedup: float = 1000.0, realtime_meeting: str | None = "m1", **kw) -> Path:
    if speedup == 1.0:
        realtime_meeting = None
    scripts = [aligned_script("m1"), merged_script("m2")] + ([aligned_script("m1")] if realtime_meeting else [])
    return e7.run_e7(
        root=repo, speech=ScriptedSpeech(scripts), backend=TruthBackend(),
        embedder=NoEmbed(), make_assistant=make_assistant, extractor=FirstTurnExtractor(),
        models={"llm": "qwen3.5:4b-q4_K_M", "speech": "whisper.cpp ggml-tiny.en", "diarizer": "pyannote"},
        prompt_versions={"team": "v1"}, say=lambda m: None, speedup=speedup, realtime_meeting=realtime_meeting,
        step_ms=10_000, **kw)


def test_tiny_fixture_writes_a_valid_record(repo: Path) -> None:
    reg = CollectorRegistry()
    path = run(repo, telemetry=Telemetry(registry=reg))
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert check("E7", root=repo, min_n=4, split="dev") == []
    assert rec["scenario_source"] == "ami" and rec["n"] == 4 and rec["config"]["phase"] == "baseline"
    assert set(rec["metrics"]) == set(e7.SYSTEMS)
    assert "wer" in rec["metrics"]["asr_pipeline"] and "der" in rec["metrics"]["asr_pipeline"]
    assert "wer" not in rec["metrics"]["clean"]
    assert "latency_p95_ms[stage=memory_ms]" in rec["metrics"]["asr_pipeline"]
    assert rec["metrics"]["asr_pipeline"]["tokens_per_card"] == 50
    assert [b["stage"] for b in rec["config"]["bottlenecks"]] == ["memory_ms", "assistance_ms"]  # by measured p95
    assert rec["config"]["obs_profile"] is True
    assert reg.get_sample_value("contextlens_presses_total", {"status": "answered", "mode": "recall"}) == 12
    assert rec["config"]["error_analysis"]["memory_items_extracted_while_unknown_now_stale"] > 0
    assert rec["claude_calls"] == 0


def test_a_saved_wer_that_does_not_recompute_is_rejected(repo: Path) -> None:
    path = run(repo)
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec["metrics"]["asr_pipeline"]["wer"] += 0.1
    path.write_text(json.dumps(rec), encoding="utf-8")
    assert any("wer does not recompute" in p for p in check("E7", root=repo, split="dev"))


def test_rerun_records_what_changed_beside_the_baseline(repo: Path) -> None:
    base = run(repo)
    with pytest.raises(ValueError, match="what changed"):
        run(repo, rerun_of=base.stem, changes=[])
    with pytest.raises(ValueError, match="belong to a rerun"):
        run(repo, changes=["x"])
    after = run(repo, rerun_of=base.stem, changes=["cap the context the memory agent reads"])
    cfg = json.loads(after.read_text(encoding="utf-8"))["config"]
    assert cfg["phase"] == "rerun" and cfg["baseline_run_id"] == base.stem
    assert cfg["bottlenecks"] == json.loads(base.read_text(encoding="utf-8"))["config"]["bottlenecks"]
    assert cfg["changes"] == ["cap the context the memory agent reads"]
    assert set(cfg["before_after"]) == {"memory_ms", "assistance_ms", "total", "quality"}
    assert cfg["before_after"]["memory_ms"]["p95"] == {"before": 30, "after": 30}
    assert check("E7", root=repo, min_n=4, split="dev") == []  # both records stand
    with pytest.raises(ValueError, match="not an E7 baseline"):
        run(repo, rerun_of=after.stem, changes=["again"])


def test_run_refuses_without_audio_or_committed_manifest(repo: Path) -> None:
    (repo / e7.AUDIO_REL / "m2.wav").unlink()
    with pytest.raises(FileNotFoundError, match="m2"):
        run(repo)
    write_silence(repo / e7.AUDIO_REL / "m2.wav", 24)
    (repo / "datasets/splits/ami-scenarios.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="committed"):
        run(repo)


def test_the_committed_dev_scenarios_are_untouched() -> None:
    manifest = json.loads((ROOT / "datasets/splits/scenarios.json").read_text(encoding="utf-8"))["splits"]
    for split in ("dev", "test"):
        digest = hashlib.sha256((ROOT / f"datasets/scenarios/{split}.jsonl").read_bytes()).hexdigest()
        assert digest == manifest[split]["hash"]


# -- building the AMI scenarios, the size check -----------------------------------------------------------------------

def test_build_ami_dev_cuts_each_meeting_at_the_question(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "datasets/splits").mkdir(parents=True)
    (tmp_path / "datasets/splits/ami.json").write_text(json.dumps({"splits": {"dev": ["m1"], "test": ["m9"]}}))
    (tmp_path / "datasets/ami-questions").mkdir(parents=True)
    q = {"meeting": "m1", "as_of_seq": 5, "question": "what?", "category": "recall",
         "gold_evidence_turn_ids": ["m1:t2"], "gold_label": None, "gold_answer": "a"}

    def questions(*qs):
        (tmp_path / e7.QUESTIONS_REL).write_text(json.dumps({"labels_reviewed": True, "questions": list(qs)}))

    monkeypatch.setattr(e7, "load_dataset", lambda name, root: [
        Conversation(id=m, turns=meeting_turns(m), source="ami") for m in ("m1", "m9")])
    questions(q, {**q, "as_of_seq": 7})
    assert e7.build_ami_dev(tmp_path) == 2
    built = [Scenario.model_validate_json(x) for x in
             (tmp_path / "datasets/raw/scenarios/ami/dev.jsonl").read_text("utf-8").splitlines()]
    assert [s.id for s in built] == ["m1-q1", "m1-q2"] and [len(s.conversation.turns) for s in built] == [6, 8]
    assert e7.meetings_needed(tmp_path) == ["m1"]
    for bad in ({**q, "meeting": "m9"}, {**q, "gold_evidence_turn_ids": ["m1:t6"]}, {**q, "as_of_seq": 99}):
        questions(bad)
        with pytest.raises(ValueError):
            e7.build_ami_dev(tmp_path)


def sized(sizes: dict, free: int) -> dict:
    return e7.size_check(list(sizes), head=lambda url: sizes[url.split("/")[-3]], free=lambda: free,
                         sleep=lambda s: None)


def test_size_check_stops_on_the_disk_rule() -> None:
    ok = sized({"a": 3 * 10**8, "b": 4 * 10**8}, 500 * GB)
    assert ok["stop"] == [] and ok["total_bytes"] == 7 * 10**8
    assert any("15 GB" in s for s in sized({"a": 16 * GB}, 900 * GB)["stop"])
    assert any("under 40 GB" in s for s in sized({"a": 2 * GB}, 41 * GB)["stop"])
    assert sized({"a": 2 * GB}, 43 * GB)["stop"] == []
    assert sized({"a": None}, 500 * GB)["stop"] == ["size of a unknown"]


def test_download_checks_size_first_and_fetches_only_what_is_missing(tmp_path: Path) -> None:
    fetched: list[str] = []

    def fetch(url: str, dest: Path) -> None:
        fetched.append(url)
        dest.write_bytes(b"x" * 10)

    kw = {"sleep": lambda s: None, "free": lambda: 900 * GB, "fetch": fetch}
    with pytest.raises(RuntimeError, match="stop"):
        e7.download(["a"], tmp_path / "audio", head=lambda url: 20 * GB, **kw)
    assert fetched == [] and not (tmp_path / "audio").exists()
    (tmp_path / "audio").mkdir()
    (tmp_path / "audio/a.wav").write_bytes(b"y" * 10)  # complete: the same size as the server gave
    got = e7.download(["a", "b"], tmp_path / "audio", head=lambda url: 10, **kw)
    assert got == ["b"] and len(fetched) == 1 and "/b/audio/b.Mix-Headset.wav" in fetched[0]
    assert (tmp_path / "audio/b.wav").read_bytes() == b"x" * 10
    assert sorted(p.name for p in (tmp_path / "audio").iterdir()) == ["a.wav", "b.wav"]


def test_download_keeps_partial_files_out_of_the_way(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / "a.wav").write_bytes(b"short")  # a partial left under the final name: the wrong size
    (audio / ".b.wav.tmp").write_bytes(b"stale")  # a crashed fetch's leftover

    def dies(url: str, dest: Path) -> None:
        dest.write_bytes(b"half")
        raise ConnectionError("cut off")

    kw = {"sleep": lambda s: None, "free": lambda: 900 * GB, "head": lambda url: 10}
    with pytest.raises(ConnectionError):
        e7.download(["b", "a"], audio, fetch=dies, **kw)
    assert sorted(p.name for p in audio.iterdir()) == ["a.wav"]  # no .wav appeared for b and no .tmp is left

    def short(url: str, dest: Path) -> None:
        dest.write_bytes(b"1234")  # ends cleanly but is not the size the server gave

    with pytest.raises(OSError, match="server said 10"):
        e7.download(["b"], audio, fetch=short, **kw)
    assert not (audio / "b.wav").exists() and not (audio / ".b.wav.tmp").exists()

    def whole(url: str, dest: Path) -> None:
        dest.write_bytes(b"z" * 10)

    assert e7.download(["a", "b"], audio, fetch=whole, **kw) == ["a", "b"]  # the wrong-size a.wav is fetched again
    assert (audio / "a.wav").read_bytes() == b"z" * 10


def test_requests_go_one_at_a_time_with_pauses_and_retry_after_a_403() -> None:
    waits: list[float] = []
    tries: dict[str, int] = {}

    def head(url: str) -> int:
        m = url.split("/")[-3]
        tries[m] = tries.get(m, 0) + 1
        if m == "b" and tries[m] < 3:
            raise e7.RateLimited()  # 403 twice, then 200
        if m == "c" and tries[m] == 1:
            raise e7.RateLimited(retry_after=200)  # the server names its own wait
        return 5

    report = e7.size_check(["a", "b", "c"], head=head, free=lambda: 900 * GB, sleep=waits.append)
    assert report["stop"] == [] and report["sizes"] == {"a": 5, "b": 5, "c": 5}
    assert waits == [2.0, 30.0, 120.0, 2.0, 200.0]  # pause; 30 then 120 for b; pause, then the Retry-After for c
    assert tries == {"a": 1, "b": 3, "c": 2}


def test_a_file_still_refused_after_three_tries_stops_the_download() -> None:
    waits: list[float] = []
    calls: list[str] = []

    def head(url: str) -> int:
        calls.append(url)
        if "/b/" in url:
            raise e7.RateLimited()
        return 5

    report = e7.size_check(["a", "b"], head=head, free=lambda: 900 * GB, sleep=waits.append)
    assert report["stop"] == ["size of b unknown"] and sum("/b/" in c for c in calls) == 3
    assert waits == [2.0, 30.0, 120.0]  # no fourth try
    fetched: list[str] = []
    with pytest.raises(RuntimeError, match="unknown"):
        e7.download(["a", "b"], Path("unused-audio-dir"), head=head, free=lambda: 900 * GB, sleep=lambda s: None,
                    fetch=lambda url, dest: fetched.append(url))
    assert fetched == []


def test_a_refused_fetch_is_retried_then_raised(tmp_path: Path) -> None:
    waits: list[float] = []
    n = {"calls": 0}

    def fetch(url: str, dest: Path) -> None:
        n["calls"] += 1
        if n["calls"] < 3:
            raise e7.RateLimited()
        dest.write_bytes(b"x" * 5)

    kw = {"head": lambda url: 5, "free": lambda: 900 * GB, "sleep": waits.append}
    assert e7.download(["a"], tmp_path / "audio", fetch=fetch, **kw) == ["a"] and waits == [30.0, 120.0]

    def always(url: str, dest: Path) -> None:
        raise e7.RateLimited()

    with pytest.raises(e7.RateLimited):
        e7.download(["z"], tmp_path / "audio", fetch=always, **kw)
    assert not (tmp_path / "audio/z.wav").exists()


def test_head_size_reads_403_429_and_retry_after(monkeypatch) -> None:
    import io
    import urllib.error

    def raising(code, headers=None):
        def urlopen(req, timeout):
            raise urllib.error.HTTPError(req.full_url, code, "x", headers or {}, io.BytesIO())
        return urlopen

    monkeypatch.setattr(e7.urllib.request, "urlopen", raising(429, {"Retry-After": "90"}))
    with pytest.raises(e7.RateLimited) as err:
        e7.head_size("http://x/y")
    assert err.value.retry_after == 90
    monkeypatch.setattr(e7.urllib.request, "urlopen", raising(403))
    with pytest.raises(e7.RateLimited):
        e7.head_size("http://x/y")
    monkeypatch.setattr(e7.urllib.request, "urlopen", raising(404))
    assert e7.head_size("http://x/y") is None  # only 403 and 429 are retried


# -- unloading the models at shutdown ---------------------------------------------------------------------------------

def test_shutdown_unloads_the_models_even_if_the_flush_fails(monkeypatch) -> None:
    unloaded: list[int] = []
    prod = production.build_services(
        store=InMemoryTranscriptStore(), embedder=NoEmbed(),
        llms={"memory": FakeLLM([]), "consistency": FakeLLM([]), "assistance": FakeLLM([])},
        background_llm=None, speech=ScriptedSpeech([]), unload=lambda: unloaded.append(1))

    def broken(timeout_s: float = 5.0) -> None:
        raise RuntimeError("flush failed")

    prod.traced.flush = broken
    with pytest.raises(RuntimeError):
        prod.services.on_shutdown()
    assert unloaded == [1]


def test_unload_local_models_asks_ollama_to_drop_both(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(production, "unload_ollama", lambda model, url: calls.append((model, url)) or True)
    production.unload_local_models()
    assert calls == [("qwen3.5:4b-q4_K_M", "http://127.0.0.1:11434"), ("nomic-embed-text", "http://127.0.0.1:11434")]


def test_a_sped_up_run_states_its_speedup_and_takes_latency_from_a_real_time_segment(repo: Path) -> None:
    rec = json.loads(run(repo).read_text(encoding="utf-8"))
    cfg = rec["config"]
    assert cfg["speedup"] == 1000.0 and cfg["latency_basis"].startswith("sped_up")
    assert cfg["pipeline_timing"] is None  # speech and speaker timings of a sped-up replay are not reported
    seg = cfg["realtime_segment"]
    assert seg["meeting"] == "m1" and seg["speedup"] == 1.0 and seg["asr_lag_ms"]["n"] == 8
    assert "latency_p95_ms" in seg["press_latency_ms"] and seg["asr_real_time_factor"] > 0
    assert SPEEDS == [1000.0, 1000.0, 1.0]  # the two meetings sped up, then the segment at real time
    assert check("E7", root=repo, min_n=4, split="dev") == []


def test_a_real_time_run_needs_no_segment_and_a_sped_up_run_cannot_skip_it(repo: Path) -> None:
    cfg = json.loads(run(repo, speedup=1.0).read_text(encoding="utf-8"))["config"]
    assert cfg["speedup"] == 1.0 and cfg["latency_basis"] == "real_time" and cfg["realtime_segment"] is None
    assert cfg["pipeline_timing"]["asr_lag_ms"]["n"] == 12 and SPEEDS == [1.0, 1.0]
    with pytest.raises(ValueError, match="realtime_meeting"):
        run(repo, realtime_meeting=None)
    with pytest.raises(ValueError, match="realtime_meeting"):
        run(repo, realtime_meeting="nope")
    with pytest.raises(ValueError, match="positive"):
        run(repo, speedup=0.0)
    with pytest.raises(ValueError, match="real time throughout"):
        e7.run_e7(root=repo, speech=ScriptedSpeech([]), backend=TruthBackend(), embedder=NoEmbed(),
                  make_assistant=make_assistant, extractor=None, models={}, prompt_versions={}, speedup=1.0,
                  realtime_meeting="m1")
