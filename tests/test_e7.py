"""E7, real-time cost: speech and speaker error metrics, the AMI scenario source, the runner, the size check."""
import asyncio
import hashlib
import json
import re
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

    BACKEND: str | None = "gpu:Fake GPU"

    async def compute_backend(self) -> str | None:
        return self.BACKEND

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
    (tmp_path / ".gitignore").write_text("datasets/raw/\nresults/partial/\nresults/logs/\nresults/local/\n", encoding="utf-8")
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


def ALL_CUDA() -> dict:
    return {"pyannote": "cuda", "embeddings": "cuda", "llm": "cuda"}


@pytest.fixture(autouse=True)
def fast_clock(monkeypatch):
    """Records the speed each replay asks for, then runs it fast so the tests do not wait in real time."""
    real = e7.transcribe
    SPEEDS.clear()

    async def quick(speech, pcm, *, speedup=1.0, settle_s=10.0, info=None):
        SPEEDS.append(speedup)
        return await real(speech, pcm, speedup=1000.0, settle_s=0.05, info=info)

    monkeypatch.setattr(e7, "transcribe", quick)


def run(repo: Path, speedup: float = 1000.0, realtime_meeting: str | None = "m1", **kw) -> Path:
    if speedup == 1.0:
        realtime_meeting = None
    scripts = [aligned_script("m1"), merged_script("m2")] + ([aligned_script("m1")] if realtime_meeting else [])
    return e7.run_e7(
        root=repo, speech=ScriptedSpeech(scripts), backend=TruthBackend(),
        embedder=NoEmbed(), make_assistant=make_assistant, extractor=FirstTurnExtractor(),
        models={"llm": "qwen3.5:4b-q4_K_M", "speech": "whisper.cpp ggml-tiny.en", "diarizer": "pyannote"},
        prompt_versions={"team": "v1"}, speedup=speedup, realtime_meeting=realtime_meeting,
        step_ms=10_000, **({"ran_alone": True, "speech_address": "127.0.0.1:50052", "device_probe": ALL_CUDA, "say": lambda m: None} | kw))


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
                  realtime_meeting="m1", ran_alone=True, speech_address=e7.GPU_SPEECH_ADDRESS, device_probe=ALL_CUDA)


# -- where the audio came from, and where each stage ran --------------------------------------------------------------

def test_every_record_says_the_audio_was_rebuilt_from_segmented_utterances(repo: Path) -> None:
    cfg = json.loads(run(repo).read_text(encoding="utf-8"))["config"]
    assert cfg["audio_source"]["method"] == "rebuilt from segmented utterances" == e7.AUDIO_METHOD
    assert "silence" in cfg["audio_source"]["note"] and cfg["audio_source"]["revision"] is None  # no manifest here
    (repo / e7.AUDIO_REL / "manifest.json").write_text(json.dumps({
        "dataset": "edinburghcstr/ami", "config": "ihm", "revision": "46f28f25", "sample_rate": 16000, "rule": "r"}))
    again = json.loads(run(repo).read_text(encoding="utf-8"))["config"]["audio_source"]
    assert again["method"] == e7.AUDIO_METHOD and again["revision"] == "46f28f25" and again["config"] == "ihm"
    assert not hasattr(e7, "AUDIO_URL") and not hasattr(e7, "download")  # the mirror path is retired


def test_the_record_states_each_stage_device_and_every_fallback_is_warned(repo: Path) -> None:
    cfg = json.loads(run(repo).read_text(encoding="utf-8"))["config"]
    assert set(cfg["devices"]) == {"whisper", "pyannote", "embeddings", "llm"}
    assert all(d["requested"] == "cuda" and d["actual"] == "cuda" for d in cfg["devices"].values())
    assert cfg["devices"]["whisper"]["reported"] == "gpu:Fake GPU" and cfg["stages_not_on_cuda"] == []
    seen: list[str] = []
    with pytest.warns(RuntimeWarning) as caught:
        rec = json.loads(run(repo, say=seen.append, device_probe=lambda: {
            "pyannote": "cpu", "embeddings": None, "llm": "mixed: 60% of the model on the GPU"}).read_text("utf-8"))
    cfg = rec["config"]
    assert [d["actual"] for d in cfg["devices"].values()] == ["cuda", "cpu", "unknown", "mixed: 60% of the model on the GPU"]
    assert cfg["stages_not_on_cuda"] == ["pyannote", "embeddings", "llm"]
    assert len(caught) == 3 and sum(" WARNING " in m for m in seen) == 3


def test_a_speech_service_on_the_cpu_is_recorded_as_cpu(repo: Path, monkeypatch) -> None:
    monkeypatch.setattr(ScriptedStream, "BACKEND", "cpu")
    with pytest.warns(RuntimeWarning, match="whisper ran on cpu"):
        cfg = json.loads(run(repo).read_text(encoding="utf-8"))["config"]
    assert cfg["devices"]["whisper"]["actual"] == "cpu" and cfg["stages_not_on_cuda"] == ["whisper"]


def test_a_shared_machine_is_recorded_and_the_cli_makes_the_person_say(repo: Path) -> None:
    alone = json.loads(run(repo).read_text(encoding="utf-8"))["config"]
    assert alone["ran_alone"] is True and alone["latency_note"] is None
    shared = json.loads(run(repo, ran_alone=False).read_text(encoding="utf-8"))["config"]
    assert shared["ran_alone"] is False and "contention" in shared["latency_note"]
    with pytest.raises(SystemExit):
        e7.main(["run"])  # --alone or --not-alone is required


def test_device_names_and_ollama_placement() -> None:
    assert [e7.device_kind(x) for x in ("cuda", "cuda:0", "gpu:RTX", "cpu", None, "mixed")] == [
        "cuda", "cuda", "cuda", "cpu", "unknown", "mixed"]

    def ps(models):
        return lambda url: {"models": models}

    full = {"name": "qwen3.5:4b-q4_K_M", "size": 100, "size_vram": 100}
    assert e7.ollama_device("qwen3.5:4b-q4_K_M", ps([full])) == "cuda"
    assert e7.ollama_device("qwen3.5:4b-q4_K_M", ps([{**full, "size_vram": 0}])) == "cpu"
    assert e7.ollama_device("qwen3.5:4b-q4_K_M", ps([{**full, "size_vram": 60}])) == "mixed: 60% of the model on the GPU"
    assert e7.ollama_device("nomic-embed-text", ps([{"name": "nomic-embed-text:latest", "size": 5, "size_vram": 5}])) == "cuda"
    assert e7.ollama_device("nomic-embed-text", ps([full])) is None  # not loaded


def test_a_one_meeting_check_writes_no_run_record(repo: Path) -> None:
    scripts = [aligned_script("m1")]
    out = e7.run_e7(
        root=repo, speech=ScriptedSpeech(scripts), backend=TruthBackend(), embedder=NoEmbed(),
        make_assistant=make_assistant, extractor=None, models={}, prompt_versions={}, say=lambda m: None,
        step_ms=10_000, ran_alone=True, speech_address=e7.GPU_SPEECH_ADDRESS, device_probe=ALL_CUDA, only_meeting="m1")
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["not_a_record"] is True and doc["n"] == 2 and "wer" in doc["metrics"]["asr_pipeline"]
    assert doc["config"]["audio_source"]["method"] == e7.AUDIO_METHOD and out.parent == repo / "results/local"
    assert not (repo / "results/runs").exists()  # nothing check_run could pick up
    for bad in ({"only_meeting": "nope"}, {"only_meeting": "m1", "speedup": 5.0}, {"only_meeting": "m1", "rerun_of": "x"}):
        with pytest.raises(ValueError):
            run(repo, **bad)


def test_the_speech_address_defaults_to_the_gpu_service_and_is_recorded(repo: Path, monkeypatch) -> None:
    assert e7.GPU_SPEECH_ADDRESS == "127.0.0.1:50052"
    cfg = json.loads(run(repo).read_text(encoding="utf-8"))["config"]
    assert cfg["speech_address"] == "127.0.0.1:50052"
    assert json.loads(run(repo, speech_address="127.0.0.1:50051").read_text(encoding="utf-8"))["config"][
        "speech_address"] == "127.0.0.1:50051"
    # the CLI: the default is the GPU service, an explicit address wins, and what the service reports is what is recorded
    seen: list[str] = []
    monkeypatch.setattr(e7, "_run_real", lambda args: seen.append(args.speech_address) or 0)
    e7.main(["run", "--alone"])
    e7.main(["run", "--alone", "--speech-address", "127.0.0.1:50051"])
    assert seen == ["127.0.0.1:50052", "127.0.0.1:50051"]
    monkeypatch.setattr(ScriptedStream, "BACKEND", "cpu")  # a CPU service answering on the GPU address is still recorded as cpu
    with pytest.warns(RuntimeWarning, match="whisper ran on cpu"):
        cfg = json.loads(run(repo).read_text(encoding="utf-8"))["config"]
    assert cfg["speech_address"] == "127.0.0.1:50052" and cfg["devices"]["whisper"]["actual"] == "cpu"


def test_a_full_run_refuses_to_start_without_the_metrics_port(monkeypatch, capsys) -> None:
    """The record says the obs profile was on, so the metrics endpoint has to be served; the check comes first,
    before any model loads. The one-meeting check needs no metrics."""
    monkeypatch.delenv("CONTEXTLENS_METRICS_PORT", raising=False)
    assert e7.main(["run", "--alone"]) == 2
    assert "CONTEXTLENS_METRICS_PORT" in capsys.readouterr().err


# -- the stages the orchestrator runs on their own --------------------------------------------------------------------

def test_the_real_time_segment_is_the_first_part_of_one_meeting_with_its_presses_counted(repo: Path) -> None:
    rec = json.loads(run(repo, segment_ms=22_000).read_text(encoding="utf-8"))
    seg = rec["config"]["realtime_segment"]
    assert seg["presses"] == 4 and seg["basis"].startswith("the first 22 s") and seg["audio_s"] == 21  # turns end at 3 s steps
    assert rec["config"]["gpu_peak"] is None and rec["config"]["subset"] is None


def test_a_gpu_peak_probe_is_recorded(repo: Path) -> None:
    peak = {"baseline_mb": 400, "peak_mb": 5100, "used_by_run_mb": 4700}
    rec = json.loads(run(repo, gpu_peak=lambda: peak).read_text(encoding="utf-8"))
    assert rec["config"]["gpu_peak"] == peak


def test_several_meetings_check_writes_no_run_record(repo: Path) -> None:
    out = e7.run_e7(
        root=repo, speech=ScriptedSpeech([aligned_script("m1"), merged_script("m2")]), backend=TruthBackend(),
        embedder=NoEmbed(), make_assistant=make_assistant, extractor=None, models={}, prompt_versions={},
        say=lambda m: None, step_ms=10_000, ran_alone=True, speech_address=e7.GPU_SPEECH_ADDRESS,
        device_probe=ALL_CUDA, only_meetings=["m1", "m2"])
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["not_a_record"] is True and doc["n"] == 4 and doc["meetings"] == ["m1", "m2"]
    assert "shortest" in out.name and not (repo / "results/runs").exists()


def test_rerun_segment_records_before_and_after_for_the_segment_and_the_token_count(repo: Path) -> None:
    base = run(repo)
    SPEEDS.clear()
    path = e7.rerun_segment(
        root=repo, speech=ScriptedSpeech([aligned_script("m1")]), backend=TruthBackend(), embedder=NoEmbed(),
        make_assistant=make_assistant, extractor=FirstTurnExtractor(),
        models={"llm": "qwen3.5:4b-q4_K_M"}, prompt_versions={"team": "v1"}, baseline_id=base.stem,
        changes=["cap the context the memory agent reads"], step_ms=10_000, ran_alone=True,
        speech_address="127.0.0.1:50052", device_probe=ALL_CUDA, say=lambda m: None)
    rec = json.loads(path.read_text(encoding="utf-8"))
    cfg = rec["config"]
    assert SPEEDS == [1.0] and cfg["phase"] == "rerun_segment" and cfg["systems"] == ["clean"] and set(rec["metrics"]) == {"clean"}
    assert cfg["baseline_run_id"] == base.stem and cfg["changes"] == ["cap the context the memory agent reads"]
    assert cfg["bottlenecks"] == json.loads(base.read_text(encoding="utf-8"))["config"]["bottlenecks"]
    assert set(cfg["before_after"]) == {"memory_ms", "assistance_ms", "total", "quality"}
    assert cfg["before_after"]["quality"]["tokens_per_card"] == {"before": 50, "after": 50}
    assert cfg["before_after"]["total"]["p95"]["before"] is not None and cfg["before_after"]["total"]["p95"]["after"] is not None
    assert cfg["realtime_segment"]["meeting"] == "m1" and rec["n"] == 4
    assert check("E7", root=repo, min_n=4, split="dev") == []
    with pytest.raises(ValueError, match="not an E7 baseline"):
        e7.rerun_segment(root=repo, speech=ScriptedSpeech([]), backend=TruthBackend(), embedder=NoEmbed(),
                         make_assistant=make_assistant, extractor=None, models={}, prompt_versions={},
                         baseline_id=path.stem, changes=["again"], ran_alone=True, speech_address="x",
                         device_probe=ALL_CUDA, say=lambda m: None)


def test_the_cli_refuses_a_rerun_without_the_metrics_port_and_a_sped_up_shortest_run(monkeypatch, capsys) -> None:
    monkeypatch.delenv("CONTEXTLENS_METRICS_PORT", raising=False)
    assert e7.main(["rerun-segment", "--baseline", "x", "--change", "c", "--alone"]) == 2
    assert "CONTEXTLENS_METRICS_PORT" in capsys.readouterr().err
    assert e7.main(["run", "--alone", "--shortest", "5", "--speedup", "4"]) == 2
    assert "real time" in capsys.readouterr().err


# -- a run is never lost to a dirty tree, and finished work is kept -----------------------------------------------------

def test_the_progress_log_folder_is_ignored_by_git() -> None:
    assert subprocess.run(["git", "check-ignore", "-q", "results/logs/e7-run.log"], cwd=ROOT).returncode == 0


def test_every_writing_stage_refuses_a_dirty_tree_before_it_opens_anything(repo: Path) -> None:
    (repo / "results").mkdir(exist_ok=True)
    (repo / "results/notes.txt").write_text("untracked", encoding="utf-8")

    class Boom:
        async def open(self):
            raise AssertionError("the speech service was opened")

    with pytest.raises(RuntimeError, match="uncommitted"):
        e7.run_e7(root=repo, speech=Boom(), backend=TruthBackend(), embedder=NoEmbed(), make_assistant=make_assistant,
                  extractor=None, models={}, prompt_versions={}, ran_alone=True, speech_address="x",
                  device_probe=ALL_CUDA, say=lambda m: None)
    with pytest.raises(RuntimeError, match="uncommitted"):
        e7.rerun_segment(root=repo, speech=Boom(), backend=TruthBackend(), embedder=NoEmbed(),
                         make_assistant=make_assistant, extractor=None, models={}, prompt_versions={},
                         baseline_id="E7-x", changes=["c"], ran_alone=True, speech_address="x",
                         device_probe=ALL_CUDA, say=lambda m: None)


def test_the_cli_checks_the_tree_before_importing_any_model(monkeypatch, capsys) -> None:
    monkeypatch.setenv("CONTEXTLENS_METRICS_PORT", "9464")
    monkeypatch.setattr(e7, "is_dirty", lambda root: True)
    assert e7.main(["run", "--alone"]) == 2
    assert "uncommitted" in capsys.readouterr().err
    assert e7.main(["rerun-segment", "--baseline", "x", "--change", "c", "--alone"]) == 2
    assert "uncommitted" in capsys.readouterr().err


class FailsToOpen:
    """A speech service that works for `ok` streams, then the run is cut off."""

    def __init__(self, scripts, ok: int) -> None:
        self.inner, self.ok, self.opened = ScriptedSpeech(scripts), ok, 0

    async def open(self):
        self.opened += 1
        if self.opened > self.ok:
            raise e7.SpeechUnavailable("cut off")
        return await self.inner.open()


def run_with(repo: Path, speech, **kw) -> Path:
    return e7.run_e7(
        root=repo, speech=speech, backend=TruthBackend(), embedder=NoEmbed(), make_assistant=make_assistant,
        extractor=FirstTurnExtractor(), models={"llm": "x"}, prompt_versions={"team": "v1"}, speedup=1000.0,
        realtime_meeting="m1", step_ms=10_000, ran_alone=True, speech_address="x", device_probe=ALL_CUDA,
        say=lambda m: None, **kw)


def test_a_cut_off_run_resumes_at_the_same_commit_and_never_redoes_finished_meetings(repo: Path) -> None:
    with pytest.raises(e7.SpeechUnavailable):  # m1 done, m2 cut off
        run_with(repo, FailsToOpen([aligned_script("m1"), merged_script("m2")], ok=1))
    assert not (repo / "results/runs").exists()  # a partial run is never a result
    SPEEDS.clear()
    second = ScriptedSpeech([merged_script("m2"), aligned_script("m1")])  # no stream for m1's meeting pass
    path = run_with(repo, second)
    assert SPEEDS == [1000.0, 1.0]  # m2, then the segment: m1 was not replayed
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert rec["config"]["resumed_stages"] == ["m1"] and rec["n"] == 4
    assert check("E7", root=repo, min_n=4, split="dev") == []
    assert not list((repo / "results/partial/e7").glob("*.jsonl"))  # the record holds them now


def test_a_cut_off_real_time_segment_is_redone_whole_not_mixed(repo: Path) -> None:
    with pytest.raises(e7.SpeechUnavailable):  # both meetings done, the segment cut off
        run_with(repo, FailsToOpen([aligned_script("m1"), merged_script("m2"), aligned_script("m1")], ok=2))
    SPEEDS.clear()
    path = run_with(repo, ScriptedSpeech([aligned_script("m1")]))
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert SPEEDS == [1.0] and rec["config"]["resumed_stages"] == ["m1", "m2"]
    assert rec["config"]["realtime_segment"]["asr_lag_ms"]["n"] == 8  # one pass, from this run


def test_rows_saved_at_another_commit_or_setting_are_not_reused(repo: Path) -> None:
    with pytest.raises(e7.SpeechUnavailable):
        run_with(repo, FailsToOpen([aligned_script("m1"), merged_script("m2")], ok=1))
    (repo / "x.txt").write_text("a new commit", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "next")
    SPEEDS.clear()
    run_with(repo, ScriptedSpeech([aligned_script("m1"), merged_script("m2"), aligned_script("m1")]))
    assert SPEEDS == [1000.0, 1000.0, 1.0]  # a different commit: everything is redone


def test_progress_lines_carry_a_timestamp(repo: Path) -> None:
    lines: list[str] = []
    e7.run_e7(root=repo, speech=ScriptedSpeech([aligned_script("m1"), merged_script("m2"), aligned_script("m1")]),
              backend=TruthBackend(), embedder=NoEmbed(), make_assistant=make_assistant, extractor=None, models={},
              prompt_versions={}, speedup=1000.0, realtime_meeting="m1", step_ms=10_000, ran_alone=True,
              speech_address="x", device_probe=ALL_CUDA, say=lines.append)
    assert lines and all(re.match(r"\d\d:\d\d:\d\d \+\d+s ", ln) for ln in lines)
    assert any("speech" in ln and "m1" in ln for ln in lines) and any("pressed" in ln for ln in lines)


def test_clean_cards_take_a_finished_meeting_from_the_saved_rows(repo: Path) -> None:
    scs = meeting_scenarios("m1") + meeting_scenarios("m2")
    calls: list[str] = []

    def counting(store, retriever, memory):
        inner = CiteLast(store)

        class Counting:
            def answer(self, req, ctx):
                calls.append(req.conversation_id)
                return inner.answer(req, ctx)

        return Counting()

    settings = {"kind": "t"}
    first = e7.PartialRows(repo, settings)
    e7.clean_cards(scs[:2], NoEmbed(), counting, None, None, lambda m: None, first)  # m1 only
    assert calls == ["m1", "m1"]
    again = e7.PartialRows(repo, settings)
    preds = e7.clean_cards(scs, NoEmbed(), counting, None, None, lambda m: None, again)
    assert calls == ["m1", "m1", "m2", "m2"] and again.resumed == ["clean:m1"] and len(preds) == 4
