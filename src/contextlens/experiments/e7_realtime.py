"""E7: what do speech and speaker errors cost, and where does the time go?

AMI dev meetings are replayed as audio through the speech service and the speaker model, and the same questions
are asked three ways on the same meeting:

- `clean`: the AMI transcript with its true speakers;
- `asr_oracle_speakers`: the speech service's transcript, each turn given the true speaker (isolates transcription error);
- `asr_pipeline`: the speech service's transcript with the speaker model's labels, as the live backend has them.

Time passes in 60 s ticks. At each tick the new turns arrive (the pipeline's as `unknown`), memory items are
extracted from them, and the speaker model runs over its last two minutes and relabels turns, as in the live
backend. A press at its cut runs after that tick's speaker run, which is taken to have finished by then (its
duration is reported in `pipeline_timing`). A memory item keeps the speaker it was extracted with.

Rows cite clean turn ids so `check_run` can check them: each turn the card cites is replaced by the clean turn it
overlaps most in time (the nearest one when none overlaps). The speaker the card names is kept as it was.

Before any audio is fetched, `size-check` prints each file's size and stops on the disk rule. The questions come
from `datasets/ami-questions/dev.json` (see `build_ami_dev`).

Run (the person runs these; the real run needs a GPU turn, Ollama, the speech service and HF_TOKEN):
  .venv/Scripts/python -m contextlens.experiments.e7_realtime size-check
  .venv/Scripts/python -m contextlens.experiments.e7_realtime download
  .venv/Scripts/python -m contextlens.experiments.e7_realtime build-scenarios      # then commit the manifest
  .venv/Scripts/python -m contextlens.experiments.e7_realtime run
  .venv/Scripts/python -m contextlens.experiments.e7_realtime run --rerun BASELINE_RUN_ID --change "what changed"
"""
import argparse
import asyncio
import json
import shutil
import sys
import time
import urllib.request
import wave
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from contextlens.data.loaders import load_dataset
from contextlens.data.scenarios import load_scenarios
from contextlens.eval.check_run import committed_unchanged, recompute_comparison, recompute_intervals, recompute_metrics
from contextlens.eval.records import ROOT, SCENARIO_SOURCES, head_commit, split_hash, write_local_manifest, write_run
from contextlens.eval.series import base_name
from contextlens.eval.stats import percentile
from contextlens.memory.extract import MemoryStore
from contextlens.memory.store import InMemoryTranscriptStore
from contextlens.obs.tracing import Telemetry, TracedAssistant
from contextlens.replay.runner import Assistant, Context, Prediction
from contextlens.retrieval.hybrid import Retriever
from contextlens.schemas import ButtonRequest, Card, Conversation, EvidenceRef, MemoryItem, Scenario, Turn
from contextlens.speech.client import Segment, SpeechService, SpeechUnavailable
from contextlens.speech.diarize import BYTES_PER_MS, SAMPLE_RATE, Diarizer, WindowBackend

SPLIT, SOURCE = "dev", "ami"
QUESTIONS_REL = Path("datasets/ami-questions/dev.json")
AUDIO_REL = Path("datasets/raw/ami-audio")  # <meeting>.wav, git-ignored
AUDIO_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{m}/audio/{m}.Mix-Headset.wav"
MAX_SINGLE_BYTES = 15 * 10**9
MIN_FREE_BYTES = 40 * 10**9
STEP_MS = 60_000  # a tick; the speaker model sees two minutes, so each run overlaps the one before
TAIL_MS = 3_000  # silence after the last cut, so the speech service closes its last utterance
CHUNK_MS = 100
SYSTEMS = ("clean", "asr_oracle_speakers", "asr_pipeline")
ASR_SYSTEMS = SYSTEMS[1:]
KS = (1, 3, 5)
SAVED = {"recall@1", "recall@3", "recall@5", "mrr", "tokens_per_card", "speaker_accuracy", "label_accuracy",
         "abstain_when_insufficient", "answer_when_answerable", "wer", "der", "latency_p50_ms", "latency_p95_ms"}
ALL = 2**62
MAPPING_RULE = ("each cited ASR turn becomes the clean turn it overlaps most in time (the nearest by midpoint "
                "when none overlaps); the speaker the card names is kept")


# -- scenarios ---------------------------------------------------------------------------------------------------

def load_questions(root: Path = ROOT) -> tuple[list[dict[str, Any]], bool]:
    """The dev questions and whether a person reviewed their labels. The file is
    {"labels_reviewed": bool, "questions": [{"meeting", "as_of_seq", "question", "category",
    "gold_evidence_turn_ids", "gold_label", "gold_answer"}, ...]}."""
    raw = json.loads((root / QUESTIONS_REL).read_text(encoding="utf-8"))
    return raw["questions"], bool(raw["labels_reviewed"])


def meetings_needed(root: Path = ROOT) -> list[str]:
    return sorted({q["meeting"] for q in load_questions(root)[0]})


def build_ami_dev(root: Path = ROOT, raw_root: Path | None = None) -> int:
    """Turn the dev questions into scenarios on the clean AMI transcript (cut at the question's as_of_seq) and
    write the local scenario file and its manifest. Refuses a question outside AMI's dev meetings or with
    evidence after its as_of_seq."""
    dev = set(json.loads((root / "datasets/splits/ami.json").read_text(encoding="utf-8"))["splits"][SPLIT])
    meetings = {c.id: c for c in load_dataset("ami", raw_root or root / "datasets/raw")}
    questions, _ = load_questions(root)
    scenarios, count = [], {}
    for q in questions:
        m = q["meeting"]
        if m not in dev or m not in meetings:
            raise ValueError(f"{m} is not an AMI dev meeting in the local data")
        turns = [t for t in meetings[m].turns if t.seq <= q["as_of_seq"]]
        if len(turns) != q["as_of_seq"] + 1:
            raise ValueError(f"{m} has no turn {q['as_of_seq']}")
        ids = {t.id for t in turns}
        if not set(q["gold_evidence_turn_ids"]) <= ids:
            raise ValueError(f"{m}: gold evidence after as_of_seq {q['as_of_seq']} or not a turn")
        count[m] = count.get(m, 0) + 1
        sid = f"{m}-q{count[m]}"
        conv = Conversation(id=m, turns=turns, source="ami", split=SPLIT)
        scenarios.append(Scenario(
            id=sid, conversation=conv, request=ButtonRequest(conversation_id=m, as_of_seq=q["as_of_seq"],
                                                             question=q["question"]),
            category=q["category"], gold_evidence_turn_ids=q["gold_evidence_turn_ids"], gold_label=q["gold_label"],
            gold_answer=q["gold_answer"], origin="dataset"))
    write_local_manifest(SOURCE, {SPLIT: scenarios}, root)
    return len(scenarios)


# -- the size check and the download -------------------------------------------------------------------------------

def head_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            length = resp.headers.get("Content-Length")
    except OSError:
        return None
    return int(length) if length and length.isdigit() else None


def size_check(meetings: list[str], *, head: Callable[[str], int | None] = head_size,
               free: Callable[[], int] | None = None, audio_dir: Path | None = None) -> dict[str, Any]:
    """Size of each meeting's audio file, the free disk, and the reasons to stop: a size that can't be read, one
    file over 15 GB, or less than 40 GB free after the download. Fetches nothing."""
    target = audio_dir or ROOT / AUDIO_REL
    free_bytes = (free or (lambda: shutil.disk_usage(_existing(target)).free))()
    sizes = {m: head(AUDIO_URL.format(m=m)) for m in meetings}
    known = [s for s in sizes.values() if s is not None]
    total = sum(known)
    stop = [f"size of {m} unknown" for m, s in sizes.items() if s is None]
    stop += [f"{m} is {s / 1e9:.1f} GB, over the 15 GB single-download limit" for m, s in sizes.items()
             if s is not None and s > MAX_SINGLE_BYTES]
    if free_bytes - total < MIN_FREE_BYTES:
        stop.append(f"{(free_bytes - total) / 1e9:.1f} GB would be free after the download, under 40 GB")
    return {"sizes": sizes, "total_bytes": total, "free_bytes": free_bytes, "stop": stop}


def _existing(path: Path) -> Path:
    while not path.exists():
        path = path.parent
    return path


def _fetch(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def download(meetings: list[str], audio_dir: Path, *, head: Callable[[str], int | None] = head_size,
             free: Callable[[], int] | None = None, fetch: Callable[[str, Path], None] = _fetch) -> list[str]:
    """Fetch the meetings' audio that is not already there, after the size check; any stop reason means nothing
    is fetched and RuntimeError is raised. Each file is written beside its target and renamed when whole."""
    report = size_check(meetings, head=head, free=free, audio_dir=audio_dir)
    if report["stop"]:
        raise RuntimeError("size check says stop: " + "; ".join(report["stop"]))
    audio_dir.mkdir(parents=True, exist_ok=True)
    got = []
    for m in meetings:
        dest = audio_dir / f"{m}.wav"
        if dest.exists():
            continue
        tmp = dest.with_name(f".{dest.name}.tmp")
        try:
            fetch(AUDIO_URL.format(m=m), tmp)
            tmp.replace(dest)
        finally:
            tmp.unlink(missing_ok=True)
        got.append(m)
    return got


# -- speech ---------------------------------------------------------------------------------------------------------

def read_wav(path: Path) -> bytes:
    """16 kHz mono 16-bit PCM from a WAV file; any other format is refused rather than converted."""
    with wave.open(str(path), "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise ValueError(f"{path.name} is not 16 kHz mono 16-bit")
        return w.readframes(w.getnframes())


async def transcribe(speech: SpeechService, pcm: bytes, *, speedup: float = 1.0,
                     settle_s: float = 10.0) -> tuple[list[Segment], list[int], float]:
    """Stream the audio at `speedup` times real time. Returns the final segments, how late each arrived after its
    audio was sent (ms), and the wall seconds taken. The service failing before the audio is all sent raises."""
    stream = await speech.open()
    segments: list[Segment] = []
    lags: list[int] = []
    t0 = time.perf_counter()
    last = t0
    failure: list[SpeechUnavailable] = []

    async def read() -> None:
        nonlocal last
        try:
            async for seg in stream.segments():
                now = time.perf_counter()
                segments.append(seg)
                lags.append(max(0, int((now - t0) * 1000 - seg.end_ms / speedup)))
                last = now
        except SpeechUnavailable as exc:
            failure.append(exc)

    reader = asyncio.create_task(read())
    try:
        step = CHUNK_MS * BYTES_PER_MS
        for i, at in enumerate(range(0, len(pcm), step), start=1):
            stream.send(pcm[at : at + step])
            await asyncio.sleep(max(0.0, t0 + i * CHUNK_MS / 1000 / speedup - time.perf_counter()))
            if failure:
                raise failure[0]
        last = max(last, time.perf_counter())
        while time.perf_counter() - last < settle_s and not failure:
            await asyncio.sleep(min(0.2, settle_s))
        if failure:
            raise failure[0]
    finally:
        reader.cancel()
        await stream.close()
    return sorted(segments, key=lambda s: (s.start_ms, s.end_ms)), lags, time.perf_counter() - t0


# -- one meeting, three ways -------------------------------------------------------------------------------------

class Extractor(Protocol):
    def update(self, new_turns: list[Turn], existing: list[MemoryItem]) -> list[MemoryItem]: ...


MakeAssistant = Callable[[InMemoryTranscriptStore, Retriever, MemoryStore], Assistant]


class Session:
    """What the live backend holds for one conversation: its turns, search index and memory, and the team."""

    def __init__(self, name: str, meeting: str, embedder: Any, make_assistant: MakeAssistant,
                 extractor: Extractor | None, telemetry: Telemetry | None) -> None:
        self.name, self.meeting, self.extractor = name, meeting, extractor
        self.store, self.memory = InMemoryTranscriptStore(), MemoryStore()
        inner = make_assistant(self.store, Retriever(self.store, embedder), self.memory)
        self.traced = TracedAssistant(inner, telemetry) if telemetry else None
        self.assistant: Assistant = self.traced or inner
        self.memory_errors = 0

    def ingest(self, turns: list[Turn]) -> None:
        for t in turns:
            self.store.add(t)
        if self.extractor and turns:
            try:
                self.memory.add(self.extractor.update(turns, self.memory.items(self.meeting, ALL)))
            except Exception:  # background work in the live backend: a failed step leaves no items
                self.memory_errors += 1

    def press(self, req: ButtonRequest) -> tuple[Card, dict[str, int], bool]:
        ctx = Context(store=self.store, as_of_seq=req.as_of_seq)
        start = time.perf_counter()
        try:
            card, errored = self.assistant.answer(req, ctx), False
        except Exception as exc:
            card = Card(mode="recall", status="abstained", findings=[],
                        missing=[f"error: {type(exc).__name__}: {exc}"])
            errored = True
        return card, {**ctx.stage_ms, "total": int((time.perf_counter() - start) * 1000)}, errored


def cut_ms(s: Scenario) -> int:
    """When the question's last turn has been said."""
    return max(t.end_ms for t in s.conversation.turns if t.seq <= s.request.as_of_seq)


def _overlap(a: Turn | Segment, b: Turn | Segment) -> int:
    return max(0, min(a.end_ms, b.end_ms) - max(a.start_ms, b.start_ms))


def nearest_clean(span: Turn | Segment, clean: list[Turn]) -> Turn:
    mid = (span.start_ms + span.end_ms) / 2
    return max(clean, key=lambda c: (_overlap(span, c), -abs((c.start_ms + c.end_ms) / 2 - mid), -c.seq))


def oracle_speaker(seg: Segment, clean: list[Turn]) -> str:
    best = max(clean, key=lambda c: (_overlap(seg, c), -c.seq), default=None)
    return best.speaker if best is not None and _overlap(seg, best) > 0 else "unknown"


def _ids(card: Card) -> list[str]:
    seen: dict[str, None] = {}
    for f in card.findings:
        for e in f.evidence:
            seen.setdefault(e.turn_id)
    return list(seen)


def with_clean_ids(card: Card, to_clean: dict[str, str]) -> Card:
    def fix(e: EvidenceRef) -> EvidenceRef:
        return e.model_copy(update={"turn_id": to_clean.get(e.turn_id, e.turn_id)})

    return card.model_copy(update={"findings": [f.model_copy(update={"evidence": [fix(e) for e in f.evidence]})
                                                for f in card.findings]})


def stale_unknown_items(memory: MemoryStore, turns: list[Turn]) -> list[MemoryItem]:
    """Items extracted while their turns were `unknown` whose turns have since been given speakers."""
    now = {t.id: t.speaker for t in turns}
    return [m for m in memory.items(turns[0].conversation_id, ALL)
            if m.speaker == "unknown" and any(now.get(i, "unknown") != "unknown" for i in m.source_turn_ids)] if turns else []


def replay_meeting(
    scs: list[Scenario], pcm: bytes, segments: list[Segment], *, backend: WindowBackend, embedder: Any,
    make_assistant: MakeAssistant, extractor: Extractor | None, telemetry: Telemetry | None,
    step_ms: int = STEP_MS, say: Callable[[str], None] = print,
) -> tuple[dict[str, dict[str, Prediction]], dict[str, Any]]:
    """Rows of each system for this meeting's scenarios, and its timing and error-analysis numbers."""
    meeting = scs[0].conversation.id
    scs = sorted(scs, key=cut_ms)
    cuts = {s.id: cut_ms(s) for s in scs}
    clean_turns = max((s.conversation.turns for s in scs), key=len)
    last_cut = max(cuts.values())
    boundaries = sorted({*range(step_ms, last_cut, step_ms), *cuts.values(), last_cut})
    sessions = {n: Session(n, meeting, embedder, make_assistant, extractor, telemetry) for n in SYSTEMS}
    diarizer = Diarizer(meeting, backend)
    rows: dict[str, dict[str, Prediction]] = {n: {} for n in SYSTEMS}
    diar_ms: list[int] = []
    oracle_n = 0
    stale_cited = stale_total = 0
    prev = -1
    for b in boundaries:
        new_segs = [g for g in segments if prev < g.end_ms <= b]
        sessions["clean"].ingest([t for t in clean_turns if prev < t.end_ms <= b])
        oracle = []
        for g in new_segs:
            oracle.append(Turn(id=f"{meeting}-o{oracle_n}", conversation_id=meeting, seq=oracle_n,
                               speaker=oracle_speaker(g, clean_turns), speaker_confidence=1.0, text=g.text.strip(),
                               start_ms=g.start_ms, end_ms=g.end_ms, source="asr"))
            oracle_n += 1
        sessions["asr_oracle_speakers"].ingest(oracle)
        diarizer.feed(pcm[max(prev, 0) * BYTES_PER_MS : b * BYTES_PER_MS])
        sessions["asr_pipeline"].ingest(diarizer.assign(new_segs))  # speakers are `unknown` until the run below
        window, start = diarizer.window()
        if window:
            t0 = time.perf_counter()
            spans = backend.spans(window)
            diar_ms.append(int((time.perf_counter() - t0) * 1000))
            store = sessions["asr_pipeline"].store
            for seq, who, conf in diarizer.speaker_updates(
                    spans, start, start + len(window) // BYTES_PER_MS, store.turns(meeting, ALL)):
                store.relabel(meeting, seq, who, conf)
        for sc in (s for s in scs if cuts[s.id] == b):
            for name, sess in sessions.items():
                if name == "clean":
                    as_of, turns = sc.request.as_of_seq, []
                else:
                    turns = sess.store.turns(meeting, ALL)
                    as_of = turns[-1].seq if turns else -1
                card, stages, errored = sess.press(sc.request.model_copy(update={"conversation_id": meeting,
                                                                                  "as_of_seq": as_of}))
                row = Prediction(scenario_id=sc.id, card=card, ranked_turn_ids=_ids(card), stage_latency_ms=stages,
                                 errored=errored)
                if name != "clean":
                    visible = [t for t in turns if t.seq <= as_of]
                    clean = [t for t in sc.conversation.turns if t.seq <= sc.request.as_of_seq]
                    mapping = {t.id: nearest_clean(t, clean).id for t in visible} if clean else {}
                    cited = set(_ids(card))
                    if name == "asr_pipeline":
                        stale = stale_unknown_items(sess.memory, visible)
                        stale_total += len(stale)
                        stale_cited += any(cited & set(m.source_turn_ids) for m in stale)
                    card = with_clean_ids(card, mapping)
                    row = row.model_copy(update={
                        "card": card, "ranked_turn_ids": _ids(card), "asr_text": " ".join(t.text for t in visible),
                        **({"speaker_spans": [(t.start_ms, t.end_ms, t.speaker) for t in visible]}
                           if name == "asr_pipeline" else {})})
                rows[name][sc.id] = row
            say(f"  {sc.id} pressed at {b / 1000:.0f} s")
        prev = b
    final = sessions["asr_pipeline"].store.turns(meeting, ALL)
    stats = {
        "diar_ms": diar_ms,
        "memory_errors": sum(s.memory_errors for s in sessions.values()),
        "memory_items": len(sessions["asr_pipeline"].memory.items(meeting, ALL)),
        "stale_unknown_items": len(stale_unknown_items(sessions["asr_pipeline"].memory, final)),
        "presses_citing_stale_unknown_items": stale_cited,
        "turns": len(final), "turns_still_unknown": sum(t.speaker == "unknown" for t in final),
    }
    for s in sessions.values():
        if s.traced:
            s.traced.flush()
    return rows, stats


# -- bottlenecks, before and after -----------------------------------------------------------------------------------

def top_bottlenecks(metrics: dict[str, float | None], n: int = 2) -> list[dict[str, Any]]:
    """The n press stages with the highest p95, from a system's saved metrics. Only stages timed in ms count;
    `total` and token counts do not."""
    stages = []
    for key, p95 in metrics.items():
        if key.startswith("latency_p95_ms[stage=") and p95 is not None:
            name = key[len("latency_p95_ms[stage=") : -1]
            if name.endswith("_ms"):
                stages.append({"stage": name, "p95_ms": p95, "p50_ms": metrics.get(f"latency_p50_ms[stage={name}]")})
    return sorted(stages, key=lambda s: -s["p95_ms"])[:n]


def before_after(before: dict[str, float | None], after: dict[str, float | None],
                 stages: list[str]) -> dict[str, dict[str, dict[str, float | None]]]:
    """Per stage (and `total`), p50 and p95 before and after, then the two quality numbers beside them."""
    out: dict[str, dict[str, dict[str, float | None]]] = {}
    for stage in [*stages, "total"]:
        suffix = "" if stage == "total" else f"[stage={stage}]"
        out[stage] = {f"p{q}": {"before": before.get(f"latency_p{q}_ms{suffix}"), "after": after.get(f"latency_p{q}_ms{suffix}")}
                      for q in (50, 95)}
    out["quality"] = {k: {"before": before.get(k), "after": after.get(k)} for k in ("recall@5", "tokens_per_card")}
    return out


# -- the run --------------------------------------------------------------------------------------------------------

def _load_dev(root: Path) -> tuple[list[Scenario], str]:
    manifest_rel = SCENARIO_SOURCES[SOURCE][0]
    entry = json.loads((root / manifest_rel).read_text(encoding="utf-8"))["splits"][SPLIT]
    digest = split_hash(SPLIT, root, SOURCE)
    if digest != entry["hash"]:
        raise RuntimeError("the AMI scenario file does not match its manifest; run build-scenarios and commit")
    return load_scenarios(SPLIT, root / SCENARIO_SOURCES[SOURCE][1]), digest


def _summary(values: list[int] | list[float]) -> dict[str, float | int | None]:
    return {"n": len(values), "p50": percentile(list(values), 50) if values else None,
            "p95": percentile(list(values), 95) if values else None}


def run_e7(
    *, root: Path = ROOT, speech: SpeechService, backend: WindowBackend, embedder: Any,
    make_assistant: MakeAssistant, extractor: Extractor | None, models: dict[str, str],
    prompt_versions: dict[str, str], telemetry: Telemetry | None = None, audio_dir: Path | None = None,
    rerun_of: str | None = None, changes: list[str] | None = None, speedup: float = 1.0, settle_s: float = 10.0,
    step_ms: int = STEP_MS, say: Callable[[str], None] = print,
) -> Path:
    """Replay every dev scenario three ways and write one E7 record. A first run is the baseline and names
    its two slowest press stages; a rerun (`rerun_of` a baseline's run id, with the `changes` made) must be on the
    same scenarios and records the same stages before and after. Nothing is written if anything fails."""
    manifest_rel = SCENARIO_SOURCES[SOURCE][0].as_posix()
    if not committed_unchanged({"git_commit": head_commit(root)}, root, manifest_rel):
        raise RuntimeError("the AMI manifest must be committed before the run")
    scenarios, digest = _load_dev(root)
    baseline = None
    if rerun_of:
        if not [c for c in (changes or []) if c.strip()]:
            raise ValueError("a rerun must list what changed")
        baseline = json.loads((root / "results/runs" / f"{rerun_of}.json").read_text(encoding="utf-8"))
        if baseline["experiment"] != "E7" or baseline["config"].get("phase") != "baseline":
            raise ValueError(f"{rerun_of} is not an E7 baseline record")
        if baseline["dataset_hash"] != digest:
            raise ValueError("the baseline was run on different scenarios")
    elif changes:
        raise ValueError("changes belong to a rerun")
    audio = audio_dir or root / AUDIO_REL
    by_meeting: dict[str, list[Scenario]] = {}
    for sc in scenarios:
        by_meeting.setdefault(sc.conversation.id, []).append(sc)
    missing = [m for m in by_meeting if not (audio / f"{m}.wav").is_file()]
    if missing:
        raise FileNotFoundError(f"no audio for {missing}; run download")
    _, reviewed = load_questions(root)
    started = datetime.now(timezone.utc).isoformat()
    rows: dict[str, dict[str, Prediction]] = {n: {} for n in SYSTEMS}
    stats: dict[str, dict[str, Any]] = {}
    asr_lags: list[int] = []
    for n, (meeting, scs) in enumerate(by_meeting.items(), 1):
        say(f"{meeting}: {len(scs)} questions ({n}/{len(by_meeting)})")
        pcm = read_wav(audio / f"{meeting}.wav")
        cut = max(cut_ms(s) for s in scs)
        stream_pcm = pcm[: cut * BYTES_PER_MS] + bytes(TAIL_MS * BYTES_PER_MS)
        segments, lags, wall_s = asyncio.run(transcribe(speech, stream_pcm, speedup=speedup, settle_s=settle_s))
        asr_lags += lags
        got, stats[meeting] = replay_meeting(
            scs, pcm, [g for g in segments if g.end_ms <= cut], backend=backend, embedder=embedder,
            make_assistant=make_assistant, extractor=extractor, telemetry=telemetry, step_ms=step_ms, say=say)
        stats[meeting].update(audio_s=cut / 1000, asr_wall_s=wall_s, asr_segments=len(segments))
        for name in SYSTEMS:
            rows[name].update(got[name])
    preds = {name: [rows[name][s.id] for s in scenarios] for name in SYSTEMS}
    # The speaker model names speakers speaker_1, speaker_2, ..., so comparing its names with AMI's says nothing;
    # `der` (which matches the names first) is its measure.
    metrics = {name: {k: v for k, v in recompute_metrics(scenarios, p, KS).items()
                      if base_name(k) in SAVED and ("[" not in k or "[stage=" in k)
                      and not (name == "asr_pipeline" and k == "speaker_accuracy")} for name, p in preds.items()}
    intervals: dict[str, dict[str, tuple[float, float]]] = {}
    for name, p in preds.items():
        wanted = ["recall@5"] + (["wer"] if name in ASR_SYSTEMS else [])
        found = {k: iv for k, iv in recompute_intervals(scenarios, p, wanted).items() if iv}
        if found:
            intervals[name] = found
    comparisons = []
    for a, b in (("asr_oracle_speakers", "clean"), ("asr_pipeline", "asr_oracle_speakers"), ("asr_pipeline", "clean")):
        if c := recompute_comparison(scenarios, preds[a], preds[b], "recall@5"):
            comparisons.append({"a": a, "b": b, **c})
    pipeline = metrics["asr_pipeline"]
    all_diar = [x for s in stats.values() for x in s["diar_ms"]]
    config: dict[str, Any] = {
        "experiment": "E7", "phase": "rerun" if rerun_of else "baseline", "systems": list(SYSTEMS),
        "step_ms": step_ms, "speedup": speedup, "mapping_rule": MAPPING_RULE, "obs_profile": telemetry is not None,
        "pipeline_timing": {
            "asr_lag_ms": _summary(asr_lags), "diarization_run_ms": _summary(all_diar),
            "asr_real_time_factor": sum(s["asr_wall_s"] * speedup for s in stats.values())
            / max(1e-9, sum(s["audio_s"] for s in stats.values())),
        },
        "error_analysis": {
            "memory_items_extracted_while_unknown_now_stale": sum(s["stale_unknown_items"] for s in stats.values()),
            "memory_items": sum(s["memory_items"] for s in stats.values()),
            "presses_citing_such_an_item": sum(s["presses_citing_stale_unknown_items"] for s in stats.values()),
            "turns": sum(s["turns"] for s in stats.values()),
            "turns_still_unknown": sum(s["turns_still_unknown"] for s in stats.values()),
            "memory_errors": sum(s["memory_errors"] for s in stats.values()),
            "errored_presses": {n: sum(p.errored for p in ps) for n, ps in preds.items()},
        },
        "per_meeting": {m: {k: v for k, v in s.items() if k != "diar_ms"} for m, s in stats.items()},
        "claude_calls": 0, "labels_reviewed": reviewed, "started_at": started,
    }
    if baseline is None:
        config["bottlenecks"] = top_bottlenecks(pipeline)
    else:
        named = baseline["config"]["bottlenecks"]
        config.update(baseline_run_id=rerun_of, bottlenecks=named, changes=changes)
        config["before_after"] = before_after(baseline["metrics"]["asr_pipeline"], pipeline, [b["stage"] for b in named])
    attempts = [{"setting": n, "ok": True, "errored_presses": sum(p.errored for p in preds[n])} for n in SYSTEMS]
    return write_run("E7", SPLIT, config, models, prompt_versions, metrics, intervals, attempts, preds, digest,
                     comparisons=comparisons, scenario_source=SOURCE, root=root)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="e7_realtime")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("size-check", "download", "build-scenarios"):
        sub.add_parser(name)
    r = sub.add_parser("run")
    r.add_argument("--rerun", metavar="BASELINE_RUN_ID")
    r.add_argument("--change", action="append", default=[])
    r.add_argument("--speedup", type=float, default=1.0)
    args = ap.parse_args(argv)
    if args.cmd == "build-scenarios":
        print(build_ami_dev())
        return 0
    meetings = meetings_needed()
    if args.cmd == "size-check":
        report = size_check(meetings)
        for m, s in report["sizes"].items():
            print(f"{m}: {'unknown' if s is None else f'{s / 1e6:.0f} MB'}")
        print(f"total {report['total_bytes'] / 1e9:.2f} GB, free {report['free_bytes'] / 1e9:.1f} GB")
        for reason in report["stop"]:
            print("STOP", reason)
        return 2 if report["stop"] else 0
    if args.cmd == "download":
        print(download(meetings, ROOT / AUDIO_REL))
        return 0
    return _run_real(args)


def _run_real(args: argparse.Namespace) -> int:
    """The real parts, as the live backend builds them. Unloads the Ollama models before returning, so the GPU
    turn can be released."""
    from contextlens.api import production
    from contextlens.llm.client import unload_ollama
    from contextlens.speech.client import GrpcSpeech
    from contextlens.speech.diarize import PyannoteBackend

    telemetry = Telemetry()
    production.serve_metrics_if_asked(telemetry)
    embedder = production.local_embedder()
    llms = production.local_llms()

    def make(store: InMemoryTranscriptStore, retriever: Retriever, memory: MemoryStore) -> Assistant:
        return production.TeamPerPress(store, retriever, memory, llms, production.DEADLINE_S)

    try:
        path = run_e7(
            speech=GrpcSpeech(), backend=PyannoteBackend(), embedder=embedder, make_assistant=make,
            extractor=production.MemoryExtractor(llms["memory"]), telemetry=telemetry,
            models={"llm": production.MODEL, "embedder": "nomic-embed-text",
                    "speech": "whisper.cpp ggml-tiny.en", "diarizer": PyannoteBackend.MODEL},
            prompt_versions={"team": production.PROMPT_VERSION}, rerun_of=args.rerun, changes=args.change,
            speedup=args.speedup)
    finally:
        for model in (production.MODEL, embedder.model):
            unload_ollama(model, production.OLLAMA_URL)
    print(path.as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())
