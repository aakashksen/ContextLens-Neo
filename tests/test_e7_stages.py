"""E7's separate stages: the 4x readiness check, the 10-minute real-time segment, and the fallback. Fakes only."""
import asyncio
import json
import wave
from pathlib import Path

import pytest

from contextlens.experiments import e7_realtime as e7
from contextlens.schemas import ButtonRequest, Conversation, Scenario, Turn
from contextlens.speech.client import Segment
from contextlens.speech.diarize import BYTES_PER_MS, SAMPLE_RATE


def turns(m: str, n: int, ms: int = 3000) -> list[Turn]:
    return [Turn(id=f"{m}:t{i}", conversation_id=m, seq=i, speaker="A", speaker_confidence=1.0, text=f"turn {i}",
                 start_ms=i * ms, end_ms=(i + 1) * ms, source="dataset") for i in range(n)]


def scenarios(m: str, n: int, as_ofs: tuple[int, ...]) -> list[Scenario]:
    out = []
    for k, a in enumerate(as_ofs, 1):
        conv = Conversation(id=m, turns=turns(m, n)[: a + 1], source="ami", split="dev")
        out.append(Scenario(id=f"{m}-q{k}", conversation=conv, request=ButtonRequest(
            conversation_id=m, as_of_seq=a, question=f"question {k}"), category="recall",
            gold_evidence_turn_ids=[f"{m}:t1"], gold_label=None, gold_answer="x", origin="dataset"))
    return out


def test_segment_presses_each_question_at_every_tick_and_at_the_end_of_ten_minutes():
    scs = scenarios("m", 300, (150, 250))  # 300 turns of 3 s = 15 minutes
    seg = e7.segment_scenarios(scs, segment_ms=600_000, every_ms=60_000)
    as_ofs = sorted({s.request.as_of_seq for s in seg})
    assert as_ofs == [39 + 20 * k for k in range(8)] + [199]  # ticks at 2..9 min (turn ending by then), then 10 min
    assert len(seg) == 2 * len(as_ofs) and len({s.id for s in seg}) == len(seg)
    assert {s.request.question for s in seg} == {"question 1", "question 2"}
    assert all(max(t.end_ms for t in s.conversation.turns) <= 600_000 for s in seg)
    assert all(len(s.conversation.turns) == s.request.as_of_seq + 1 for s in seg)
    assert all(not s.gold_evidence_turn_ids and s.gold_label is None for s in seg)  # latency only, nothing scored


def test_a_meeting_shorter_than_the_segment_is_pressed_to_its_last_turn():
    seg = e7.segment_scenarios(scenarios("m", 8, (5, 7)), segment_ms=600_000, every_ms=10_000)
    assert sorted({s.request.as_of_seq for s in seg}) == [5, 7]


def lags_growing(n: int) -> list[int]:
    return [1000 + 2000 * i for i in range(n)]


def test_keeps_up_when_lag_stays_flat_and_not_when_it_grows():
    flat = [1500, 900, 2100, 1200, 1800, 1000, 2000, 1300, 1700]
    ok, why = e7.keeps_up(flat)
    assert ok and why["last_third_p50_ms"] <= why["first_third_p50_ms"] + e7.MAX_LAG_GROWTH_MS
    ok, why = e7.keeps_up(lags_growing(30))
    assert not ok and why["last_third_p50_ms"] > why["first_third_p50_ms"] + e7.MAX_LAG_GROWTH_MS
    assert not e7.keeps_up([1000] * 8 + [e7.MAX_FINAL_LAG_MS + 1])[0]  # flat, but the end is far behind
    with pytest.raises(ValueError, match="too few"):
        e7.keeps_up([1, 2, 3])


def write_wav(path: Path, seconds: float) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(bytes(int(seconds * SAMPLE_RATE) * 2))


def test_the_shortest_meetings_are_picked_by_audio_length(tmp_path: Path):
    for m, s in (("a", 30), ("b", 10), ("c", 20), ("d", 5), ("e", 40), ("f", 15)):
        write_wav(tmp_path / f"{m}.wav", s)
    assert e7.shortest_meetings(tmp_path, ["a", "b", "c", "d", "e", "f"], 5) == ["d", "b", "f", "c", "a"]


class Stream:
    def __init__(self, lag_s: float) -> None:
        self.lag_s, self.sent = lag_s, 0

    def send(self, pcm: bytes) -> None:
        self.sent += len(pcm)

    async def compute_backend(self):
        return "gpu:Fake"

    async def segments(self):
        i = 0
        while True:
            while (i + 1) * 3000 * BYTES_PER_MS <= self.sent:
                yield Segment(text=f"w{i}", start_ms=i * 3000, end_ms=(i + 1) * 3000)
                i += 1
            await asyncio.sleep(0.002)

    async def close(self):
        pass


class Speech:
    async def open(self):
        return Stream(0)


def test_readiness_reports_the_plan_for_a_speech_service_that_keeps_up(tmp_path: Path):
    write_wav(tmp_path / "m.wav", 60)
    out = asyncio.run(e7.readiness(Speech(), tmp_path, "m", speedup=200.0, audio_s=60, settle_s=0.05))
    assert out["keeps_up"] is True and out["speedup"] == 200.0 and out["segments"] == 21
    assert out["plan"].startswith("all meetings at 200")
    assert out["compute_backend"] == "gpu:Fake" and "fallback" not in out


def test_readiness_names_the_fallback_when_speech_cannot_keep_up(tmp_path: Path):
    for m, s in (("a", 30), ("b", 10), ("c", 20), ("d", 5), ("e", 40), ("f", 15)):
        write_wav(tmp_path / f"{m}.wav", s)

    class Slow(Speech):
        async def open(self):
            s = Stream(0)
            real = s.segments

            async def late():
                n = 0
                async for seg in real():
                    await asyncio.sleep(0.03 * n)  # each segment later than the last
                    n += 1
                    yield seg

            s.segments = late
            return s

    out = asyncio.run(e7.readiness(Slow(), tmp_path, "a", speedup=200.0, audio_s=30, settle_s=1.0, max_growth_ms=100,
                                   needed=["a", "b", "c", "d", "e", "f"], question_counts=dict.fromkeys("abcdef", 3)))
    assert out["keeps_up"] is False
    assert out["fallback"]["meetings"] == ["d", "b", "f", "c", "a"] and out["fallback"]["speedup"] == 1.0
    assert "30-question rule" in out["fallback"]["note"] and "15 questions" in out["fallback"]["note"]
    json.dumps(out)


def test_gpu_sampler_reports_baseline_peak_and_what_the_run_added():
    import time

    from contextlens.experiments.e7_realtime import GpuSampler

    seen = iter([400, 900, 5100, 4800, 600] + [600] * 100)
    sampler = GpuSampler(read=lambda: next(seen), every_s=0.001)
    sampler.start()
    time.sleep(0.1)
    assert sampler.stop() == {"baseline_mb": 400, "peak_mb": 5100, "used_by_run_mb": 4700}


def test_gpu_sampler_says_nothing_when_the_gpu_cannot_be_read():
    from contextlens.experiments.e7_realtime import GpuSampler

    sampler = GpuSampler(read=lambda: None, every_s=0.001)
    sampler.start()
    assert sampler.stop() is None


def test_ready_command_writes_its_answer_under_results_local(tmp_path, monkeypatch, capsys):
    write_wav(tmp_path / "m.wav", 60)
    monkeypatch.setattr(e7, "ROOT", tmp_path)

    async def fake(speech, audio_dir, meeting, **kw):
        return {"meeting": meeting, "keeps_up": True, "plan": "p", "speedup": kw["speedup"]}

    monkeypatch.setattr(e7, "readiness", fake)
    monkeypatch.setattr(e7, "_open_speech", lambda addr: Speech())
    assert e7.main(["ready", "--meeting", "m", "--audio-dir", str(tmp_path)]) == 0
    path = Path(capsys.readouterr().out.splitlines()[0])
    assert path.parent == tmp_path / "results/local" and json.loads(path.read_text())["keeps_up"] is True

    async def slow(speech, audio_dir, meeting, **kw):
        return {"meeting": meeting, "keeps_up": False, "plan": "the shortest 5 meetings in real time"}

    monkeypatch.setattr(e7, "readiness", slow)
    assert e7.main(["ready", "--meeting", "m", "--audio-dir", str(tmp_path)]) == 3


def test_transcribe_does_not_wait_for_the_backend_name_before_sending_audio():
    """gRPC sends a server's initial metadata with its first reply, so asking for it before any audio is sent never
    returns against the real service (seen on the real GPU service: `ready` hung)."""
    from contextlens.speech.client import Segment

    class LateMetadata:
        sent = 0

        def send(self, pcm: bytes) -> None:
            self.sent += len(pcm)

        async def compute_backend(self):
            while not self.sent:  # the header only comes once the service has been given audio
                await asyncio.sleep(0.005)
            await asyncio.sleep(0.05)
            return "gpu:CUDA0"

        async def segments(self):
            while self.sent < 3000 * BYTES_PER_MS:
                await asyncio.sleep(0.005)
            yield Segment(text="hi", start_ms=0, end_ms=3000)
            await asyncio.sleep(3600)

        async def close(self):
            pass

    class Speech:
        async def open(self):
            return LateMetadata()

    class Never(LateMetadata):
        async def compute_backend(self):
            await asyncio.sleep(3600)

    class NeverSpeech:
        async def open(self):
            return Never()

    info: dict = {}
    segs, _, _ = asyncio.run(asyncio.wait_for(
        e7.transcribe(Speech(), bytes(5000 * BYTES_PER_MS), speedup=100.0, settle_s=0.2, info=info), 10))
    assert len(segs) == 1 and info["compute_backend"] == "gpu:CUDA0"
    info = {}
    asyncio.run(asyncio.wait_for(
        e7.transcribe(NeverSpeech(), bytes(5000 * BYTES_PER_MS), speedup=100.0, settle_s=0.2, info=info), 10))
    assert info["compute_backend"] is None  # never reported: recorded as unknown, not waited for


# -- the speech service's voice gate is a fixed level; quiet rebuilt audio fell under it ------------------------------------

def tone(seconds: float, rms: float, silence_s: float = 0.0):
    import numpy as np

    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    wave_ = np.sin(2 * np.pi * 220 * t) * rms * 2**0.5  # a sine of this RMS
    x = np.concatenate([wave_, np.zeros(int(silence_s * SAMPLE_RATE))])
    return (x * 32768).astype(np.int16).tobytes()


def speech_median_rms(pcm: bytes) -> float:
    import numpy as np

    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768
    n = len(a) // 320
    r = np.sqrt((a[: n * 320].reshape(n, 320) ** 2).mean(1))
    return float(np.median(r[r > e7.LEVEL_FLOOR]))


def test_quiet_speech_is_brought_to_twice_the_voice_gate_without_touching_silence():
    quiet = tone(5, 0.004, silence_s=5)  # speech at 0.004 RMS is under the service's 0.01 gate
    gain = e7.level_gain(quiet)
    leveled = e7.apply_gain(quiet, gain)
    assert gain == pytest.approx(e7.LEVEL_TARGET / 0.004, rel=0.05)
    assert speech_median_rms(leveled) == pytest.approx(e7.LEVEL_TARGET, rel=0.05)
    assert leveled[len(leveled) // 2 + 1000 :] == bytes(len(leveled) // 2 - 1000)  # the digital silence stays silence
    assert e7.LEVEL_TARGET >= 2 * 0.01  # the gate threshold in speech/src/vad.h, which is not changed


def test_the_gain_is_bounded_by_the_floor_loud_audio_is_not_wrapped_and_silence_has_gain_one():
    import numpy as np

    g = e7.level_gain(tone(2, 0.0021))
    assert g == pytest.approx(e7.LEVEL_TARGET / 0.0021, rel=0.05) and g <= e7.LEVEL_TARGET / e7.LEVEL_FLOOR
    assert e7.level_gain(tone(2, 0.0015)) == 1.0  # all under the floor: it is noise, not speech, and is left alone
    assert e7.level_gain(bytes(32000)) == 1.0  # nothing above the floor
    loud = tone(1, 0.5)
    out = np.frombuffer(e7.apply_gain(loud, 3.0), dtype=np.int16)
    assert out.max() <= 32767 and out.min() >= -32768 and np.abs(out).max() == 32767  # saturates, never wraps


def test_read_leveled_wav_returns_the_pcm_and_the_gain_used(tmp_path: Path):
    path = tmp_path / "m.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(tone(4, 0.005, silence_s=1))
    pcm, gain = e7.read_leveled_wav(path)
    assert gain == pytest.approx(e7.LEVEL_TARGET / 0.005, rel=0.05) and speech_median_rms(pcm) == pytest.approx(e7.LEVEL_TARGET, rel=0.05)
