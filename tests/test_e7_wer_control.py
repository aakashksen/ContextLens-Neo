"""The WER control: offline Whisper on the same rebuilt audio, beside the streamed WER in a saved E7 record.
Fakes only: no model is loaded."""
import json
import wave
from pathlib import Path

import pytest

from contextlens.experiments import e7_wer_control as ctl
from contextlens.replay.runner import Prediction
from contextlens.schemas import ButtonRequest, Conversation, Scenario, Turn
from contextlens.speech.diarize import SAMPLE_RATE


def turns(m: str) -> list[Turn]:
    # two speakers overlap between 2 s and 3 s, in the first and second turns
    spec = [("A", 0, 3000, "alpha beta gamma"), ("B", 2000, 5000, "delta epsilon zeta"), ("A", 6000, 9000, "eta theta iota")]
    return [Turn(id=f"{m}:t{i}", conversation_id=m, seq=i, speaker=s, speaker_confidence=1.0, text=t, start_ms=a,
                 end_ms=b, source="dataset") for i, (s, a, b, t) in enumerate(spec)]


def scenario(m: str, as_of: int) -> Scenario:
    conv = Conversation(id=m, turns=turns(m)[: as_of + 1], source="ami", split="dev")
    req = ButtonRequest(conversation_id=m, as_of_seq=as_of, question="q")
    return Scenario(id=f"{m}-q{as_of}", conversation=conv, request=req, category="recall", gold_evidence_turn_ids=[],
                    gold_label=None, gold_answer=None, origin="dataset")


def write_wav(path: Path, seconds: float) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(bytes(int(seconds * SAMPLE_RATE) * 2))


class Fake:
    def __init__(self, text: str) -> None:
        self.text, self.seen, self.closed = text, [], False

    def __call__(self, audio) -> str:
        self.seen.append(len(audio))
        return self.text

    def close(self) -> None:
        self.closed = True


def test_control_compares_streamed_and_offline_wer_on_the_same_reference_and_audio(tmp_path: Path):
    scs = [scenario("m1", 1), scenario("m1", 2), scenario("m2", 1)]
    for m in ("m1", "m2"):
        write_wav(tmp_path / f"{m}.wav", 10)
    streamed = {"m1-q2": "alpha beta gamma delta epsilon zeta eta theta xxx", "m2-q1": "alpha beta gamma delta epsilon zeta"}
    rows = {sid: Prediction(scenario_id=sid, ranked_turn_ids=[], asr_text=t) for sid, t in streamed.items()}
    good = Fake("alpha beta gamma delta epsilon zeta eta theta iota")
    models = {"tiny.en": lambda: good, "base.en": lambda: Fake("nothing")}
    out = ctl.control(scs, rows, tmp_path, ["m1", "m2"], models, say=lambda m: None)
    m1 = out["per_meeting"]["m1"]
    assert m1["scenario"] == "m1-q2" and m1["reference_words"] == 9  # the last question of the meeting, all its turns
    assert m1["streamed_wer"] == pytest.approx(1 / 9)
    assert m1["offline_wer"]["tiny.en"] == 0.0 and m1["offline_wer"]["base.en"] == 1.0
    assert good.seen == [9000 * SAMPLE_RATE // 1000, 5000 * SAMPLE_RATE // 1000] and good.closed  # audio up to each cut
    assert out["overlap_share"]["m1"] == pytest.approx(6 / 9)  # the two overlapping turns hold 6 of 9 words
    assert out["totals"]["streamed_wer"] > 0 and out["totals"]["offline_wer"]["tiny.en"] == pytest.approx(3 / 15)  # m2 gets three extra words


def test_the_output_is_labelled_a_diagnostic_and_not_an_e7_record(tmp_path: Path):
    path = ctl.write_diagnostic({"per_meeting": {}}, tmp_path, "E7-1")
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert path.parent == tmp_path / "results/local" and path.name.startswith("e7-wer-control-")
    assert doc["diagnostic"] is True and doc["not_an_e7_record"] is True and doc["e7_record"] == "E7-1"
    assert "base.en" in doc["note"] and "never E7's pipeline" in doc["note"]
    assert "whisper.cpp" in doc["limits"][0]  # offline here is the transformers implementation, not the service's
