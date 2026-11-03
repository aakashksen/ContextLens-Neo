"""E7, real-time cost: speech and speaker error metrics, the AMI scenario source, the runner, the size check."""
import pytest

from contextlens.eval.check_run import recompute_metrics
from contextlens.eval.metrics import diarization_error_rate, word_error_rate
from contextlens.replay.runner import Prediction
from contextlens.schemas import ButtonRequest, Conversation, Scenario, Turn


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


def _scenario(sid: str = "m1") -> Scenario:
    spans = [("A", 0, 1000, "we ship friday"), ("B", 1000, 2000, "agreed then"), ("A", 2000, 3000, "great")]
    turns = [Turn(id=f"{sid}:t{i}", conversation_id=sid, seq=i, speaker=sp, text=tx, start_ms=a, end_ms=b,
                  source="dataset") for i, (sp, a, b, tx) in enumerate(spans)]
    return Scenario(id=sid, conversation=Conversation(id=sid, turns=turns, source="ami"),
                    request=ButtonRequest(conversation_id=sid, as_of_seq=1, question="when do we ship?"),
                    category="recall", gold_evidence_turn_ids=[f"{sid}:t0"], gold_label=None, gold_answer="friday",
                    origin="dataset")


def test_row_speech_fields_recompute_wer_and_der():
    sc = _scenario()  # only the turns up to as_of_seq=1 count
    row = Prediction(scenario_id="m1", ranked_turn_ids=["m1:t0"], asr_text="we ship thursday agreed then",
                     speaker_spans=[(0, 1000, "s1"), (1000, 2000, "s2")])
    fresh = recompute_metrics([sc], [row])
    assert fresh["wer"] == pytest.approx(1 / 5)
    assert fresh["der"] == 0.0
    bare = recompute_metrics([sc], [Prediction(scenario_id="m1", ranked_turn_ids=["m1:t0"])])
    assert "wer" not in bare and "der" not in bare  # no speech output on the row, nothing to recompute
