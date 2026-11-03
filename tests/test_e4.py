"""E4: trained consistency model. Tiny models, fixtures and invented text only; no GPU, no network."""
import pytest

from contextlens.eval import metrics as M
from contextlens.eval import series
from contextlens.eval.check_run import recompute_metrics
from contextlens.replay.runner import Prediction
from contextlens.schemas import ButtonRequest, Conversation, Scenario, Turn


def _sc(sid: str, gold: str | None, category: str = "contradiction") -> Scenario:
    turns = [Turn(id=f"{sid}:t{i}", conversation_id=sid, seq=i, speaker="A", text=f"remark {i} in {sid}",
                  start_ms=i * 1000, end_ms=i * 1000 + 900, source="synthetic") for i in range(3)]
    return Scenario(id=sid, conversation=Conversation(id=sid, turns=turns, source="synthetic", split="dev"),
                    request=ButtonRequest(conversation_id=sid, as_of_seq=2), category=category,
                    gold_evidence_turn_ids=[], gold_label=gold, gold_answer=None, origin="synthetic")  # type: ignore[arg-type]


# ---- calibration ----

def test_ece_is_the_weighted_gap_between_confidence_and_accuracy() -> None:
    assert M.ece([True, True, False, False], [0.9] * 4) == pytest.approx(0.4)
    assert M.ece([True] * 5, [1.0] * 5) == 0.0  # a probability of exactly 1 lands in the last bin
    # two bins: 0.15 -> acc 0 (gap .15, weight .5); 0.85 -> acc 1 (gap .15, weight .5)
    assert M.ece([False, True], [0.15, 0.85]) == pytest.approx(0.15)
    assert M.ece([], []) is None
    with pytest.raises(ValueError):
        M.ece([True], [0.5, 0.5])


def test_label_rows_carry_the_prediction_without_a_card() -> None:
    row = Prediction(scenario_id="x", label="contradiction", probability=0.7)
    assert row.card is None and row.raw_output is None
    with pytest.raises(ValueError):
        Prediction(scenario_id="x")
    with pytest.raises(ValueError):
        Prediction(scenario_id="x", label="update", probability=1.5)


def test_ece_and_label_metrics_recompute_from_label_rows() -> None:
    scs = [_sc("a", "contradiction"), _sc("b", "consistent"), _sc("c", "update"), _sc("d", None)]
    preds = [Prediction(scenario_id="a", label="contradiction", probability=0.9),
             Prediction(scenario_id="b", label="contradiction", probability=0.8),
             Prediction(scenario_id="c", label="update", probability=0.6),
             Prediction(scenario_id="d", label="consistent", probability=0.99)]
    m = recompute_metrics(scs, preds)
    # the unlabelled case d is not scored: correct = a, c; confidences .9 .8 .6
    assert m["ece"] == pytest.approx(M.ece([True, False, True], [0.9, 0.8, 0.6]))
    assert m["false_alert_rate"] == pytest.approx(0.5)  # of b and c, only b is called a contradiction
    assert "ece[category=contradiction]" in m


def test_ece_is_left_out_when_a_scored_row_has_no_probability() -> None:
    scs = [_sc("a", "contradiction"), _sc("b", "consistent")]
    preds = [Prediction(scenario_id="a", label="contradiction", probability=0.9),
             Prediction(scenario_id="b", label="consistent")]
    assert "ece" not in recompute_metrics(scs, preds)
    assert "ece" not in recompute_metrics(scs, [Prediction(scenario_id=p.scenario_id, label=p.label) for p in preds])


def test_ece_has_an_interval_from_resampled_conversations() -> None:
    scs = [_sc(f"s{i}", "contradiction" if i % 2 else "consistent") for i in range(12)]
    preds = [Prediction(scenario_id=s.id, label="contradiction", probability=0.5 + (i % 5) / 10) for i, s in enumerate(scs)]
    series.RESAMPLES = 100
    try:
        lo, hi = series.interval(list(zip(scs, preds)), "ece")  # type: ignore[misc]
    finally:
        series.RESAMPLES = 10_000
    assert 0 <= lo <= hi <= 1


# ---- the train-only scenario set ----

from pathlib import Path  # noqa: E402

from contextlens.data import train_scenarios as ts  # noqa: E402
from contextlens.data.scenarios import load_scenarios  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_training_set_covers_every_label_and_card_mode_and_is_synthetic() -> None:
    scs = ts.load_train_scenarios(ROOT / "datasets/scenarios_train")
    c = ts.counts(scs)
    assert set(c["by_label"]) >= {"consistent", "update", "contradiction", "insufficient"}
    assert set(c["by_mode"]) == {"recall", "clarify", "respond"}
    assert c["abstention_cases"]["insufficient"] >= 5 and c["injected_turn_cases"]["with_injection"] >= 5
    assert all(s.origin == "synthetic" and s.conversation.split == "train" for s in scs)
    assert c["cases"]["total"] == c["cases"]["conversations"] == len({s.id for s in scs})
    assert scs == ts.build_train_scenarios()  # the committed file is what the code builds


def test_training_set_shares_nothing_with_dev_or_test_scenarios() -> None:
    train = ts.load_train_scenarios(ROOT / "datasets/scenarios_train")
    held = [s for sp in ("dev", "test") for s in load_scenarios(sp, ROOT / "datasets/scenarios")]
    assert ts.overlap_problems(train, held) == []
    assert not {s.conversation.id for s in train} & {s.conversation.id for s in held}


def test_overlap_check_catches_an_id_a_turn_and_an_eight_word_run() -> None:
    train = ts.build_train_scenarios()
    held = [train[0].model_copy(update={"id": "x"})]
    problems = ts.overlap_problems(train[:1], held)
    assert any("repeats a held-out turn" in p for p in problems) and any("8-word runs" in p for p in problems)
    other = train[1]
    reworded = other.model_copy(update={"id": "y", "conversation": other.conversation.model_copy(update={"id": "y"})})
    assert any("also in" in p for p in ts.overlap_problems([other], [other])) and ts.overlap_problems([other], [reworded])


def test_training_cases_are_not_near_duplicates_of_each_other() -> None:
    scs = ts.build_train_scenarios()
    seen: dict[str, str] = {}
    for s in scs:
        for run in ts.word_runs([t.text for t in s.conversation.turns], 6):
            assert seen.setdefault(run, s.id) == s.id, f"{run!r} appears in {seen[run]} and {s.id}"


def test_deadline_update_pair_is_in_eval_set() -> None:
    from contextlens.consistency.data import build_pairs

    pairs = build_pairs("dev", ROOT / "tests/fixtures/data", ROOT / "datasets/scenarios")
    hit = [p for p in pairs if p.source == "scenarios" and p.label == "update"
           and p.statement == "We moved the deadline to Wednesday"]
    assert len(hit) == 1 and hit[0].earlier == ["The deadline is Monday"]
