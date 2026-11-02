import pytest

from contextlens.eval.metrics import (
    abstention_accuracy, false_alert_rate, first_gold_rank, grounded_rate,
    macro_f1, mrr, recall_at_k, speaker_accuracy,
)
from contextlens.eval.stats import bootstrap_ci, mcnemar_exact, paired_bootstrap_diff, percentile
from contextlens.schemas import Card, EvidenceRef, Finding


def test_recall_at_k_gold_rank_3():
    ranked = ["a", "b", "g", "d"]
    assert recall_at_k(ranked, ["g"], 2) == 0
    assert recall_at_k(ranked, ["g"], 3) == 1
    assert first_gold_rank(ranked, ["g"]) == 3


def test_mrr():
    assert mrr([1, 2]) == 0.75
    assert mrr([1, None]) == 0.5


def test_mcnemar_exact():
    a = [True] * 1 + [False] * 9
    b = [False] * 1 + [True] * 9
    assert mcnemar_exact(a, b) == pytest.approx(0.0215, abs=0.0005)
    assert mcnemar_exact([True], [True]) == 1.0


def test_macro_f1_matches_sklearn_value():
    from sklearn.metrics import f1_score

    gold = ["consistent", "consistent", "update", "update",
            "contradiction", "contradiction", "insufficient", "insufficient"]
    pred = ["consistent", "update", "update", "update",
            "contradiction", "consistent", "insufficient", "insufficient"]
    assert macro_f1(gold, pred) == pytest.approx(f1_score(gold, pred, average="macro"), abs=1e-9)
    # by hand: (0.5 + 0.8 + 2/3 + 1) / 4
    assert macro_f1(gold, pred) == pytest.approx(0.741667, abs=1e-5)


def test_macro_f1_matches_sklearn_with_prediction_only_class():
    from sklearn.metrics import f1_score

    gold = ["consistent", "consistent", "update", "update"]
    pred = ["consistent", "insufficient", "update", "insufficient"]
    expected = f1_score(gold, pred, average="macro")
    assert macro_f1(gold, pred) == pytest.approx(expected, abs=1e-9)


def test_false_alert_rate():
    gold = ["consistent", "update", "contradiction", "insufficient"]
    pred = ["contradiction", "update", "contradiction", "contradiction"]
    assert false_alert_rate(gold, pred) == pytest.approx(2 / 3)


def test_abstention_and_speaker():
    r = abstention_accuracy([True, False, True, True], [True, True, False, False])
    assert r == {"abstain_when_insufficient": 0.5, "answer_when_answerable": 0.0}
    assert speaker_accuracy(["A", "B"], ["A", "A"]) == 0.5


def _card(turn_id: str) -> Card:
    ev = EvidenceRef(turn_id=turn_id, quote="q", speaker="A", start_ms=0)
    f = Finding(agent="x", kind="recall", statement="s", evidence=[ev], confidence=1.0)
    return Card(mode="recall", status="answered", findings=[f])


def test_grounded_rate_future_turn_is_zero():
    seqs = {"t1": 1, "t5": 5}
    assert grounded_rate([_card("t1")], seqs, as_of_seq=3) == 1
    assert grounded_rate([_card("t5")], seqs, as_of_seq=3) == 0
    assert grounded_rate([_card("missing")], seqs, as_of_seq=3) == 0
    assert grounded_rate([_card("t1")], seqs, 3, supports=lambda f: False) == 0


def test_percentile():
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([0, 10], 95) == pytest.approx(9.5)


def test_bootstrap_resamples_whole_conversations():
    # Two conversations, all 0s and all 1s: resampling whole groups gives only
    # means 0, 0.5 or 1, so the interval spans the full range.
    values = [0.0] * 5 + [1.0] * 5
    groups = ["a"] * 5 + ["b"] * 5
    lo, hi = bootstrap_ci(values, groups, n=500)
    assert (lo, hi) == (0.0, 1.0)
    assert bootstrap_ci(values, groups, n=500) == (lo, hi)


def test_paired_bootstrap_diff():
    a = [1.0, 1.0, 1.0, 1.0]
    b = [0.0, 0.0, 0.0, 0.0]
    d, lo, hi = paired_bootstrap_diff(a, b, ["x", "x", "y", "y"], n=200)
    assert (d, lo, hi) == (1.0, 1.0, 1.0)
