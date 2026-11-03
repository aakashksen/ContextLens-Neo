import pytest

from contextlens.eval.metrics import (
    LABELS, abstention_accuracy, false_alert_rate, first_gold_rank, grounded_rate,
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
    expected = f1_score(gold, pred, labels=list(LABELS), average="macro", zero_division=0)
    assert macro_f1(gold, pred) == pytest.approx(expected, abs=1e-9)
    # by hand: (0.5 + 0.8 + 2/3 + 1) / 4
    assert macro_f1(gold, pred) == pytest.approx(0.741667, abs=1e-5)


def test_macro_f1_matches_sklearn_with_prediction_only_class():
    from sklearn.metrics import f1_score

    gold = ["consistent", "consistent", "update", "update"]
    pred = ["consistent", "insufficient", "update", "insufficient"]
    expected = f1_score(gold, pred, labels=list(LABELS), average="macro", zero_division=0)
    assert macro_f1(gold, pred) == pytest.approx(expected, abs=1e-9)
    # a label absent from both lists (contradiction) counts as 0 by the definition in docs/TESTING.md
    # by hand: (2/3 + 2/3 + 0 + 0) / 4 = 1/3
    assert macro_f1(gold, pred) == pytest.approx(1 / 3, abs=1e-9)


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


@pytest.mark.parametrize("call", [
    lambda: macro_f1(["update", "update"], ["update"]),
    lambda: false_alert_rate(["consistent"], []),
    lambda: abstention_accuracy([True], [True, False]),
    lambda: speaker_accuracy(["a", "b", "c"], ["a"]),
    lambda: speaker_accuracy(["a"], ["a", "b"]),
])
def test_paired_metrics_reject_unequal_lengths(call) -> None:
    with pytest.raises(ValueError):
        call()


def test_empty_input_is_none_not_zero() -> None:
    assert mrr([]) is None
    assert grounded_rate([], {}, 0) is None
    assert grounded_rate([Card(mode="recall", status="abstained", findings=[])], {}, 0) is None
    assert speaker_accuracy([], []) is None
    assert macro_f1([], []) is None
    assert false_alert_rate(["consistent"], ["consistent"]) == 0.0
    assert false_alert_rate(["contradiction"], ["contradiction"]) is None
    assert recall_at_k(["a"], [], 1) is None
    assert abstention_accuracy([], []) == {"abstain_when_insufficient": None, "answer_when_answerable": None}
    assert abstention_accuracy([True], [True]) == {"abstain_when_insufficient": 1.0, "answer_when_answerable": None}
    assert abstention_accuracy([True], [False]) == {"abstain_when_insufficient": None, "answer_when_answerable": 0.0}


def test_bootstrap_ci_empty_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="no values"):
        bootstrap_ci([], [])


def test_label_f1_is_the_per_label_term_of_macro_f1_and_none_without_gold_cases():
    from contextlens.eval.metrics import label_f1

    gold = ["consistent", "consistent", "update", "update",
            "contradiction", "contradiction", "insufficient", "insufficient"]
    pred = ["consistent", "update", "update", "update",
            "contradiction", "consistent", "insufficient", "insufficient"]
    got = {x: label_f1(gold, pred, x) for x in LABELS}
    assert got == pytest.approx({"consistent": 0.5, "update": 0.8, "contradiction": 2 / 3, "insufficient": 1.0})
    assert sum(got.values()) / 4 == pytest.approx(macro_f1(gold, pred))
    # a label that is only predicted has no gold cases: undefined, not 0.0 (macro_f1 still counts it as 0)
    assert label_f1(["consistent"], ["update"], "update") is None
    assert label_f1([], [], "update") is None
