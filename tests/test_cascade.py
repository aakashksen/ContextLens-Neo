import pytest

from contextlens.agents.cascade import CascadeChecker
from contextlens.agents.consistency_check import PromptedChecker
from contextlens.llm.client import FakeLLM, LLMError
from contextlens.schemas import Turn


def turn(text: str = "x", seq: int = 1) -> Turn:
    return Turn(id=f"t{seq}", conversation_id="c", seq=seq, speaker="A", text=text, start_ms=0, end_ms=1, source="synthetic")


def probs(contradiction: float, consistent: float = 0.0, update: float = 0.0, insufficient: float = 0.0) -> dict:
    return {"consistent": consistent, "update": update, "contradiction": contradiction, "insufficient": insufficient}


class StubFirst:
    def __init__(self, p: dict, fail: bool = False) -> None:
        self.p, self.fail, self.calls = p, fail, 0

    def probabilities(self, statement, earlier):
        self.calls += 1
        if self.fail:
            raise RuntimeError("trained model broke")
        return self.p


class StubSecond:
    def __init__(self, label: str = "contradiction", conf: float = 0.9, fail: bool = False, tokens: int | None = None) -> None:
        self.label, self.conf, self.fail, self.calls = label, conf, fail, 0
        self.last_tokens = tokens

    def check(self, statement, earlier):
        self.calls += 1
        if self.fail:
            raise LLMError("server down")
        return self.label, self.conf


def test_unflagged_case_never_reaches_the_second_stage() -> None:
    second = StubSecond()
    c = CascadeChecker(StubFirst(probs(0.1, consistent=0.8, update=0.1)), second, flag_threshold=0.3)
    assert c.check(turn(), []) == ("consistent", 0.8)
    assert second.calls == 0
    assert c.records[0].flagged is False and c.records[0].second_ms is None


def test_flagged_case_is_confirmed_by_the_second_stage_and_keeps_the_trained_probability() -> None:
    c = CascadeChecker(StubFirst(probs(0.4, consistent=0.35, update=0.25)), StubSecond("contradiction", 1.0), flag_threshold=0.3)
    assert c.check(turn(), []) == ("contradiction", 0.4)


@pytest.mark.parametrize("label", ["consistent", "update", "insufficient"])
def test_second_stage_verdicts_other_than_contradiction_flow_through_and_raise_no_alert(label: str) -> None:
    c = CascadeChecker(StubFirst(probs(0.9, consistent=0.1)), StubSecond(label, 0.7), flag_threshold=0.5)
    assert c.check(turn(), []) == (label, 0.7)


def test_flag_threshold_is_inclusive_and_acts_on_the_contradiction_probability_not_the_top_label() -> None:
    second = StubSecond()
    c = CascadeChecker(StubFirst(probs(0.25, consistent=0.5, update=0.25)), second, flag_threshold=0.25)
    assert c.check(turn(), [])[0] == "contradiction"  # consistent is the top label, but 0.25 >= 0.25 flags it
    assert second.calls == 1


def test_unflagged_label_is_the_best_non_contradiction_label_even_if_contradiction_is_top() -> None:
    # threshold above P(contradiction): the case is not flagged, so contradiction cannot be the answer
    c = CascadeChecker(StubFirst(probs(0.6, consistent=0.3, update=0.1)), StubSecond(), flag_threshold=0.9, abstain_below=0.2)
    assert c.check(turn(), []) == ("consistent", 0.3)


def test_unflagged_uncertain_case_abstains_like_the_trained_checker() -> None:
    c = CascadeChecker(StubFirst(probs(0.1, consistent=0.4, update=0.3, insufficient=0.2)), StubSecond(), flag_threshold=0.5,
                       abstain_below=0.5)
    assert c.check(turn(), []) == ("insufficient", 0.2)


def test_second_stage_failure_fails_closed_without_an_alert_and_is_counted() -> None:
    c = CascadeChecker(StubFirst(probs(0.8, insufficient=0.1)), StubSecond(fail=True), flag_threshold=0.5)
    assert c.check(turn(), []) == ("insufficient", 0.1)
    assert c.records[0].second_error == "LLMError: server down"
    assert c.summary()["second_failed"] == 1


def test_first_stage_failure_propagates_and_never_falls_back_to_the_second_stage() -> None:
    second = StubSecond()
    c = CascadeChecker(StubFirst(probs(0.8), fail=True), second, flag_threshold=0.5)
    with pytest.raises(RuntimeError, match="trained model broke"):
        c.check(turn(), [])
    assert second.calls == 0


def test_latency_and_tokens_are_recorded_per_stage_and_missing_tokens_stay_missing() -> None:
    first = StubFirst(probs(0.9))
    c = CascadeChecker(first, StubSecond(tokens=120), flag_threshold=0.5)
    c.check(turn(), [])
    c.second = StubSecond(tokens=None)
    c.check(turn(), [])
    c.first = StubFirst(probs(0.0, consistent=1.0))
    c.check(turn(), [])
    a, b, u = c.records
    assert a.first_ms >= 0 and a.second_ms is not None and a.second_tokens == 120
    assert b.second_ms is not None and b.second_tokens is None
    assert u.second_ms is None and u.second_tokens is None
    s = c.summary()
    assert s["calls"] == 3 and s["flagged"] == 2
    assert s["second_tokens_total"] == 120 and s["second_tokens_unreported"] == 1
    assert s["first_stage_llm_tokens"] == 0


def test_prompted_checker_exposes_the_tokens_of_its_last_call() -> None:
    llm = FakeLLM([{"label": "contradiction", "confidence": 0.9}])
    p = PromptedChecker(llm)
    assert p.last_tokens is None
    p.check(turn("hi", 2), [turn("hello", 1)])
    assert p.last_tokens == 0  # FakeLLM reports no tokens; a real client reports prompt plus completion tokens


def test_trained_checker_probabilities_are_keyed_by_label_and_check_still_abstains(monkeypatch) -> None:
    from contextlens.consistency import model as cm

    monkeypatch.setattr(cm, "predict_proba", lambda m, t, ex: [[0.3, 0.2, 0.35, 0.15]])
    t = cm.TrainedChecker(object(), object())
    assert t.probabilities(turn(), []) == {"consistent": 0.3, "update": 0.2, "contradiction": 0.35, "insufficient": 0.15}
    assert t.check(turn(), []) == ("insufficient", 0.15)  # top probability 0.35 is below the 0.5 threshold
