"""E7's judging stage: after the GPU is released, Haiku through `claude -p`, one claim per call. Fakes only."""
import json
import threading
from pathlib import Path

import pytest

from contextlens.experiments import e7_judge
from contextlens.experiments.judge_calibrate import UsageGuard
from contextlens.experiments.judge_check import Refused
from contextlens.llm.client import FakeLLM, LLMResult
from contextlens.replay.runner import Prediction
from contextlens.schemas import Card, EvidenceRef, Finding

SECRET = "the budget is 4242 dollars"


class Fake(FakeLLM):
    """Supported exactly when the claim number is even."""
    def __init__(self) -> None:
        super().__init__([])
        self.model = "fake-haiku"
        self.calls_made, self.lock = 0, threading.Lock()

    def _raw(self, messages, schema, max_tokens, temperature, timeout_s) -> LLMResult:
        with self.lock:
            self.calls_made += 1
        num = int(messages[-1]["content"].split("<claim>claim ", 1)[1].split(" ", 1)[0])
        return LLMResult(text=json.dumps({"supported": num % 2 == 0}), model="fake")


def row(sid: str, first: int, n: int) -> Prediction:
    fs = [Finding(agent="memory", kind="recall", statement=f"claim {first + i} {SECRET}", confidence=0.9,
                  evidence=[EvidenceRef(turn_id="m:t1", quote=f"quote {first + i}", speaker="Maya", start_ms=0)])
          for i in range(n)]
    return Prediction(scenario_id=sid, card=Card(mode="recall", status="answered", findings=fs), ranked_turn_ids=["m:t1"])


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "results/runs").mkdir(parents=True)
    rows = {"clean": [row("s1", 0, 2), row("s2", 2, 1)], "asr_pipeline": [row("s1", 10, 2), row("s2", 13, 1)]}
    rec = {"run_id": "E7-1", "experiment": "E7", "predictions": {}}
    for system, preds in rows.items():
        rel = f"results/runs/E7-1.{system}.predictions.jsonl"
        (tmp_path / rel).write_text("\n".join(p.model_dump_json() for p in preds) + "\n", encoding="utf-8")
        rec["predictions"][system] = rel
    (tmp_path / "results/runs/E7-1.json").write_text(json.dumps(rec), encoding="utf-8")
    return tmp_path


def guard(pct: float = 10.0) -> UsageGuard:
    return UsageGuard(90.0, read=lambda: pct)


def test_every_claim_is_graded_once_and_rates_are_per_system(root: Path):
    llm = Fake()
    out = e7_judge.judge("E7-1", llm=llm, root=root, counter_path=root / "calls.json", guard=guard(), say=lambda m: None, sleep=lambda s: None)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert llm.calls_made == 6 and doc["complete"] is True and doc["model"] == "fake-haiku"
    assert doc["per_system"]["clean"] == {"claims": 3, "graded": 3, "supported": 2, "grounded_rate": 2 / 3}
    assert doc["per_system"]["asr_pipeline"]["supported"] == 1
    assert doc["rubric"] == "e2-v1" and "entry 56" in doc["note"] and "not proof" in doc["note"]
    assert SECRET not in out.read_text(encoding="utf-8")  # ids and verdicts only
    assert out.parent == root / "results/e7-judge"


def test_a_second_run_resumes_and_asks_nothing_new(root: Path):
    e7_judge.judge("E7-1", llm=Fake(), root=root, counter_path=root / "calls.json", guard=guard(), say=lambda m: None, sleep=lambda s: None)
    again = Fake()
    e7_judge.judge("E7-1", llm=again, root=root, counter_path=root / "calls.json", guard=guard(), say=lambda m: None, sleep=lambda s: None)
    assert again.calls_made == 0


def test_a_weekly_usage_at_the_limit_refuses_before_any_call(root: Path):
    llm = Fake()
    with pytest.raises(Refused, match="weekly usage"):
        e7_judge.judge("E7-1", llm=llm, root=root, counter_path=root / "calls.json", guard=guard(95.0), say=lambda m: None, sleep=lambda s: None)
    assert llm.calls_made == 0


def test_the_guard_is_required_unless_turned_off(root: Path):
    with pytest.raises(Refused, match="guard"):
        e7_judge.judge("E7-1", llm=Fake(), root=root, counter_path=root / "calls.json", guard=None, say=lambda m: None, sleep=lambda s: None)


def test_a_missing_record_is_refused(root: Path):
    with pytest.raises(Refused, match="no run record"):
        e7_judge.judge("E7-9", llm=Fake(), root=root, guard=guard(), say=lambda m: None, sleep=lambda s: None)


def test_the_cli_stops_at_98_percent_weekly_usage_by_default(monkeypatch):
    seen = {}

    def fake_judge(run_id, *, guard, **kw):
        seen["limit"] = guard.limit
        return Path("x")

    monkeypatch.setattr(e7_judge, "judge", fake_judge)
    monkeypatch.setattr(e7_judge.pg, "CountedClaude", lambda: None)
    assert e7_judge.main(["E7-1"]) == 0 and seen["limit"] == 98.0
