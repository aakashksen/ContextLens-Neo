"""T9.1's precondition: `team_shipping_changed` with retrieval before the memory model call, once, on the new dev cases,
judged by Haiku beside the earlier arm's rows. No real model."""
import json
import shutil
from pathlib import Path
import threading
import time

import pytest

from contextlens.eval import series
from contextlens.eval.check_run import check
from contextlens.experiments import team_prefetch as tp
from contextlens.experiments.e2_context import RunBroken
from contextlens.experiments.judge_calibrate import UsageGuard
from contextlens.llm.client import FakeLLM, LLMResult
from contextlens.llm.ollama_ctl import OllamaCtl
from tests.test_e1 import HashEmbedder
from tests.test_e2 import NAMES, RETRIEVAL, Wire, _git
from tests.test_team_attempt_run import REAL, Writer


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(series, "RESAMPLES", 100)
    for rel in ("datasets/scenarios", "datasets/scenarios_extra", "datasets/splits"):
        shutil.copytree(REAL / rel, tmp_path / rel)
    (tmp_path / ".gitignore").write_text("results/partial/\nresults/local/\n", encoding="utf-8")
    for args in (("init", "-q"), ("add", "-A"), ("commit", "-q", "-m", "init")):
        _git(tmp_path, *args)
    return tmp_path


class PrefetchedWriter(Writer):
    """The scripted writer finds the case in the first user message; with a prefetch the last message is a tool result,
    so it is shown the first two messages only (the full prompt is what `calls` keeps)."""

    def _raw(self, messages, *a):
        out = super()._raw(messages[:2], *a)
        self.calls[-1] = messages
        return out


def go(repo, writer=None):
    return tp.run_prefetch(root=repo, writer=writer or PrefetchedWriter(), embedder=HashEmbedder(), ctl=OllamaCtl(request=Wire()),
                           names=NAMES, retrieval_config=RETRIEVAL, run_conditions="test", say=lambda t: None)


def test_the_run_writes_one_valid_record_on_the_new_cases_and_says_what_it_is(repo):
    writer = PrefetchedWriter()
    path = go(repo, writer)
    rec = json.loads(path.read_text(encoding="utf-8"))
    cfg = rec["config"]
    assert rec["experiment"] == "E3TP" and rec["scenario_source"] == "scenarios_extra" and rec["n"] == 108
    assert set(rec["predictions"]) == {tp.ARM} and cfg["arm"] == tp.ARM and cfg["judged"] is False
    assert "retrieval before the memory model call" in cfg["change"] and cfg["based_on_rows"] == tp.EARLIER_RECORD
    assert set(cfg["deadline"]) == {tp.ARM} and "cut_by_deadline" in cfg["deadline"][tp.ARM]
    assert any(m["role"] == "user" and m["content"].startswith("<tool_result>\n") for c in writer.calls for m in c)  # prefetched
    assert check("E3TP", root=repo, min_n=100, split="dev") == []


def test_a_dirty_tree_is_refused_before_a_model_is_touched(repo):
    (repo / "x.txt").write_text("x", encoding="utf-8")
    writer = PrefetchedWriter()
    with pytest.raises(RunBroken, match="uncommitted"):
        go(repo, writer)
    assert not writer.calls


def test_a_stopped_run_resumes_at_the_same_commit_and_runs_only_the_rest(repo):
    with pytest.raises(RunBroken):
        go(repo, PrefetchedWriter(die_after=150))
    saved = (repo / tp.PARTIAL_REL).read_text(encoding="utf-8").splitlines()
    assert saved
    again = PrefetchedWriter()
    path = go(repo, again)
    assert json.loads(path.read_text(encoding="utf-8"))["config"]["rows_reused_from_partial"] >= 1


# -- judging --------------------------------------------------------------------------------------------------------

class Fake(FakeLLM):
    """Supported exactly when the claim text has an even length."""

    def __init__(self, model: str) -> None:
        super().__init__([])
        self.model, self.calls_made, self.lock = model, 0, threading.Lock()

    def _raw(self, messages, schema, max_tokens, temperature, timeout_s) -> LLMResult:
        with self.lock:
            self.calls_made += 1
        time.sleep(0)
        text = messages[-1]["content"].split("<claim>", 1)[1].split("</claim>", 1)[0]
        return LLMResult(text=json.dumps({"supported": len(text) % 2 == 0}), model="fake")


def test_the_score_is_the_written_score_with_one_judge_for_both_arms():
    m = {"answer_when_answerable": 1.0, "abstain_when_insufficient": 0.6}
    assert tp.written_score(0.9, m) == pytest.approx((0.9 + 0.6) / 2)
    assert tp.written_score(None, m) is None


def test_both_arms_are_judged_by_the_same_model_and_sonnet_only_when_asked(repo, tmp_path):
    new = go(repo)
    old_id = tp.EARLIER_RECORD
    # an earlier record to compare with: the new record's rows stand in for the earlier arm's
    (repo / "results/runs").mkdir(exist_ok=True)
    rec = json.loads(new.read_text(encoding="utf-8"))
    stand_in = repo / "results/runs" / f"{old_id}.json"
    stand_in.write_text(json.dumps({**rec, "run_id": old_id, "predictions": {tp.EARLIER_ARM: rec["predictions"][tp.ARM]},
                                    "metrics": {tp.EARLIER_ARM: rec["metrics"][tp.ARM]}}), encoding="utf-8")
    haiku, sonnet = Fake("claude-haiku"), Fake("sonnet")
    out = tp.judge(new.stem, judges=[haiku], root=repo, guard=UsageGuard(95.0, read=lambda: 10.0), counter_path=tmp_path / "calls.json", say=lambda m: None,
                   sleep=lambda s: None)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert set(doc["judges"]) == {"claude-haiku"} and set(doc["judges"]["claude-haiku"]["arms"]) == {"earlier", "fixed"}
    assert doc["judges"]["claude-haiku"]["arms"]["earlier"]["claims"] == doc["judges"]["claude-haiku"]["arms"]["fixed"]["claims"]
    assert haiku.calls_made > 0 and sonnet.calls_made == 0
    assert "deadline" in doc and set(doc["deadline"]) == {"earlier", "fixed"}
    out2 = tp.judge(new.stem, judges=[haiku, sonnet], root=repo, guard=UsageGuard(95.0, read=lambda: 10.0), counter_path=tmp_path / "calls.json",
                    say=lambda m: None, sleep=lambda s: None)
    assert set(json.loads(out2.read_text(encoding="utf-8"))["judges"]) == {"claude-haiku", "sonnet"} and sonnet.calls_made > 0


def test_the_dev_check_builds_the_team_with_both_fixes_on_and_production_ships_neither():
    """The dev check measured the two fixes that were then dropped; production now ships neither. The check keeps pressing
    the fixed team (its own FIXES), so it stays a record of what was measured."""
    from contextlens.api import production
    from contextlens.memory.extract import MemoryStore

    sc = tp.load_verified(tp.SOURCE, tp.ta.SPLIT, REAL)[0][0]
    team = tp.make_assistant(sc, Writer(), HashEmbedder(), RETRIEVAL, MemoryStore())
    assert team.prefetch_memory is True and team.specialist_deadline_s == production.SPECIALIST_DEADLINE_S == 5.5
    assert tp.FIXES == {"prefetch_memory": True, "specialist_deadline_s": production.SPECIALIST_DEADLINE_S}
    assert production.TEAM_SETTINGS == {"prefetch_memory": False, "specialist_deadline_s": None}


def test_the_arm_and_the_change_name_both_fixes():
    assert "subdeadline" in tp.ARM and "prefetch" in tp.ARM
    assert "retrieval before the memory model call" in tp.CHANGE and "sub-deadline" in tp.CHANGE


def test_the_judge_command_stops_at_98_percent_weekly_usage_by_default(monkeypatch):
    seen = {}

    def fake_judge(record, *, guard, **kw):
        seen["limit"] = guard.limit
        return Path("x")

    monkeypatch.setattr(tp, "judge", fake_judge)
    import contextlens.experiments.parallel_grader as pg

    monkeypatch.setattr(pg, "CountedClaude", lambda *a: None)
    assert tp.main(["judge", "E3TP-1"]) == 0 and seen["limit"] == 98.0
