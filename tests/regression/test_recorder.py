"""The recorder and the replay machinery, with a scripted model. No model is called for real."""
import json
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from contextlens.data.scenarios import load_scenarios
from contextlens.experiments import e2_context as e2
from contextlens.experiments import regression as reg
from contextlens.llm.client import FakeLLM

DEV = load_scenarios("dev", Path(__file__).resolve().parents[2] / "datasets/scenarios")


def test_selection_is_thirty_fixed_invented_cases_covering_every_category() -> None:
    picked = reg.select_cases(DEV)
    assert len(picked) == 30
    assert [s.id for s in picked] == [s.id for s in reg.select_cases(list(reversed(DEV)))]  # not an accident of order
    assert all(s.origin == "synthetic" and s.conversation.source == "synthetic" and s.conversation.split == "dev" for s in picked)
    assert set(Counter(s.category for s in picked)) == {s.category for s in DEV}
    assert {c: n for c, n in Counter(s.category for s in picked).items()} == {
        "contradiction": 4, "hypothetical_or_joking": 4, "insufficient_evidence": 4, "legitimate_update": 4,
        "prompt_injection": 4, "recall": 4, "respond_with_suggestion": 3, "wrong_speaker_trap": 3,
    }
    assert reg.select_cases(DEV, seed="another") != picked


def _scripted(sc) -> list[str]:
    """The memory call, then a step that quotes the first turn, then the final card."""
    t0 = sc.conversation.turns[0]
    final = {"tool": "final", "card": {
        "mode": "recall", "status": "answered", "findings": [{
            "agent": "single", "kind": "recall", "statement": t0.text,
            "evidence": [{"turn_id": t0.id, "quote": t0.text, "speaker": t0.speaker, "start_ms": t0.start_ms}],
            "confidence": 0.9}]}}
    return [json.dumps({"items": []}), json.dumps({"tool": "get_turns", "args": {"turn_ids": [t0.id]}}), json.dumps(final)]


def test_record_keeps_every_raw_message_in_order_and_the_replay_reproduces_the_card() -> None:
    sc = reg.select_cases(DEV)[0]
    scripted = _scripted(sc)
    entry = reg.record_case(sc, FakeLLM(list(scripted)), reg.HashEmbedder())
    assert entry["calls"] == scripted
    assert entry["expected"]["status"] == "answered"
    assert entry["expected"]["findings"][0]["evidence"] == [[sc.conversation.turns[0].id, sc.conversation.turns[0].text]]
    assert reg.replay_case(entry) == entry["expected"]


def test_replay_fails_when_the_code_asks_for_more_or_fewer_calls_than_recorded() -> None:
    sc = reg.select_cases(DEV)[0]
    entry = reg.record_case(sc, FakeLLM(_scripted(sc)), reg.HashEmbedder())
    with pytest.raises(reg.ReplayMismatch):
        reg.replay_case({**entry, "calls": entry["calls"] + ['{"items": []}']})  # one reply never used


def test_a_failed_call_stops_the_recording() -> None:
    sc = reg.select_cases(DEV)[0]
    with pytest.raises(e2.RunBroken):
        reg.record_case(sc, FakeLLM([]), reg.HashEmbedder())  # the memory call fails
    with pytest.raises(e2.RunBroken):
        reg.record_case(sc, FakeLLM(_scripted(sc)[:2]), reg.HashEmbedder())  # the agent runs out of replies


def test_the_recorder_refuses_a_dirty_tree(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "x.txt").write_text("x")
    with pytest.raises(e2.RunBroken, match="uncommitted"):
        reg.record_all(tmp_path, DEV, FakeLLM([]), reg.HashEmbedder(), "model")


def test_the_recorder_rejects_a_case_that_is_not_invented() -> None:
    real = next(s for s in DEV if s.origin != "synthetic")
    with pytest.raises(e2.RunBroken, match="invented"):
        reg.record_case(real, FakeLLM([]), reg.HashEmbedder())

