"""Replay of the recorded real writer replies (tests/regression/recorded/): no model is called.

A prompt change must be followed by a re-recording (`python -m contextlens.experiments.regression`), or the hash test fails."""
import json
from pathlib import Path

import pytest

from contextlens.experiments import regression as reg

FILE = Path(__file__).resolve().parents[2] / reg.RECORDED_REL
DOC = json.loads(FILE.read_text(encoding="utf-8")) if FILE.exists() else {"cases": []}


def test_the_recordings_exist() -> None:
    assert FILE.exists(), f"no recording at {reg.RECORDED_REL}; record it with python -m contextlens.experiments.regression"


def test_recordings_match_current_prompt_hash() -> None:
    assert DOC["prompt_version"] == reg.PROMPT_VERSION and DOC["system"] == reg.SYSTEM
    assert DOC["system_message_sha256"] == reg.system_sha256(), "the system message changed: re-record the cases"


def test_thirty_invented_dev_cases_cover_every_category() -> None:
    cases = DOC["cases"]
    assert len(cases) == reg.N_CASES
    assert {c["category"] for c in cases} == {
        "contradiction", "hypothetical_or_joking", "insufficient_evidence", "legitimate_update", "prompt_injection",
        "recall", "respond_with_suggestion", "wrong_speaker_trap"}
    assert all(c["scenario"]["origin"] == "synthetic" and c["scenario"]["conversation"]["source"] == "synthetic"
               and c["scenario"]["conversation"]["split"] == "dev" for c in cases)


@pytest.mark.parametrize("case", DOC["cases"], ids=[c["id"] for c in DOC["cases"]])
def test_replay_gives_the_recorded_status_labels_and_citations(case: dict) -> None:
    assert reg.replay_case(case) == case["expected"]
