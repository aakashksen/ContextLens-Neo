"""CONTEXTLENS_ASSISTANT picks the system behind the button; the default stays the team."""

import pytest
from prometheus_client import CollectorRegistry

from contextlens.agents.graph import TeamAssistant
from contextlens.agents.single import SingleAgent
from contextlens.api import production
from contextlens.llm.client import FakeLLM
from contextlens.memory.store import InMemoryTranscriptStore
from contextlens.obs.tracing import Telemetry
from contextlens.replay.runner import Context
from contextlens.schemas import ButtonRequest, Turn
from test_api import FakeSpeech, NoEmbed

QUOTE = "The deadline is Monday"


def _store():
    store = InMemoryTranscriptStore()
    store.add(Turn(id="c1-t0", conversation_id="c1", seq=0, speaker="a", text=QUOTE, start_ms=0, end_ms=900, source="synthetic"))
    return store


def _evidence(agent):
    ev = {"turn_id": "c1-t0", "quote": QUOTE, "speaker": "a", "start_ms": 0}
    return {"agent": agent, "kind": "recall", "statement": "The deadline is Monday", "confidence": 0.9, "evidence": [ev]}


def _build(llms, **kw):
    return production.build_services(store=_store(), embedder=NoEmbed(), llms=llms, background_llm=None, speech=FakeSpeech(),
                                     telemetry=Telemetry(registry=CollectorRegistry()), **kw)


def _press(prod):
    req = ButtonRequest(conversation_id="c1", as_of_seq=0, question="When is the deadline?")
    return prod.traced.inner.answer(req, Context(store=prod.services.store, as_of_seq=0))


def test_the_default_is_the_team_and_it_answers_with_a_consistency_checker(monkeypatch):
    monkeypatch.delenv("CONTEXTLENS_ASSISTANT", raising=False)
    assert production.assistant_kind() == "team" and production.DEFAULT_ASSISTANT == "team"
    seen = []

    class Spy(TeamAssistant):
        def __init__(self, llms, tools, *a, **kw):
            seen.append(tools.checker)
            super().__init__(llms, tools, *a, **kw)

    monkeypatch.setattr(production, "TeamAssistant", Spy)
    llms = {"memory": FakeLLM([{"tool": "final", "findings": [_evidence("memory")]}] * 3),
            "consistency": FakeLLM([{"tool": "final", "findings": []}] * 3),
            "assistance": FakeLLM([{"mode": "recall", "followup": None}] * 3)}
    prod = _build(llms)
    card = _press(prod)
    assert card.status == "answered" and card.findings[0].evidence[0].turn_id == "c1-t0"
    assert len(seen) == 1 and isinstance(seen[0], production.PromptedChecker)


def test_single_answers_through_the_same_path_with_the_assistance_model(monkeypatch):
    seen = []

    class Spy(SingleAgent):
        def __init__(self, llm, tools, *a, **kw):
            seen.append(tools.checker)
            super().__init__(llm, tools, *a, **kw)

    monkeypatch.setattr(production, "SingleAgent", Spy)
    card_json = {"mode": "recall", "status": "answered", "findings": [_evidence("single")]}
    assistance = FakeLLM([{"tool": "final", "card": card_json}] * 2)
    never = FakeLLM([])
    prod = _build({"memory": never, "consistency": never, "assistance": assistance}, assistant="single")
    card = _press(prod)
    assert card.status == "answered" and card.findings[0].evidence[0].quote == QUOTE
    assert seen == [None] and len(assistance.calls) == 1 and never.calls == []
    assert not isinstance(prod.traced.inner, TeamAssistant)


def test_the_setting_is_read_from_the_environment_and_a_bad_value_is_refused(monkeypatch):
    monkeypatch.setenv("CONTEXTLENS_ASSISTANT", "single")
    assert production.assistant_kind() == "single"
    monkeypatch.setenv("CONTEXTLENS_ASSISTANT", "both")
    with pytest.raises(ValueError, match="CONTEXTLENS_ASSISTANT"):
        production.assistant_kind()
    with pytest.raises(ValueError, match="assistant must be"):
        _build({}, assistant="both")
