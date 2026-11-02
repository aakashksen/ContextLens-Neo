import pytest
from pydantic import ValidationError

from contextlens.schemas import Card, Conversation, EvidenceRef, Finding, Turn


def _ev() -> EvidenceRef:
    return EvidenceRef(turn_id="t1", quote="we ship friday", speaker="A", start_ms=0)


def _finding(**kw) -> Finding:
    base = dict(agent="recall", kind="recall", statement="s", evidence=[_ev()], confidence=0.9)
    base.update(kw)
    return Finding(**base)


def test_card_rejects_unknown_status():
    with pytest.raises(ValidationError):
        Card(mode="recall", status="maybe", findings=[])


def test_answered_card_requires_a_finding():
    with pytest.raises(ValidationError):
        Card(mode="recall", status="answered", findings=[])


def test_finding_recall_requires_evidence():
    with pytest.raises(ValidationError):
        _finding(evidence=[])


def test_suggestion_is_separate_field():
    card = Card(mode="respond", status="answered", findings=[_finding()], suggestion="say X")
    assert card.suggestion == "say X"
    assert all("say X" not in f.statement for f in card.findings)
    assert Card(mode="recall", status="abstained", findings=[]).suggestion is None


def test_round_trip_json():
    card = Card(mode="recall", status="answered", findings=[_finding(label="update")], missing=["m"])
    assert Card.model_validate_json(card.model_dump_json()) == card
    turn = Turn(id="t1", conversation_id="c", seq=0, speaker="A", text="hi",
                start_ms=0, end_ms=5, source="synthetic")
    conv = Conversation(id="c", turns=[turn], source="x", split="dev")
    assert Conversation.model_validate_json(conv.model_dump_json()) == conv
