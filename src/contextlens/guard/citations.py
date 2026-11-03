import re

from contextlens.guard.instructions import is_instruction
from contextlens.llm.untrusted import effective_speaker
from contextlens.memory.store import TranscriptStore
from contextlens.schemas import Card, EvidenceRef, Finding

SCENE_PREFIX = "scene:"  # reserved: scene notes only. Never use it for a turn id; guard_card drops such citations


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def check_findings(
    card: Card, store: TranscriptStore, as_of_seq: int, conv_id: str
) -> tuple[list[Finding], list[tuple[Finding, str]]]:
    """(kept findings with evidence rewritten from the store, dropped findings with the reason).
    Dropped for `citation`: a citation is not a real quote at or before `as_of_seq`. Dropped for `instruction`:
    a cited turn gives orders to the assistant (guard/instructions.py), and an order is not evidence of a fact."""
    known = {t.id: t for t in store.turns(conv_id, as_of_seq)}
    kept: list[Finding] = []
    dropped: list[tuple[Finding, str]] = []
    for f in card.findings:
        refs: list[EvidenceRef] = []
        why = ""
        for e in f.evidence:
            t = known.get(e.turn_id)
            if e.turn_id.startswith(SCENE_PREFIX) or t is None or not e.quote.strip() or _norm(e.quote) not in _norm(t.text):
                why = "citation"
                break
            if is_instruction(t.text):
                why = "instruction"
                break
            refs.append(e.model_copy(update={"speaker": effective_speaker(t), "start_ms": t.start_ms}))
        if not why and refs:
            kept.append(f.model_copy(update={"evidence": refs}))
        else:
            dropped.append((f, why or "citation"))
    return kept, dropped


def guard_card(
    card: Card, store: TranscriptStore, as_of_seq: int, conv_id: str
) -> Card:
    """Keep only findings that `check_findings` keeps."""
    kept, _ = check_findings(card, store, as_of_seq, conv_id)
    if not kept:
        return card.model_copy(
            update={"status": "abstained", "findings": [], "suggestion": None}
        )
    status = (
        "partial"
        if len(kept) < len(card.findings) and card.status == "answered"
        else card.status
    )
    return card.model_copy(update={"findings": kept, "status": status})
