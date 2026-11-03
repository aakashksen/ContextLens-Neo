import re

from contextlens.llm.untrusted import effective_speaker
from contextlens.memory.store import TranscriptStore
from contextlens.schemas import Card, EvidenceRef, Finding

SCENE_PREFIX = "scene:"  # reserved: scene notes only. Never use it for a turn id; guard_card drops such citations


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def guard_card(
    card: Card, store: TranscriptStore, as_of_seq: int, conv_id: str
) -> Card:
    """Keep only findings whose every citation is a real turn at or before `as_of_seq` and quotes it."""
    known = {t.id: t for t in store.turns(conv_id, as_of_seq)}
    kept: list[Finding] = []
    for f in card.findings:
        refs: list[EvidenceRef] = []
        for e in f.evidence:
            t = known.get(e.turn_id)
            if e.turn_id.startswith(SCENE_PREFIX) or t is None or not e.quote.strip() or _norm(e.quote) not in _norm(t.text):
                break
            refs.append(
                e.model_copy(
                    update={"speaker": effective_speaker(t), "start_ms": t.start_ms}
                )
            )
        else:
            if refs:
                kept.append(f.model_copy(update={"evidence": refs}))
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
