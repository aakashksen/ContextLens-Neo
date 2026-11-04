"""What the saved card rows say about time and checks, per system: how many cards were cut short by the deadline, how
many hit a model error and how many of those errors were timeouts, and which findings the checks dropped, and why.

Computed from the rows alone (notes on the card, `stage_latency_ms`, `drops`), so an old record and a new one are read
the same way. Used by E2's and E3's records, with a before-and-after against the record they replace."""

from typing import Any

from contextlens.replay.runner import Prediction

DEADLINE_NOTES = ("deadline reached", "did not finish in time", "did not answer in time")
TIMEOUT_WORDS = ("timed out", "timeout")
SAMPLES = 3
SAMPLE_CHARS = 160
SLACK_MS = 500  # MARGIN_S of the agents: a card may be returned this long after the deadline


def _notes(p: Prediction) -> list[str]:
    return p.card.missing if p.card else []


def row_report(preds: list[Prediction], deadline_s: float) -> dict[str, Any]:
    """Counts of cards, not of notes: a card with two timeout notes counts once."""
    errors = timeouts = cut = over = 0
    samples: list[str] = []
    reasons: dict[str, int] = {}
    short = with_drops = 0
    for p in preds:
        notes = _notes(p)
        err = [n for n in notes if "llm error" in n]
        errors += bool(err)
        timeouts += any(w in n.lower() for n in err for w in TIMEOUT_WORDS)
        for n in err:
            text = n[:SAMPLE_CHARS]
            if text not in samples and len(samples) < SAMPLES:
                samples.append(text)
        cut += any(w in n for n in notes for w in DEADLINE_NOTES)
        over += p.stage_latency_ms.get("total", 0) > deadline_s * 1000 + SLACK_MS
        with_drops += bool(p.drops)
        for d in p.drops:
            reasons[d.reason] = reasons.get(d.reason, 0) + 1
            short += d.reason == "citation" and any(":" not in i for i in d.turn_ids)
    return {
        "cards": len(preds), "deadline_s": deadline_s,
        "cards_with_llm_error": errors, "cards_with_llm_timeout": timeouts,
        "cards_with_llm_error_not_timeout": errors - timeouts,
        "cards_cut_by_the_deadline": cut, "cards_returned_after_the_deadline_and_margin": over,
        "deadline_cut_note": "cards whose notes say the deadline was reached or a stage did not finish in time",
        "llm_error_samples": samples,
        "cards_with_drops": with_drops, "findings_dropped_by_reason": dict(sorted(reasons.items())),
        "citation_drops_with_a_short_id": short,
        "drops_saved_in_rows": any(p.drops for p in preds),
    }


def before_after(old: list[Prediction], new: list[Prediction], deadline_s: float) -> dict[str, Any]:
    """The same report for the superseded rows and the new rows. Old rows from before drops were saved show no drops:
    `drops_saved_in_rows` says which."""
    return {"before": row_report(old, deadline_s), "after": row_report(new, deadline_s)}
