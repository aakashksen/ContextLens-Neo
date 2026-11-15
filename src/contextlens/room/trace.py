"""The "how it got here" record of one press: stage timings, who contributed which finding, what the evidence guard
dropped and why, and why the card abstained. Built only from what the assistant already logs (`Context.stage_ms`,
`Context.drops`, the card); nothing here times or guesses anything the pipeline did not report."""

from contextlens.replay.runner import DroppedFinding
from contextlens.schemas import Card

SPECIALISTS = ("memory", "assistance", "consistency")
DROP_WHY = {
    "citation": "it cited a turn that is not in the transcript, or quoted words that turn does not contain",
    "instruction": "it cited a turn that gives orders to the assistant, which is not evidence",
    "number": "a number in it does not appear in the quotes it cites",
    "memory_empty": "memory looked and found nothing to support it",
}


def _ms(stage: dict[str, int], key: str) -> int | None:
    return int(stage[key]) if key in stage else None


def _specialists(stage: dict[str, int], card: Card) -> list[dict]:
    out = []
    routed = any(k.startswith("ran_") for k in stage)
    for name in SPECIALISTS:
        ran = bool(stage.get(f"ran_{name}")) if routed else f"{name}_ms" in stage
        out.append({"name": name, "ran": ran, "ms": _ms(stage, f"{name}_ms"), "tokens": _ms(stage, f"{name}_tokens"),
                    "findings": [i for i, f in enumerate(card.findings) if f.agent == name]})
    if "llm" in stage or "tools" in stage:  # the single agent has no specialists: one model, one tool loop
        out = [{"name": "single", "ran": True, "ms": int(stage.get("llm", 0)) + int(stage.get("tools", 0)),
                "llm_ms": _ms(stage, "llm"), "tools_ms": _ms(stage, "tools"), "tokens": None,
                "findings": [i for i, f in enumerate(card.findings) if f.agent not in SPECIALISTS]}]
    return out


def build_trace(name: str, card: Card, stage: dict[str, int], drops: list[DroppedFinding], *,
                speech_ms: int | None = None, speech_note: str = "") -> dict:
    """`speech_ms` is how long the newest caption before the press took after its audio ended (None if no speech
    service ran), `speech_note` says what stood in for it."""
    labels = [k[len("checker_"):] for k, v in stage.items() if k.startswith("checker_") and v == 1
              and k not in ("checker_ms", "checker_failed")]
    checker_ran = "checker_ms" in stage
    specialists = _specialists(stage, card)
    ran_ms = [s["ms"] for s in specialists if s["ran"] and s["ms"] is not None]
    stages = [
        {"id": "speech", "label": "Speech", "ms": speech_ms, "note": speech_note or ("caption delay" if speech_ms is not None else "not used")},
        {"id": "memory", "label": "Memory", "ms": _ms(stage, "memory_ms"), "note": "memory specialist" if "memory_ms" in stage else "not timed separately"},
        {"id": "search", "label": "Search", "ms": _ms(stage, "search_ms"), "note": f"{stage.get('search_hits', 0)} earlier turns" if "search_ms" in stage else "inside the tool calls"},
        {"id": "checker", "label": "Checker", "ms": _ms(stage, "checker_ms"), "note": (labels[0] if labels else ("failed" if stage.get("checker_failed") else ("ran" if checker_ran else "did not run")))},
        {"id": "specialists", "label": "Specialists", "ms": sum(ran_ms) if ran_ms else None, "note": ", ".join(s["name"] for s in specialists if s["ran"]) or "none ran"},
        {"id": "guard", "label": "Evidence guard", "ms": None, "note": f"{len(drops)} finding(s) dropped" if drops else "nothing dropped"},
    ]
    return {
        "assistant": name,
        "total_ms": card.latency_ms,
        "tokens": card.tokens,
        "stages": stages,
        "specialists": specialists,
        "checker": {"ran": checker_ran, "label": labels[0] if labels else None, "failed": bool(stage.get("checker_failed"))},
        "findings": [{"agent": f.agent, "statement": f.statement, "label": f.label, "turn_ids": [e.turn_id for e in f.evidence]} for f in card.findings],
        "drops": [{"reason": d.reason, "why": DROP_WHY.get(d.reason, d.reason), "statement": d.statement, "turn_ids": d.turn_ids} for d in drops],
        "abstained": {"yes": card.status == "abstained", "why": list(card.missing) if card.status == "abstained" else []},
        "notes": list(card.missing) if card.status != "abstained" else [],
    }
