"""Writes site/room-data.json: the invented, scripted meeting the static page replays when no server answers.

Nothing here ran a model. Cards and traces follow the real shapes (Card, and room/trace.py's build_trace), but the
stage timings are illustrative and the page says so. Run: .venv/Scripts/python tools/make_room_data.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contextlens.replay.runner import DroppedFinding  # noqa: E402
from contextlens.room.trace import build_trace  # noqa: E402
from contextlens.schemas import Card, EvidenceRef, Finding  # noqa: E402

TURNS = [
    ("Maya", 0, 4000, "Let’s ship the beta on Friday."),
    ("Jordan", 4500, 9000, "Design still owes the onboarding screens."),
    ("Priya", 9500, 14000, "I can have the screens done by Wednesday."),
    ("Jordan", 14500, 19000, "Then QA needs two days, minimum."),
    ("Maya", 19500, 23000, "Okay, then make the beta ship Monday."),
    ("Sam", 24000, 29000, "The budget for the launch video is five thousand dollars."),
    ("Maya", 29500, 33000, "Fine, five thousand is approved."),
    ("Jordan", 34000, 38000, "I’ll send the QA plan tomorrow."),
    ("Priya", 39000, 44000, "Do we announce before or after the beta?"),
    ("Maya", 44500, 49000, "After. Priya should see the QA plan before we announce."),
    ("Sam", 50000, 55000, "The video budget has to come down to three thousand."),
    ("Maya", 55500, 59000, "Then I withdraw the five thousand. Make it three."),
    ("Jordan", 60000, 65000, "Who owns the press list?"),
    ("Sam", 65500, 70000, "Not me. I thought Priya had it."),
    ("Priya", 70500, 75000, "I don’t. Nobody has taken it."),
    ("Maya", 75500, 80000, "Okay, Sam, take the press list."),
]
T = [{"id": f"t{i}", "seq": i, "speaker": s, "text": x, "start_ms": a, "end_ms": b} for i, (s, a, b, x) in enumerate(TURNS, 1)]
BY = {t["id"]: t for t in T}

MEMORY = [
    ("m1", "decision", "Beta date", "The beta ships Monday, not Friday.", "Maya", ["t1", "t5"], None, 23000),
    ("m2", "decision", "Video budget", "The launch video budget is $5,000.", "Maya", ["t6", "t7"], None, 33000),
    ("m3", "commitment", "QA plan", "Jordan sends the QA plan tomorrow.", "Jordan", ["t8"], None, 38000),
    ("m4", "correction", "Video budget", "The launch video budget is $3,000, replacing $5,000.", "Maya", ["t11", "t12"], "m2", 59000),
    ("m5", "commitment", "Press list", "Sam owns the press list.", "Maya", ["t16"], None, 80000),
]
FLAGS = [
    {"turn_id": "t5", "label": "update", "confidence": 0.86, "earlier_ids": ["t1"], "at_ms": 23000},
    {"turn_id": "t11", "label": "contradiction", "confidence": 0.81, "earlier_ids": ["t7"], "at_ms": 55000},
    {"turn_id": "t12", "label": "update", "confidence": 0.9, "earlier_ids": ["t7", "t11"], "at_ms": 59000},
]


def ev(*ids):
    return [EvidenceRef(turn_id=i, quote=BY[i]["text"], speaker=BY[i]["speaker"], start_ms=BY[i]["start_ms"]) for i in ids]


def find(agent, statement, ids, label=None):
    return Finding(agent=agent, kind="recall", statement=statement, evidence=ev(*ids), label=label, confidence=0.9)


def result(name, status, findings, stage, *, missing=(), drops=(), suggestion=None, ms=0, speech=None):
    ms = sum(stage.get(k, 0) for k in ("llm", "tools", "memory_ms", "consistency_ms", "search_ms", "checker_ms"))  # the stages add up
    card = Card(mode="recall", status=status, findings=findings, missing=list(missing), suggestion=suggestion, latency_ms=ms, tokens=ms // 4)
    return {"assistant": name, "card": card.model_dump(mode="json"),
            "trace": build_trace(name, card, stage, list(drops), speech_ms=speech, speech_note="scripted, illustrative")}


def routed(findings, *, label=None, status="answered", missing=(), drops=(), ms=4800, suggestion=None, memory=True):
    stage = {"ran_memory": 1, "memory_ms": 1900, "memory_tokens": 420, "search_ms": 85, "search_hits": 6, "checker_ms": 380}
    if label:
        stage[f"checker_{label}"] = 1
        stage.update({"ran_consistency": 1, "consistency_ms": 1700, "consistency_tokens": 380})
    else:
        stage["checker_consistent"] = 1
    return result("team_routed", status, findings, stage, missing=missing, drops=drops, suggestion=suggestion, ms=ms, speech=640)


def plus(findings, *, label=None, status="answered", missing=(), drops=(), ms=3900, suggestion=None):
    stage = {"llm": 3300, "tools": 260, "search_ms": 85, "search_hits": 6, "checker_ms": 380, "checker_" + (label or "consistent"): 1}
    return result("single_plus", status, findings, stage, missing=missing, drops=drops, suggestion=suggestion, ms=ms, speech=640)


PRESSES = [
    {"key": "ship", "label": "When does the beta ship?", "after_ms": 4000, "results": [
        plus([find("single", "Maya proposed shipping the beta on Friday.", ["t1"])]),
        routed([find("memory", "Maya proposed shipping the beta on Friday.", ["t1"])])]},
    {"key": "ship", "label": "When does the beta ship?", "after_ms": 23000, "results": [
        plus([find("single", "Maya moved the beta from Friday to Monday after QA needed two days.", ["t1", "t4", "t5"], "update")], label="update", suggestion="Confirm Monday with design before announcing."),
        routed([find("memory", "The beta now ships Monday.", ["t5"]),
                find("consistency", "Monday replaces the Friday date Maya gave first.", ["t1", "t5"], "update")], label="update", suggestion="Confirm Monday with design before announcing.")]},
    {"key": "budget", "label": "What is the video budget?", "after_ms": 33000, "results": [
        plus([find("single", "Sam gave five thousand dollars and Maya approved it.", ["t6", "t7"])]),
        routed([find("memory", "The launch video budget is $5,000, approved by Maya.", ["t6", "t7"])])]},
    {"key": "budget", "label": "What is the video budget?", "after_ms": 59000, "results": [
        plus([find("single", "Sam asked for three thousand and Maya withdrew the five thousand.", ["t7", "t11", "t12"], "update")], label="update",
             drops=[DroppedFinding(reason="number", statement="The budget is $4,000.", turn_ids=["t11"])]),
        routed([find("memory", "The launch video budget is now $3,000.", ["t11", "t12"]),
                find("consistency", "Sam’s three thousand contradicted the approved five thousand; Maya then withdrew it.", ["t7", "t11", "t12"], "update")], label="update",
               drops=[DroppedFinding(reason="citation", statement="Maya approved four thousand.", turn_ids=["t7"])])]},
    {"key": "qa", "label": "Who owns the QA plan?", "after_ms": 38000, "results": [
        plus([find("single", "Jordan will send the QA plan tomorrow.", ["t8"])], status="partial", missing=["which day “tomorrow” is"]),
        routed([find("memory", "Jordan owns the QA plan and said he would send it tomorrow.", ["t8"])], status="partial", missing=["which day “tomorrow” is"])]},
    {"key": "press", "label": "Who owns the press list?", "after_ms": 80000, "results": [
        plus([find("single", "Maya assigned the press list to Sam after nobody had taken it.", ["t14", "t15", "t16"])]),
        routed([find("memory", "Sam owns the press list.", ["t16"])])]},
]
ABSTAIN = {  # used when no scripted press is ready yet: the card abstains rather than guess
    "single_plus": plus([], status="abstained", missing=["no turn so far answers that"], ms=2100),
    "team_routed": routed([], status="abstained", missing=["memory found nothing relevant"], ms=2600),
}


def main() -> None:
    data = {
        "id": "launch", "title": "Launch planning (invented)", "duration_ms": 82000, "turns": T,
        "memory": [{"id": i, "kind": k, "subject": s, "statement": st, "speaker": sp, "source_turn_ids": src, "supersedes": sup, "at_ms": at} for i, k, s, st, sp, src, sup, at in MEMORY],
        "flags": FLAGS, "presses": PRESSES, "abstain": ABSTAIN,
        "note": "Invented meeting, scripted cards, illustrative timings. No model ran.",
    }
    out = ROOT / "site" / "room-data.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
