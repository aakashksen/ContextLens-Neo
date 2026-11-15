"""Replay mode: AMI's own transcript with timings stands in for the speech service, and a plain word-overlap search
stands in for the assistants, so the room can be built and tested with no model. The timing rules here (what has been
said at a position, which turns a press may use) are the ones the live mode follows too."""

import re
import time

from contextlens.replay.runner import Context
from contextlens.room.meetings import content_words
from contextlens.schemas import ButtonRequest, Card, EvidenceRef, Finding, MemoryItem, Turn


def spoken_by(turns: list[Turn], position_ms: int) -> list[Turn]:
    """The turns that have finished by `position_ms`. A turn still being spoken is not used."""
    return [t for t in turns if t.end_ms <= position_ms]


def as_of_seq(turns: list[Turn], position_ms: int) -> int:
    """Seq of the newest finished turn, or -1 before anything has been said."""
    done = spoken_by(turns, position_ms)
    return done[-1].seq if done else -1


def turn_json(t: Turn, offset_ms: int = 0) -> dict:
    return {"id": t.id, "seq": t.seq, "speaker": t.speaker, "text": t.text,
            "start_ms": t.start_ms + offset_ms, "end_ms": t.end_ms + offset_ms}


def memory_json(m: MemoryItem, at_ms: int) -> dict:
    return {"id": m.id, "kind": m.kind, "subject": m.subject, "statement": m.statement, "speaker": m.speaker,
            "source_turn_ids": m.source_turn_ids, "supersedes": m.supersedes, "at_ms": at_ms}


class ReplayAssistant:
    """No model. The top turns by shared content words with the question (or with the latest turn), as a card whose
    findings quote those turns. It exists so the page, the highlighting and the as-of rule can be exercised."""

    def __init__(self, label: str = "replay") -> None:
        self.label = label

    def answer(self, req: ButtonRequest, ctx: Context) -> Card:
        t0 = time.perf_counter()
        turns = ctx.store.turns(req.conversation_id, ctx.as_of_seq)
        if not turns:
            return Card(mode="recall", status="abstained", findings=[], missing=["nothing has been said yet"])
        query = content_words(req.question or turns[-1].text)
        pool = turns if req.question else turns[:-1]
        scored = sorted(((len(query & content_words(t.text)), t) for t in pool), key=lambda p: (-p[0], -p[1].seq))
        top = [t for s, t in scored if s >= 1][:2]
        ctx.stage_ms["search_ms"] = int((time.perf_counter() - t0) * 1000)
        ctx.stage_ms["search_hits"] = len(top)
        if not top:
            return Card(mode="recall", status="abstained", findings=[], missing=["no earlier turn shares words with it"])
        top.sort(key=lambda t: t.seq)
        evidence = [EvidenceRef(turn_id=t.id, quote=re.sub(r"\s+", " ", t.text)[:160], speaker=t.speaker, start_ms=t.start_ms) for t in top]
        statement = "Closest earlier turns by shared words: " + "; ".join(f"{t.speaker} at {t.start_ms // 1000}s" for t in top) + "."
        finding = Finding(agent=self.label, kind="recall", confidence=0.5, evidence=evidence, statement=statement)
        return Card(mode="recall", status="answered", findings=[finding], latency_ms=ctx.stage_ms["search_ms"])
