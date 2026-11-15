"""Code that runs before the first model call of `team_routed` and `single_plus`: a search over the transcript with the
question (or the latest turn), and the consistency checker on the latest turn against the candidate earlier turns."""

import functools
import time
from dataclasses import dataclass, field

from contextlens.agents.consistency_check import ConsistencyChecker
from contextlens.agents.single import MARGIN_S, _bounded
from contextlens.agents.tools import Tools
from contextlens.llm.untrusted import escape
from contextlens.prompts.registry import render_transcript_block
from contextlens.schemas import ButtonRequest, ConsistencyLabel, Turn

HITS_K = 8
EARLIER_CAP = 12  # earlier turns the checker sees: every search hit first, then the newest window turns

CONFLICT = (
    "Report the conflict as one finding that cites both turns involved, the earlier statement and the latest one "
    "(at least two evidence entries, one quote from each)."
)


@dataclass
class Prep:
    context: str = ""  # appended to the model's first user message
    label: ConsistencyLabel | None = None  # the checker's verdict; None if it did not run or failed
    confidence: float = 0.0
    checker_ms: int = 0
    search_ms: int = 0
    hits: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return self.label in ("contradiction", "update")

    def stage(self) -> dict[str, int]:
        """Ints only, for `Context.stage_ms`: times, hit count, and the verdict as a `checker_<label>` flag."""
        out = {"checker_ms": self.checker_ms, "search_ms": self.search_ms, "search_hits": self.hits}
        if self.label:
            out[f"checker_{self.label}"] = 1
        if any(n.startswith("checker failed") for n in self.notes):
            out["checker_failed"] = 1
        return out


def prepare(tools: Tools, req: ButtonRequest, turns: list[Turn], end: float) -> Prep:
    """Never raises: a search or checker that fails or runs out of time is noted and skipped."""
    prep = Prep()
    if not turns:
        return prep
    statement, window = turns[-1], turns[:-1]
    query = (req.question or "").strip() or statement.text
    t0 = time.perf_counter()
    try:
        scored = _bounded(functools.partial(tools.search_transcript, query, HITS_K), max(end - t0, 0.1) + MARGIN_S)
    except Exception as exc:
        scored = []
        prep.notes.append(f"search failed: {type(exc).__name__}")
    prep.search_ms = int((time.perf_counter() - t0) * 1000)
    hits = [h.turn for h in scored if h.turn.seq < statement.seq]
    prep.hits = len(hits)
    shown = {t.id for t in turns}
    far = [t for t in hits if t.id not in shown]
    if far:
        prep.context += "\nEarlier turns found by searching the whole conversation:\n" + render_transcript_block(far)
    if tools.checker is not None:
        keep = {t.id: t for t in hits}
        for t in reversed(window):
            if len(keep) >= EARLIER_CAP:
                break
            keep.setdefault(t.id, t)
        earlier = sorted(keep.values(), key=lambda t: t.seq)
        t0 = time.perf_counter()
        try:
            label, conf = _bounded(functools.partial(tools.checker.check, statement, earlier),
                                   max(end - t0, 0.1) + MARGIN_S)
            prep.label, prep.confidence = label, conf
        except Exception as exc:  # fail safe: no verdict, so nothing is flagged
            prep.notes.append(f"checker failed: {type(exc).__name__}")
        prep.checker_ms = int((time.perf_counter() - t0) * 1000)
    if prep.label:
        prep.context += (f"\nConsistency check, run in code on the latest turn against earlier turns: "
                         f"{escape(prep.label)} (confidence {prep.confidence:.2f}). ")
        if prep.flagged:
            prep.context += CONFLICT
    return prep
