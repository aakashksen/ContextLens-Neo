"""A two-stage consistency checker: the trained model screens every case, and only the cases it flags go to the
prompted checker, whose verdict decides.

What happens, stage by stage:
- Flagged means P(contradiction) from the trained model is at least `flag_threshold`.
- Not flagged: the answer is the trained model's best label other than contradiction (so never an alert), or
  "insufficient" when even that label is below `abstain_below`, as `TrainedChecker` abstains.
- Flagged: the second stage's verdict flows through. "contradiction" is the only alert; it keeps the trained
  probability as its confidence. Any other label, including "insufficient" (the second stage abstaining), is
  returned as it is and raises no alert.
- The second stage raising: fail closed. No unconfirmed alert; the answer is "insufficient", and the failure is kept
  in `records` and `summary()`.
- The first stage raising: the error propagates. Sending every case to the slow stage would silently change the cost
  and latency of the system.
"""
import time
from dataclasses import dataclass
from typing import Protocol

from contextlens.agents.consistency_check import ConsistencyChecker
from contextlens.schemas import ConsistencyLabel, Turn

DEFAULT_ABSTAIN_BELOW = 0.5  # the same default as contextlens.consistency.model.THRESHOLD


class ProbabilityChecker(Protocol):
    def probabilities(self, statement: Turn, earlier: list[Turn]) -> dict[ConsistencyLabel, float]: ...


def is_flagged(p_contradiction: float, flag_threshold: float) -> bool:
    """The one screening rule, shared with the offline threshold sweep."""
    return p_contradiction >= flag_threshold


@dataclass
class StageRecord:
    flagged: bool
    first_ms: float
    second_ms: float | None = None  # None: the second stage was not called
    second_tokens: int | None = None  # None: not called, or the checker did not report tokens
    second_error: str | None = None


class CascadeChecker:
    def __init__(self, first: ProbabilityChecker, second: ConsistencyChecker, flag_threshold: float,
                 abstain_below: float = DEFAULT_ABSTAIN_BELOW) -> None:
        self.first, self.second = first, second
        self.flag_threshold, self.abstain_below = flag_threshold, abstain_below
        self.records: list[StageRecord] = []

    def check(self, statement: Turn, earlier: list[Turn]) -> tuple[ConsistencyLabel, float]:
        start = time.perf_counter()
        p = self.first.probabilities(statement, earlier)
        rec = StageRecord(flagged=is_flagged(p["contradiction"], self.flag_threshold),
                          first_ms=(time.perf_counter() - start) * 1000)
        self.records.append(rec)
        if not rec.flagged:
            label = max((x for x in p if x != "contradiction"), key=p.__getitem__)
            return (label, p[label]) if p[label] >= self.abstain_below else ("insufficient", p["insufficient"])
        start = time.perf_counter()
        try:
            label, conf = self.second.check(statement, earlier)
        except Exception as exc:
            rec.second_ms = (time.perf_counter() - start) * 1000
            rec.second_error = f"{type(exc).__name__}: {exc}"
            return "insufficient", p["insufficient"]
        rec.second_ms = (time.perf_counter() - start) * 1000
        rec.second_tokens = getattr(self.second, "last_tokens", None)
        return (label, p["contradiction"]) if label == "contradiction" else (label, conf)

    def summary(self) -> dict[str, float | int]:
        """What was measured while checking: counts, and stage time and tokens summed over the calls made."""
        called = [r for r in self.records if r.second_ms is not None]
        return {
            "calls": len(self.records), "flagged": sum(r.flagged for r in self.records),
            "second_failed": sum(r.second_error is not None for r in self.records),
            "first_ms_total": sum(r.first_ms for r in self.records),
            "second_ms_total": sum(r.second_ms for r in called if r.second_ms is not None),
            "first_stage_llm_tokens": 0,  # the trained model is not a language model call
            "second_tokens_total": sum(r.second_tokens for r in called if r.second_tokens is not None),
            "second_tokens_unreported": sum(r.second_tokens is None for r in called),
        }
