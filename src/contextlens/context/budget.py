import math

from pydantic import BaseModel


class Section(BaseModel):
    text: str
    seq: int
    turn_id: str | None = None
    evidence: bool = False


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def fit_context(sections: list[Section], max_tokens: int) -> list[Section]:
    """Drop the oldest non-evidence sections until the rest fit; evidence is always kept."""
    keep = list(sections)
    total = sum(estimate_tokens(s.text) for s in keep)
    for s in sorted(sections, key=lambda s: s.seq):
        if total <= max_tokens:
            break
        if not s.evidence:
            keep.remove(s)
            total -= estimate_tokens(s.text)
    return keep
