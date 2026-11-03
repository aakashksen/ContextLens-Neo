"""E6: which prompt choices move groundedness and abstention?

One strategy (`retrieval_memory`, the one E2 carried forward), one writer model, the whole dev split, and the same
cases for every prompt. The baseline is prompt version v1. Every other version changes exactly one factor of it, and
says which in its `factor.json` (see `declared_versions`). Nothing else is varied: the memory notes are built once per
case and shared by all versions, and the judge is the one E2 used.

The selection rule is `SELECTION_RULE`; it is fixed here, in a commit before any run. The test split is never read.

Run: .venv/Scripts/python -m contextlens.experiments.e6_prompts
"""

import difflib
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contextlens.prompts.registry import get_prompt

BASELINE = "v1"
FACTORS = ("few_shot", "evidence_format", "abstention", "output_format", "length_limit", "role_split")
VERSIONS = ("v1", "v2", "v3", "v4", "v5", "v6", "v7")  # the order they run in, and the tie-break order
PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
PROMPT_NAME = "recall"

# Not a factor here: a prompt that defines the card modes (recall, clarify, respond). T3.3 tried one and reverted it
# (record E2-20261103T155026435232Z); T4.2 sets the respond mode in code. The record still reports respond mode and
# injected claims, in findings and in suggestions, for every version.
GATE_METRICS = ("grounded_rate", "abstain_when_insufficient", "answer_when_answerable")
GAIN_METRICS = ("grounded_rate", "abstain_when_insufficient")
SELECTION_RULE = (
    "Each candidate version is compared with the baseline v1 on the same cases. score(v) is the mean of grounded_rate "
    "and the lower of abstain_when_insufficient and answer_when_answerable (the weaker abstention side); a version "
    "whose three metrics are not all defined is not eligible. A candidate qualifies only if (1) it has a measurable gain: "
    "the 95% interval of its difference from v1 on grounded_rate or on abstain_when_insufficient lies above zero, "
    "(2) it has no measurable loss: the interval of its difference from v1 on none of grounded_rate, "
    "abstain_when_insufficient and answer_when_answerable lies below zero, and (3) its score is above v1's. The chosen "
    "version is the qualifier with the highest score, ties broken by lower median latency, then by the order of VERSIONS. "
    "With no qualifier the baseline v1 is kept and the record says there was no measurable improvement. A version whose "
    "output was invalid or timed out too often to be measured is listed in attempts and cannot qualify. Respond mode and "
    "injected claims are reported for every version and do not enter the score. No correction is made for comparing six "
    "candidates, so a qualifier is a candidate for a confirming run, not a proven improvement."
)


@dataclass(frozen=True)
class Version:
    version: str
    factor: str
    summary: str
    added: tuple[str, ...]
    removed: tuple[str, ...]
    text: str

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def _lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.strip()]


def line_diff(baseline: str, other: str) -> tuple[list[str], list[str]]:
    """(added, removed) non-blank lines of `other` against `baseline`, each in text order."""
    d = list(difflib.ndiff(_lines(baseline), _lines(other)))
    return [x[2:] for x in d if x.startswith("+ ")], [x[2:] for x in d if x.startswith("- ")]


def declared_versions(prompts_dir: Path = PROMPTS_DIR, names: tuple[str, ...] = VERSIONS) -> list[Version]:
    """Every version, read through the registry, with the factor its `factor.json` declares. Raises ValueError when a
    version does not differ from v1 by exactly its one declared factor: an unknown or repeated factor, a declared
    change that is not the real line difference, or a text another version already has."""
    out: list[Version] = []
    for name in names:
        meta = json.loads((prompts_dir / name / "factor.json").read_text(encoding="utf-8"))
        text = get_prompt(PROMPT_NAME, name).text
        out.append(Version(name, meta["factor"], meta["summary"], tuple(meta["added"]), tuple(meta["removed"]), text))
    base = out[0]
    if base.version != BASELINE or base.factor != "baseline" or base.added or base.removed:
        raise ValueError(f"{BASELINE} must be the baseline and declare no change")
    seen_factors: set[str] = set()
    seen_hashes = {base.text_sha256: base.version}
    for v in out[1:]:
        if v.factor not in FACTORS:
            raise ValueError(f"{v.version}: factor {v.factor!r} is not one of {FACTORS}")
        if v.factor in seen_factors:
            raise ValueError(f"{v.version}: factor {v.factor} is already tried by another version")
        seen_factors.add(v.factor)
        added, removed = line_diff(base.text, v.text)
        if (list(v.added), list(v.removed)) != (added, removed) or not (added or removed):
            raise ValueError(f"{v.version}: the declared change is not its difference from {BASELINE}")
        if v.text_sha256 in seen_hashes:
            raise ValueError(f"{v.version} has the same text as {seen_hashes[v.text_sha256]}")
        seen_hashes[v.text_sha256] = v.version
    return out


def score(m: dict[str, float | None]) -> float | None:
    """Mean of grounded_rate and the weaker abstention side; None unless all three metrics are defined."""
    g, a, r = (m.get(k) for k in ("grounded_rate", "abstain_when_insufficient", "answer_when_answerable"))
    if g is None or a is None or r is None:
        return None
    return (g + min(a, r)) / 2


def _interval(comparisons: list[dict[str, Any]], candidate: str, metric: str) -> list[float] | None:
    for c in comparisons:
        if (c["a"], c["b"], c["metric"]) == (candidate, BASELINE, metric):
            return c["ci"]
    return None


def pick_best(metrics: dict[str, dict[str, float | None]], comparisons: list[dict[str, Any]],
              order: tuple[str, ...] = VERSIONS) -> dict[str, Any]:
    """Apply SELECTION_RULE. `comparisons` holds candidate-minus-baseline entries (`a` the candidate, `b` v1).
    Returns {"selected", "scores", "qualifies"} where qualifies maps each candidate to the reason it did or did not."""
    scores = {v: score(metrics[v]) for v in order if v in metrics}
    base = scores.get(BASELINE)
    if base is None:
        raise ValueError(f"{BASELINE} has an undefined metric, so nothing can be compared with it")
    qualifies: dict[str, str] = {}
    for v in order:
        if v == BASELINE or v not in metrics:
            continue
        ci = {m: _interval(comparisons, v, m) for m in GATE_METRICS}
        if scores[v] is None:
            qualifies[v] = "not eligible: a metric is undefined"
        elif any(ci[m] is not None and ci[m][1] < 0 for m in GATE_METRICS):
            qualifies[v] = "no: a measurable loss on " + ", ".join(m for m in GATE_METRICS if ci[m] and ci[m][1] < 0)
        elif not any(ci[m] is not None and ci[m][0] > 0 for m in GAIN_METRICS):
            qualifies[v] = "no: no measurable gain on grounded_rate or abstain_when_insufficient"
        elif scores[v] <= base:  # type: ignore[operator]
            qualifies[v] = "no: score not above the baseline's"
        else:
            qualifies[v] = "yes"
    eligible = [v for v, why in qualifies.items() if why == "yes"]
    if not eligible:
        return {"selected": BASELINE, "scores": scores, "qualifies": qualifies}
    best = min(eligible, key=lambda v: (-scores[v], metrics[v].get("latency_p50_ms") or 0.0, order.index(v)))  # type: ignore[operator]
    return {"selected": best, "scores": scores, "qualifies": qualifies}
