"""Compare one system's saved rows between two commits of an interrupted team attempt. No model is called.

Reads results/partial/e3t-rows.jsonl (git-ignored). For each case saved at both commits it says whether the card's mode,
status and findings differ. It prints counts and case ids only, never transcript or model text.

  .venv/Scripts/python -m contextlens.experiments.partial_rows COMMIT_A COMMIT_B [--system single]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from contextlens.eval.records import ROOT
from contextlens.experiments.team_attempt import PARTIAL_REL
from contextlens.replay.runner import Prediction
from contextlens.schemas import Card


def saved_by_case(path: Path, commit: str, system: str) -> dict[str, Prediction]:
    """The first saved row of each case for this commit and system; a line cut short by a crash is skipped."""
    out: dict[str, Prediction] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
            if (e["commit"], e["winner"]) == (commit, system):
                row = Prediction.model_validate(e["row"])
                out.setdefault(row.scenario_id, row)
        except (ValueError, KeyError, TypeError):
            continue
    return out


def _findings(card: Card) -> list[tuple[Any, ...]]:
    return sorted((f.agent, f.kind, f.statement, f.label, tuple((e.turn_id, e.quote) for e in f.evidence)) for f in card.findings)


def compare(path: Path, commit_a: str, commit_b: str, system: str = "single") -> dict[str, Any]:
    a, b = saved_by_case(path, commit_a, system), saved_by_case(path, commit_b, system)
    both = sorted(set(a) & set(b))
    differ: dict[str, list[str]] = {"mode": [], "status": [], "findings": [], "card_missing": []}
    for sid in both:
        ca, cb = a[sid].card, b[sid].card
        if ca is None or cb is None:
            if (ca is None) != (cb is None):
                differ["card_missing"].append(sid)
            continue
        if ca.mode != cb.mode:
            differ["mode"].append(sid)
        if ca.status != cb.status:
            differ["status"].append(sid)
        if _findings(ca) != _findings(cb):
            differ["findings"].append(sid)
    any_diff = sorted({sid for ids in differ.values() for sid in ids})
    return {"system": system, "commits": [commit_a, commit_b], "rows": [len(a), len(b)], "cases_in_both": len(both),
            "only_in_first": sorted(set(a) - set(b)), "only_in_second": sorted(set(b) - set(a)),
            "differ": {k: {"count": len(v), "ids": v} for k, v in differ.items()},
            "identical": len(both) - len(any_diff), "any_difference": {"count": len(any_diff), "ids": any_diff}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("commit_a")
    ap.add_argument("commit_b")
    ap.add_argument("--system", default="single")
    args = ap.parse_args(argv)
    print(json.dumps(compare(ROOT / PARTIAL_REL, args.commit_a, args.commit_b, args.system), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
