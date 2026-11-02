import json
import sys
from pathlib import Path
from typing import Any

from contextlens.eval.check_run import check_record, review_problems, superseded_by
from contextlens.eval.records import ROOT, RUNS_REL
from contextlens.eval.series import JUDGED, base_name

CLAIMS_REL = Path("results/claims.json")
COMPARISON_FIELDS = ("diff", "ci", "mcnemar_p")


def _system(c: dict[str, Any], rec: dict[str, Any]) -> tuple[str | None, str | None]:
    """The system a claim is about; a record with several systems needs it named."""
    systems = list(rec.get("predictions") or {})
    name = c.get("system")
    if name is None:
        if len(systems) == 1:
            return systems[0], None
        return None, "names no system, and the run has several"
    if name not in systems:
        return None, f"system {name} is not in the run"
    return name, None


def _cited(c: dict[str, Any], rec: dict[str, Any]) -> tuple[object, str | None, str | None]:
    """(the record's value, the metric it belongs to, a problem)."""
    kind = c.get("kind", "metric")
    metric = c.get("metric")
    if kind == "comparison":
        for saved in rec.get("comparisons") or []:
            if (isinstance(saved, dict) and saved.get("a") == c.get("a") and saved.get("b") == c.get("b")
                    and saved.get("metric") == metric):
                if c.get("field") not in COMPARISON_FIELDS:
                    return None, metric, f"field must be one of {', '.join(COMPARISON_FIELDS)}"
                return saved.get(c["field"]), metric, None
        return None, metric, f"comparison of {metric} for {c.get('a')} vs {c.get('b')} not in run"
    if kind not in ("metric", "interval"):
        return None, metric, f"unknown kind {kind}"
    system, problem = _system(c, rec)
    if system is None:
        return None, metric, problem
    table = (rec.get("metrics") if kind == "metric" else rec.get("intervals") or {}).get(system) or {}
    if metric not in table:
        return None, metric, f"{kind} {metric} not in run"
    return table[metric], metric, None


def check_claims(root: Path = ROOT) -> list[str]:
    claims = json.loads((root / CLAIMS_REL).read_text(encoding="utf-8")).get("claims")
    if not isinstance(claims, list):
        return ["claims.json must have a 'claims' key holding a list"]
    problems: list[str] = []
    for c in claims:
        cid = c.get("id", "?")
        path = root / RUNS_REL / f"{c.get('run_id')}.json"
        if not path.is_file():
            problems.append(f"claim {cid}: no run record {c.get('run_id')}")
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        saved, metric, problem = _cited(c, rec)
        if problem:
            problems.append(f"claim {cid}: {problem}")
        elif saved is None:
            problems.append(f"claim {cid}: {metric} is undefined (None) in the run")
        elif saved != c.get("value"):
            problems.append(f"claim {cid}: value differs from the {c.get('kind', 'metric')} in the run")
        if metric and base_name(metric) in JUDGED:
            problems += [f"claim {cid}: {m}" for m in review_problems(rec.get("config"))]
        if not rec.get("labels_reviewed"):
            problems.append(f"claim {cid}: run has labels_reviewed false")
        if c.get("run_id") in superseded_by(rec.get("experiment", ""), root):
            problems.append(f"claim {cid}: run {c.get('run_id')} is superseded by a later record")
        problems += [f"claim {cid}: run: {m}" for m in check_record(rec, root)]
    return problems


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args != ["check"]:
        print("usage: python -m contextlens.eval.claims check")
        return 2
    problems = check_claims()
    for p in problems:
        print("FAIL", p)
    if not problems:
        print("claims ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
