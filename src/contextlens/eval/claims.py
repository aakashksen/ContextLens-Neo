import json
import sys
from pathlib import Path
from typing import Any

from contextlens.eval.check_run import _test_once_problems, check_record, review_problems, superseded_by
from contextlens.eval.records import ROOT, RUNS_REL, record_label_source
from contextlens.eval.series import JUDGED, base_name

CLAIMS_REL = Path("results/claims.json")
COMPARISON_FIELDS = ("diff", "ci", "mcnemar_p")
PUBLISHED_WORDS = "on published labels"


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


def _value_problem(c: dict[str, Any], rec: dict[str, Any]) -> tuple[str | None, str | None]:
    """(a problem with the claim's value against the record, the metric it cites)."""
    saved, metric, problem = _cited(c, rec)
    if problem:
        return problem, metric
    if saved is None:
        return f"{metric} is undefined (None) in the run", metric
    if saved != c.get("value"):
        return f"value differs from the {c.get('kind', 'metric')} in the run", metric
    return None, metric


def _label_problems(cid: str, c: dict[str, Any], rec: dict[str, Any]) -> list[str]:
    """Publisher labels count as reviewed (the person's rule), but the claim must say they are published labels.
    Our scenarios and AMI keep their sign-off, as recorded in the run. An unknown source is not reviewed."""
    kind = record_label_source(rec)
    if kind is None:
        return [f"claim {cid}: the run's label source cannot be determined, so its labels are not reviewed"]
    if kind == "publisher":
        text = c.get("text")
        if isinstance(text, str) and PUBLISHED_WORDS in text.lower():
            return []
        return [f"claim {cid}: the run's labels come from the publisher, so the claim's text must say '{PUBLISHED_WORDS}'"]
    return [] if rec.get("labels_reviewed") else [f"claim {cid}: run has labels_reviewed false"]


def _claim_problems(c: dict[str, Any], rec: dict[str, Any], root: Path, repeated: list[str],
                    record_problems: dict[str, list[str]]) -> list[str]:
    cid = c.get("id", "?")
    problem, metric = _value_problem(c, rec)
    problems = [f"claim {cid}: {problem}"] if problem else []
    if metric and base_name(metric) in JUDGED:
        problems += [f"claim {cid}: {m}" for m in review_problems(rec.get("config"))]
    problems += _label_problems(cid, c, rec)
    if c.get("run_id") in superseded_by(rec.get("experiment", ""), root):
        problems.append(f"claim {cid}: run {c.get('run_id')} is superseded by a later record")
    # The full record check re-bootstraps every interval, so it runs once per record in this call (never cached across runs).
    rid = str(c.get("run_id"))
    if rid not in record_problems:
        record_problems[rid] = check_record(rec, root)
    problems += [f"claim {cid}: run: {m}" for m in record_problems[rid]]
    problems += [f"claim {cid}: run: {m}" for m in repeated if m.startswith(f"{c.get('run_id')}:")]
    return problems


def check_claims(root: Path = ROOT) -> list[str]:
    claims = json.loads((root / CLAIMS_REL).read_text(encoding="utf-8")).get("claims")
    if not isinstance(claims, list):
        return ["claims.json must have a 'claims' key holding a list"]
    problems: list[str] = []
    repeated: list[str] | None = None
    record_problems: dict[str, list[str]] = {}
    for c in claims:
        path = root / RUNS_REL / f"{c.get('run_id')}.json"
        if not path.is_file():
            problems.append(f"claim {c.get('id', '?')}: no run record {c.get('run_id')}")
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("split") == "test" and repeated is None:
            repeated = _test_once_problems(root)  # computed once, and only when a claim cites a test record
        problems += _claim_problems(c, rec, root, repeated if rec.get("split") == "test" and repeated else [],
                                   record_problems)
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
