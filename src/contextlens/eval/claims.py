import json
import sys
from pathlib import Path

from contextlens.eval.check_run import check_record
from contextlens.eval.records import ROOT, RUNS_REL

CLAIMS_REL = Path("results/claims.json")


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
        if c.get("metric") not in rec.get("metrics", {}):
            problems.append(f"claim {cid}: metric {c.get('metric')} not in run")
        elif rec["metrics"][c["metric"]] is None:
            problems.append(f"claim {cid}: metric {c.get('metric')} is undefined (None) in the run")
        elif rec["metrics"][c["metric"]] != c.get("value"):
            problems.append(f"claim {cid}: value differs from the metric in the run")
        if not rec.get("labels_reviewed"):
            problems.append(f"claim {cid}: run has labels_reviewed false")
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
