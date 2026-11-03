"""E9: the small user study (kit only; the sessions themselves are human gate H4).

Each person does four staged tasks, recall and clarify, once with the assistant and once alone. `assign` gives
each person a counterbalanced order; `analyze` turns the anonymous results file into per-person paired
differences with an interval. It never writes to results/claims.json: a claim goes through the normal path.

Run: .venv/Scripts/python -m contextlens.experiments.e9_study plan 6
     .venv/Scripts/python -m contextlens.experiments.e9_study analyze results/study/results.csv --withdrawn P03
"""
import argparse
import csv
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from contextlens.eval.stats import paired_bootstrap_diff

KINDS = ("recall", "clarify")
VERSIONS = ("a", "b")
TASKS = {f"{k}_{v}": 2 for k in KINDS for v in VERSIONS}  # task id -> points available
CONDITIONS = ("assisted", "alone")
RESULT_FIELDS = ("participant_code", "task_id", "condition", "points")
MIN_PEOPLE, MAX_PEOPLE = 5, 8
# Two-sided 95% t critical values by degrees of freedom (n - 1).
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}
# Arrangements as (assisted_first, recall_first, assisted_version_is_a). Any first n of them (5 to 8) split each
# choice evenly, or off by one when n is odd, so the order is fixed here, not shuffled.
CELLS = ((0, 0, 0), (1, 1, 1), (0, 1, 1), (1, 0, 0), (1, 0, 1), (0, 1, 0), (1, 1, 0), (0, 0, 1))


@dataclass(frozen=True)
class Plan:
    code: str
    assisted_first: bool  # the first half of the session has the assistant
    kind_first: str  # which kind is done first inside each half
    assisted_version: str  # which staged conversation of each pair is done with the assistant

    def tasks(self) -> list[tuple[str, str]]:
        """The four (task_id, condition) in the order the person does them."""
        other = VERSIONS[1 - VERSIONS.index(self.assisted_version)]
        kinds = sorted(KINDS, key=lambda k: k != self.kind_first)
        assisted = [(f"{k}_{self.assisted_version}", "assisted") for k in kinds]
        alone = [(f"{k}_{other}", "alone") for k in kinds]
        return assisted + alone if self.assisted_first else alone + assisted


def assign(n: int) -> list[Plan]:
    if not MIN_PEOPLE <= n <= MAX_PEOPLE:
        raise ValueError(f"the study plans for {MIN_PEOPLE} to {MAX_PEOPLE} people, got {n}")
    return [
        Plan(f"P{i + 1:02d}", bool(af), KINDS[0] if kf else KINDS[1], VERSIONS[0] if av else VERSIONS[1])
        for i, (af, kf, av) in enumerate(CELLS[:n])
    ]


def _t_interval(diffs: list[float]) -> tuple[float, float] | None:
    n = len(diffs)
    if n < 2:
        return None
    half = T975.get(n - 1, 1.96) * statistics.stdev(diffs) / n**0.5
    mean = statistics.fmean(diffs)
    return mean - half, mean + half


def _kind_diffs(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    """code -> kind -> (assisted fraction - alone fraction); a kind missing either task is left out."""
    pts: dict[tuple[str, str, str], float] = {}
    for r in rows:
        kind = r["task_id"].split("_")[0]
        pts[(r["participant_code"], kind, r["condition"])] = int(r["points"]) / TASKS[r["task_id"]]
    out: dict[str, dict[str, float]] = {}
    for (code, kind, cond), v in pts.items():
        if cond == "assisted" and (code, kind, "alone") in pts:
            out.setdefault(code, {})[kind] = v - pts[(code, kind, "alone")]
    return out


def _summary(diffs: list[float]) -> dict:
    return {"n": len(diffs), "mean_diff": statistics.fmean(diffs) if diffs else None}


def analyze(rows: list[dict[str, str]], withdrawn: set[str] | frozenset[str] = frozenset()) -> dict:
    kept = [r for r in rows if r["participant_code"] not in withdrawn]
    by_person = _kind_diffs(kept)
    per_person = {c: statistics.fmean(k.values()) for c, k in sorted(by_person.items())}
    diffs = list(per_person.values())
    res: dict = {
        "n": len(diffs),
        "withdrawn_removed": len({r["participant_code"] for r in rows} & set(withdrawn)),
        "per_person": per_person,
        "per_kind": {k: _summary([d[k] for d in by_person.values() if k in d]) for k in KINDS},
        "mean_diff": statistics.fmean(diffs) if diffs else None,
        "t_interval": _t_interval(diffs),
        "bootstrap_interval": None,
    }
    if len(diffs) >= 2:
        # one pair per person, so resampling the groups resamples people
        _, lo, hi = paired_bootstrap_diff(diffs, [0.0] * len(diffs), list(per_person))
        res["bootstrap_interval"] = (lo, hi)
    return res


METHOD = (
    "Method: for each person and kind, the score fraction with the assistant minus the fraction alone; a person's "
    "difference is the mean over the kinds where both tasks exist (a missing task drops that kind for that "
    "person, nothing is filled in). The interval is a t interval on the per-person differences (95%). The "
    "project's paired bootstrap over people is shown beside it; with so few people it is narrower and less "
    "trustworthy. No p-value is computed."
)


def report(res: dict) -> str:
    n = res["n"]
    removed = f" ({res['withdrawn_removed']} withdrawn, their data removed)" if res["withdrawn_removed"] else ""
    lines = [f"People analysed: {n}{removed}"]
    if n < MIN_PEOPLE:
        lines.append(f"This is fewer than the {MIN_PEOPLE} people planned; read it as anecdote.")
    lines.append("Per person (assisted minus alone, fraction of points): "
                 + ", ".join(f"{c} {d:+.2f}" for c, d in res["per_person"].items()))
    for k, s in res["per_kind"].items():
        m = "n/a" if s["mean_diff"] is None else f"{s['mean_diff']:+.2f}"
        lines.append(f"{k}: {s['n']} people, mean difference {m}")
    ti = res["t_interval"]
    if ti is None:
        lines.append("Too few people for an interval.")
    else:
        lines.append(f"Mean difference {res['mean_diff']:+.2f}, 95% t interval [{ti[0]:+.2f}, {ti[1]:+.2f}]")
        if res["bootstrap_interval"]:
            b = res["bootstrap_interval"]
            lines.append(f"Bootstrap interval over people [{b[0]:+.2f}, {b[1]:+.2f}]")
        if ti[0] <= 0 <= ti[1]:
            lines.append("The interval includes zero: these people do not show a difference either way.")
        else:
            lines.append("The interval excludes zero for these people only; the sample is small and "
                         "self-selected, so it does not show the effect holds in general.")
    lines.append(METHOD)
    return "\n".join(lines)


def load_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if set(r) != set(RESULT_FIELDS):
            raise ValueError(f"unexpected columns {sorted(r)}; expected {list(RESULT_FIELDS)}")
        if r["task_id"] not in TASKS or r["condition"] not in CONDITIONS:
            raise ValueError(f"bad task or condition in row {r}")
        if not 0 <= int(r["points"]) <= TASKS[r["task_id"]]:
            raise ValueError(f"points out of range in row {r}")
    return rows


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="e9_study")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("n", type=int)
    a = sub.add_parser("analyze")
    a.add_argument("results", type=Path)
    a.add_argument("--withdrawn", nargs="*", default=[])
    args = ap.parse_args(argv)
    if args.cmd == "plan":
        for plan in assign(args.n):
            print(plan.code, " -> ".join(f"{t}({c})" for t, c in plan.tasks()))
    else:
        print(report(analyze(load_rows(args.results), set(args.withdrawn))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
