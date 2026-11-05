"""Judge check: the cloud grader (Sonnet, through `claude -p`) re-grades a seeded sample of claims the local judge
already graded, on dev only, to see whether the local judge can be trusted and whether it flatters results.

    python -m contextlens.experiments.judge_check RUN_ID [RUN_ID ...]
    python -m contextlens.experiments.judge_check RUN_ID [RUN_ID ...] --systems S [S ...] --all-claims   (judge_check_all.py)

Each RUN_ID is a dev run record whose saved rows hold the cards and the local judge's verdict per claim. The method
and its thresholds are in docs/DECISIONS.md. The rows are saved as they arrive under results/partial/judge-check/
(ids and verdicts only, never text), so a stopped run resumes at the same commit with the same sample. The result is
provisional: no claim comes from this record alone."""

import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contextlens.eval import series
from contextlens.eval.check_run import recompute_intervals, recompute_metrics, superseded_by
from contextlens.eval.records import (
    IDS_ONLY_SOURCES,
    ROOT,
    RUNS_REL,
    head_commit,
    is_dirty,
    load_verified,
    write_run,
)
from contextlens.experiments.e2_context import RUBRIC_VERSION, Judge, JudgeFailed
from contextlens.llm.client import CLAUDE_CALL_CAP, CLAUDE_CALLS_PATH, ClaudeCodeClient, LLMClient, _load_calls
from contextlens.replay.runner import Prediction
from contextlens.schemas import Card, Finding, Scenario

EXPERIMENT = "JC"
SPLIT = "dev"
SOURCE = "scenarios"  # the only source whose saved rows hold the statements and quotes the judge saw
SEED = 20261103  # fixed here, in the committed code, before any run
N_SAMPLE = 200
MIN_RARE = 40  # fewer of the local judge's rarer verdict than this in the natural sample adds a second stratum
POOR_KAPPA = 0.6  # kappa below this is poor agreement; see docs/DECISIONS.md for where the number comes from
CALLS_PER_ASK_MAX = 2  # one call, plus the client's one retry when a reply is not valid JSON
MAX_FAILURES_IN_A_ROW = 5
PARTIAL_REL = Path("results/partial/judge-check")
NATURAL, OVERSAMPLE = "natural", "oversample"
SAVED_METRICS = ("grounded_rate", *series.JUDGE_CHECK)
INTERVAL_METRICS = ("grounded_rate", "cloud_grounded_rate", "grounded_gap", "judge_agreement", "judge_kappa")
SAMPLING_RULE = (
    "Pool: every claim (one finding) in the named dev records that the local judge graded, with a claim that repeats "
    "the same system, case, statement and quotes in a later record dropped. Strata: record, system and case category. "
    "Natural sample: visit the strata in a seeded random order, take one random unused claim from each in turn, and go "
    "round again until 200 are taken or the pool is empty, so small strata run out first. Natural stratum: those "
    "claims, in their natural share of local verdicts within each stratum. Oversample stratum: if fewer than 40 of "
    "them carry the local judge's rarer verdict, random unsampled claims with that verdict are added up to 40 rare "
    "claims in all, or until none are left. Both strata are reported and the verdict on the local judge rests on the "
    "natural one."
)
KAPPA_BASIS = (
    "0.6 is the lower edge of the 'substantial' band in Landis and Koch (1977), rounded down from 0.61. It is a "
    "convention, not a value derived from this data; no better-grounded cut was found."
)


class Refused(RuntimeError):
    """The inputs or the state do not allow a meaningful check, so nothing is spent."""


@dataclass(frozen=True)
class Claim:
    id: str
    run_id: str
    experiment: str
    system: str
    scenario_id: str
    category: str
    origin: str
    local: bool  # the local judge's verdict
    finding: Finding

    @property
    def stratum_key(self) -> tuple[str, str, str]:
        return (self.run_id, self.system, self.category)


def _read_record(run_id: str, root: Path) -> dict[str, Any]:
    path = root / RUNS_REL / f"{run_id}.json"
    if not path.is_file():
        raise Refused(f"no run record {run_id} under {RUNS_REL.as_posix()}")
    rec = json.loads(path.read_text(encoding="utf-8"))
    if rec.get("run_id") != run_id:
        raise Refused(f"{path.name} holds run_id {rec.get('run_id')!r}")
    if rec.get("split") != SPLIT:
        raise Refused(f"{run_id} is on split {rec.get('split')!r}; only dev records may be graded here")
    gone = superseded_by(rec["experiment"], root)
    if run_id in gone:
        raise Refused(f"{run_id} is superseded by {gone[run_id][0]}: {gone[run_id][1]}")
    judge = (rec.get("config") or {}).get("judge")
    if not (isinstance(judge, dict) and judge.get("model") and judge.get("rubric_version") == RUBRIC_VERSION):
        raise Refused(f"{run_id} does not name a local judge with rubric {RUBRIC_VERSION}, so the question Sonnet would "
                      "be asked is not the one the local judge was asked")
    return rec


def _claims_of(rec: dict[str, Any], scenarios: dict[str, Scenario], root: Path) -> tuple[list[Claim], list[str]]:
    out: list[Claim] = []
    skipped: list[str] = []
    for system, rel in rec["predictions"].items():
        rows = [Prediction.model_validate_json(x)
                for x in (root / rel).read_text(encoding="utf-8").splitlines() if x.strip()]
        if not any(p.card is not None and p.card.findings for p in rows):
            skipped.append(system)
            continue
        for p in rows:
            if p.card is None or not p.card.findings:
                continue
            if p.claim_supported is None or len(p.claim_supported) != len(p.card.findings):
                raise Refused(f"{rec['run_id']} {system} {p.scenario_id}: the saved rows do not hold one local verdict "
                              "per claim, so the claim cannot be rebuilt as the judge saw it")
            sc = scenarios.get(p.scenario_id)
            if sc is None:
                raise Refused(f"{rec['run_id']} {system}: case {p.scenario_id} is not in the dev split")
            for i, (f, v) in enumerate(zip(p.card.findings, p.claim_supported, strict=True)):
                out.append(Claim(f"{rec['run_id']}/{system}/{p.scenario_id}/{i}", rec["run_id"], rec["experiment"],
                                 system, p.scenario_id, sc.category, sc.origin, v, f))
    return out, skipped


def load_pool(run_ids: list[str], root: Path = ROOT) -> tuple[list[Claim], dict[str, Any]]:
    """Every gradable claim in the named records, and what was found per record and source."""
    if not run_ids or len(set(run_ids)) != len(run_ids):
        raise Refused("name at least one run record, each once")
    recs = [_read_record(r, root) for r in run_ids]
    kinds = {(r["config"]["judge"]["model"], r["config"]["judge"].get("same_model_and_prompt_as_writer")) for r in recs}
    if len(kinds) > 1:
        raise Refused(f"the records were judged by different local judges: {sorted(map(str, kinds))}")
    scenarios: dict[str, Scenario] = {}
    digest = None
    pool: list[Claim] = []
    seen: set[tuple[Any, ...]] = set()
    coverage: dict[str, Any] = {"records": {}, "sources": defaultdict(lambda: {"records": 0, "claims": 0}),
                                "duplicates_dropped": 0}
    for rec in recs:
        source = rec["scenario_source"]
        entry = coverage["records"][rec["run_id"]] = {"source": source, "claims": 0, "systems_without_claims": []}
        coverage["sources"][source]["records"] += 1
        if source != SOURCE:
            why = "its rows hold ids and verdicts only" if source in IDS_ONLY_SOURCES else "its rows are not rebuildable"
            entry["note"] = f"no statements or quotes can be rebuilt: {why}"
            continue
        if not scenarios:
            loaded, digest = load_verified(SOURCE, SPLIT, root)
            scenarios = {s.id: s for s in loaded}
        if rec["dataset_hash"] != digest:
            raise Refused(f"{rec['run_id']} was made on a different dev scenario file than the one on disk")
        claims, skipped = _claims_of(rec, scenarios, root)
        entry["systems_without_claims"] = skipped
        for c in claims:
            key = (c.system, c.scenario_id, c.finding.statement, tuple((e.speaker, e.quote) for e in c.finding.evidence))
            if key in seen:
                coverage["duplicates_dropped"] += 1
                continue
            seen.add(key)
            pool.append(c)
            entry["claims"] += 1
            coverage["sources"][source]["claims"] += 1
    coverage["sources"] = dict(coverage["sources"])
    coverage["origins"] = {o: sum(c.origin == o for c in pool) for o in sorted({c.origin for c in pool})}
    return pool, coverage


@dataclass(frozen=True)
class Sample:
    natural: list[Claim]
    extra: list[Claim]
    rare_local_verdict: bool | None

    @property
    def all(self) -> list[Claim]:
        return [*self.natural, *self.extra]

    @property
    def sample_id(self) -> str:
        text = json.dumps([SEED, [c.id for c in self.natural], [c.id for c in self.extra]])
        return hashlib.sha256(text.encode()).hexdigest()[:12]


def draw_sample(pool: list[Claim], n: int = N_SAMPLE, seed: int = SEED, min_rare: int = MIN_RARE) -> Sample:
    """The natural sample by round-robin over strata, then the oversampled rare verdict if too few; see SAMPLING_RULE."""
    rng = random.Random(seed)
    strata: dict[tuple[str, str, str], list[Claim]] = defaultdict(list)
    for c in sorted(pool, key=lambda c: c.id):
        strata[c.stratum_key].append(c)
    order = sorted(strata)
    rng.shuffle(order)
    for key in order:
        rng.shuffle(strata[key])
    natural: list[Claim] = []
    while len(natural) < n and any(strata.values()):
        for key in order:
            if strata[key] and len(natural) < n:
                natural.append(strata[key].pop())
    if not natural:
        return Sample([], [], None)
    yes = sum(c.local for c in natural)
    rare = yes < len(natural) - yes  # the rarer local verdict: True means "supported" is the rare one
    have = len(natural) - yes if not rare else yes
    rest = [c for key in order for c in strata[key] if c.local is rare]
    rng.shuffle(rest)
    extra = rest[: max(0, min_rare - have)]
    return Sample(natural, extra, rare)


def stratum_of(sample: Sample) -> dict[str, str]:
    return {**{c.id: NATURAL for c in sample.natural}, **{c.id: OVERSAMPLE for c in sample.extra}}


def _fsync_append(path: Path, line: str) -> None:
    with open(path, "a", encoding="utf-8", newline="") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def open_rows(path: Path, header: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], int]:
    """The saved rows (the last row per claim wins) and how many asks were made so far. A new file starts with its
    header; an old one must carry the same header, which holds the commit and the sample."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        _fsync_append(path, json.dumps(header, sort_keys=True))
        return {}, 0
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or json.loads(lines[0]) != json.loads(json.dumps(header)):
        raise Refused(f"{path.name} was started with a different commit or sample; it cannot be resumed")
    rows: dict[str, dict[str, Any]] = {}
    asks = 0
    for ln in lines[1:]:
        try:
            row = json.loads(ln)
        except ValueError:  # a line cut short by a crash is a call that never counted as an answer
            continue
        rows[row["claim_id"]] = row
        asks += 1
    return rows, asks


def preflight(pending: int, counter_path: Path, cap: int) -> int:
    """The shared count of Claude calls, or Refused if the worst case for the pending asks would pass the cap."""
    used = _load_calls(counter_path)["count"]
    if used + pending * CALLS_PER_ASK_MAX > cap:
        raise Refused(f"{pending} asks could use up to {pending * CALLS_PER_ASK_MAX} calls and the shared count is "
                      f"{used} of {cap}; only the person can raise the cap")
    return used


def _card(claims: list[Claim]) -> Card:
    if not claims:
        return Card(mode="recall", status="abstained", findings=[], missing=["no claim of this system was sampled"])
    return Card(mode="recall", status="answered", findings=[c.finding for c in claims])


def build_systems(
    scored: list[tuple[Claim, bool, str]], scenarios: list[Scenario]
) -> dict[str, list[Prediction]]:
    """One row per dev case for each system. A system holds only the scored claims named for it, so a case with none
    appears as an abstained card with no findings; such rows carry no claim and do not enter any metric here."""
    groups: dict[str, list[tuple[Claim, bool]]] = {"all_natural": [], "all_with_oversample": []}
    for c, cloud, stratum in scored:
        groups["all_with_oversample"].append((c, cloud))
        if stratum == NATURAL:
            groups["all_natural"].append((c, cloud))
            groups.setdefault(f"exp_{c.experiment}", []).append((c, cloud))
            groups.setdefault(f"w_{c.run_id}_{c.system}", []).append((c, cloud))
    systems: dict[str, list[Prediction]] = {}
    for name, items in groups.items():
        by_case: dict[str, list[tuple[Claim, bool]]] = defaultdict(list)
        for c, cloud in items:
            by_case[c.scenario_id].append((c, cloud))
        rows = []
        for s in scenarios:
            got = by_case.get(s.id, [])
            rows.append(Prediction(
                scenario_id=s.id, card=_card([c for c, _ in got]),
                claim_supported=[c.local for c, _ in got] if got else None,
                cloud_supported=[v for _, v in got] if got else None))
        systems[name] = rows
    return systems


def tables(scored: list[tuple[Claim, bool, str]]) -> dict[str, Any]:
    """The 2x2 table of local against cloud verdicts, and each judge's share of supported, per stratum choice."""
    out: dict[str, Any] = {}
    for label, keep in (("natural", {NATURAL}), ("with_oversample", {NATURAL, OVERSAMPLE})):
        pairs = [(c.local, v) for c, v, st in scored if st in keep]
        n = len(pairs)
        cell = {f"local_{'yes' if a else 'no'}_cloud_{'yes' if b else 'no'}": sum(p == (a, b) for p in pairs)
                for a in (True, False) for b in (True, False)}
        out[label] = {"n": n, **cell,
                      "local_share_supported": sum(a for a, _ in pairs) / n if n else None,
                      "cloud_share_supported": sum(b for _, b in pairs) / n if n else None}
    return out


def verdict_on(kappa: float | None) -> str:
    if kappa is None:
        return "undetermined"
    return "poor" if kappa < POOR_KAPPA else "not poor"


def _notes(verdict: str, kappa: float | None, tab: dict[str, Any], n_scored: int, n_sample: int) -> list[str]:
    nat = tab["natural"]
    notes = ["Provisional: no claim comes from this record alone. It adds to the person-reviewed sample in "
             "docs/TESTING.md and does not replace it.",
             "A system here holds only the scored claims named for it; a case with none is an abstained card with no "
             "findings, so abstention and latency metrics in these rows mean nothing.",
             "If the cloud grader also wrote some of the graded answers, its verdicts on those may favour its own wording."]
    if n_scored < n_sample:
        notes.append(f"{n_sample - n_scored} sampled claims got no cloud verdict and are listed in config.sample.failed; "
                     "they are left out, not counted as disagreement.")
    if verdict == "poor":
        notes.append(f"Agreement is poor: Cohen's kappa {kappa:.3f} is below {POOR_KAPPA} on the natural sample, so the "
                     "local judge's grounded rates should not be taken at face value. Proposal: use Sonnet as the judge "
                     "for the final numbers.")
    elif verdict == "undetermined":
        notes.append("Cohen's kappa is undefined on the natural sample (one answer throughout from both judges, or no "
                     "scored claims), so agreement cannot be called poor or acceptable from this record.")
    else:
        notes.append(f"Cohen's kappa {kappa:.3f} is not below {POOR_KAPPA}; this is not evidence for replacing the local "
                     "judge, and the interval on judge_kappa says how firmly.")
    if nat["n"]:
        notes.append(f"Natural sample: the local judge called {nat['local_share_supported']:.3f} supported and the cloud "
                     f"grader {nat['cloud_share_supported']:.3f}. Kappa is unstable when one answer is much commoner "
                     "than the other, so read it with these shares and the 2x2 table.")
    return notes


def _grade(pending: list[Claim], judge: Judge, path: Path, rows: dict[str, dict[str, Any]], asks: int) -> int:
    """Ask the grader about each pending claim, saving every answer as it arrives; the number of asks made so far."""
    in_a_row = 0
    for c in pending:
        try:
            verdict = judge.supports(c.finding)
            row = {"claim_id": c.id, "status": "ok", "cloud": verdict}
            in_a_row = 0
        except JudgeFailed as exc:
            row = {"claim_id": c.id, "status": "failed", "error": str(exc)[:200]}
            in_a_row += 1
        _fsync_append(path, json.dumps(row, sort_keys=True))
        rows[c.id] = row
        asks += 1
        if in_a_row >= MAX_FAILURES_IN_A_ROW:
            raise Refused(f"{in_a_row} asks failed in a row; stopped with {len(rows)} rows kept, rerun to resume")
    return asks


def run(
    run_ids: list[str], *, llm: LLMClient, grader_name: str, root: Path = ROOT, counter_path: Path = CLAUDE_CALLS_PATH,
    cap: int = CLAUDE_CALL_CAP, n_sample: int = N_SAMPLE,
) -> Path:
    pool, coverage = load_pool(run_ids, root)
    sample = draw_sample(pool, n_sample)
    if not sample.natural:
        raise Refused("no gradable claim in the named records; see the per-source counts: " + json.dumps(coverage))
    if is_dirty(root):
        raise Refused("working tree has uncommitted changes; commit first, so the rows and the record share a commit")
    commit = head_commit(root)
    header = {"kind": "judge-check-rows", "sample_id": sample.sample_id, "git_commit": commit, "seed": SEED,
              "run_ids": run_ids, "n_natural": len(sample.natural), "n_oversample": len(sample.extra)}
    path = root / PARTIAL_REL / f"{sample.sample_id}.rows.jsonl"
    rows, asks = open_rows(path, header)
    pending = [c for c in sample.all if rows.get(c.id, {}).get("status") != "ok"]
    before = preflight(len(pending), counter_path, cap)
    started = _now()
    asks = _grade(pending, Judge(llm, tries=1), path, rows, asks)  # the same code that asked the local judge, so the same messages
    strata = stratum_of(sample)
    scored = [(c, rows[c.id]["cloud"], strata[c.id]) for c in sample.all if rows[c.id]["status"] == "ok"]
    failed = [c.id for c in sample.all if rows[c.id]["status"] != "ok"]
    if not any(st == NATURAL for _, _, st in scored):
        raise Refused(f"no natural-sample claim got a cloud verdict ({len(failed)} failed); rows kept, rerun to resume")
    scenarios, digest = load_verified(SOURCE, SPLIT, root)
    return _write(run_ids, sample, scored, failed, scenarios, digest, coverage, commit, asks, before, counter_path,
                  grader_name, started, root)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(
    run_ids: list[str], sample: Sample, scored: list[tuple[Claim, bool, str]], failed: list[str],
    scenarios: list[Scenario], digest: str, coverage: dict[str, Any], commit: str, asks: int, before: int,
    counter_path: Path, grader_name: str, started: str, root: Path,
) -> Path:
    systems = build_systems(scored, scenarios)
    metrics: dict[str, dict[str, float | None]] = {}
    for name, rows in systems.items():
        fresh = recompute_metrics(scenarios, rows)
        metrics[name] = {k: v for k, v in fresh.items() if series.base_name(k) in SAVED_METRICS}
    intervals: dict[str, dict[str, tuple[float, float]]] = {}
    for name in ("all_natural", "all_with_oversample"):
        got = recompute_intervals(scenarios, systems[name], INTERVAL_METRICS)
        intervals[name] = {k: v for k, v in got.items() if v is not None}
    kappa = metrics["all_natural"].get("judge_kappa")
    verdict = verdict_on(kappa)
    tab = tables(scored)
    src = _read_record(run_ids[0], root)["config"]["judge"]
    config = {
        "experiment": EXPERIMENT, "provisional": True, "seed": SEED, "sample_id": sample.sample_id,
        "judge": {"model": src["model"], "rubric_version": RUBRIC_VERSION,
                  "same_model_and_prompt_as_writer": bool(src.get("same_model_and_prompt_as_writer"))},
        "cloud_grader": {"model": grader_name, "role": "re-grades the claims the judge above graded, blind to its "
                         "verdict, the writing system and the experiment"},
        "source_runs": run_ids, "sampling_rule": SAMPLING_RULE,
        "sample": {"target": N_SAMPLE, "natural": len(sample.natural), "oversample": len(sample.extra),
                   "rare_local_verdict_is_supported": sample.rare_local_verdict, "min_rare": MIN_RARE,
                   "scored": len(scored), "failed": failed},
        "coverage": coverage, "tables": tab,
        "poor_if": {"metric": "judge_kappa", "below": POOR_KAPPA, "on": "all_natural", "basis": KAPPA_BASIS},
        "verdict": verdict,
        "claude_calls": asks, "claude_calls_max": asks * CALLS_PER_ASK_MAX,
        "claude_calls_note": "asks made, one call each; the client may retry a reply that is not valid JSON, so the "
                             "calls used are between claude_calls and claude_calls_max",
        "shared_counter_before": before, "shared_counter_after": _counter(counter_path),
        "started_at": started,
        "notes": _notes(verdict, kappa, tab, len(scored), len(sample.all)),
    }
    attempts = [{"setting": "cloud_grader", "kind": "grading", "ok": not failed, "error": None if not failed else
                 f"{len(failed)} claims got no cloud verdict", "asks": asks}]
    return write_run(EXPERIMENT, SPLIT, config, {"judge": src["model"], "cloud_grader": grader_name},
                     {"judge_rubric": RUBRIC_VERSION}, metrics, intervals, attempts, systems, digest,
                     scenario_source=SOURCE, root=root)


def _counter(path: Path) -> int:
    return _load_calls(path)["count"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    ap.add_argument("run_ids", nargs="+", help="dev run records to draw claims from")
    ap.add_argument("--systems", nargs="+", help="with --all-claims: grade every claim of these systems and of no other")
    ap.add_argument("--all-claims", action="store_true", help="grade every claim of the named systems instead of a sample")
    args = ap.parse_args(argv)
    if bool(args.systems) != args.all_claims:
        ap.error("--systems and --all-claims go together")
    llm = ClaudeCodeClient(model="sonnet")
    try:
        if args.all_claims:
            from contextlens.experiments.judge_check_all import run_all

            paths = run_all(args.run_ids, args.systems, llm=llm, grader_name=llm.model)
        else:
            paths = [run(args.run_ids, llm=llm, grader_name=llm.model)]
    except Refused as exc:
        print("refused:", exc, file=sys.stderr)
        return 1
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
