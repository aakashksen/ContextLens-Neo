"""Judge check, every claim of named systems: the cloud grader (through `claude -p`) grades all the claims of the systems
named, in the named dev records, and of no other system. Used after the E3T and E3TS runs, whose records hold the local
judge's verdicts.

    python -m contextlens.experiments.judge_check RUN_ID [RUN_ID ...] --systems single team_changed --all-claims

Each record may be of the old dev set (`scenarios`) or the new one (`scenarios_extra`). The result is one JC record per set
that holds a named system, with that system's saved rows plus the cloud verdict on every finding, and the same table in
`config.scores` on each: per system, the written score under the local judge and under the cloud grader on the old set, the new
set and both. A system that was run on only one set has `null` for `both`. The answers are saved as they arrive under
results/partial/judge-check/ (ids and verdicts only, never text), keyed by commit, so a stopped run resumes."""

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contextlens.eval import series
from contextlens.eval.check_run import recompute_intervals, recompute_metrics
from contextlens.eval.records import ROOT, head_commit, is_dirty, load_verified, write_run
from contextlens.experiments import e3_team as e3
from contextlens.experiments.e2_context import RUBRIC_VERSION, Judge
from contextlens.experiments.judge_check import (
    CALLS_PER_ASK_MAX,
    EXPERIMENT,
    PARTIAL_REL,
    SAVED_METRICS,
    SPLIT,
    Claim,
    Refused,
    _counter,
    _grade,
    _now,
    _read_record,
    open_rows,
    preflight,
)
from contextlens.llm.client import CLAUDE_CALL_CAP, CLAUDE_CALLS_PATH, LLMClient
from contextlens.replay.runner import Prediction
from contextlens.schemas import Scenario

SET_OF = {"scenarios": "old", "scenarios_extra": "new"}  # the sets a score is given on, besides "both"
SAVED = (*SAVED_METRICS, "answer_when_answerable", "abstain_when_insufficient")
INTERVALS = ("grounded_rate", "cloud_grounded_rate", "grounded_gap", "judge_agreement", "judge_kappa")
SCORE_NOTE = ("written score = mean(grounded rate, min(answer_when_answerable, abstain_when_insufficient)), as in "
              "docs/DECISIONS.md. 'local' takes the grounded rate from the local judge's verdicts and 'cloud' from the cloud "
              "grader's, on the same claims; the answer and abstain rates come from the cards and do not depend on a judge. "
              "old = the 148 dev cases of `scenarios`, new = the 108 dev cases of `scenarios_extra`, both = the two together.")


@dataclass(frozen=True)
class Held:
    """One system's saved rows from one record, and the claims in them."""
    run_id: str
    source: str
    system: str
    rows: list[Prediction]
    claims: list[Claim]


def load_systems(run_ids: list[str], systems: list[str], root: Path = ROOT) -> tuple[list[Held], dict[str, tuple[list[Scenario], str]], dict[str, Any]]:
    """The named systems' rows and every claim in them. Claims of any other system are never built, so never sent."""
    if not systems or len(set(systems)) != len(systems):
        raise Refused("name at least one system, each once")
    if not run_ids or len(set(run_ids)) != len(run_ids):
        raise Refused("name at least one run record, each once")
    recs = [_read_record(r, root) for r in run_ids]
    judges = {(r["config"]["judge"]["model"], r["config"]["judge"].get("same_model_and_prompt_as_writer")) for r in recs}
    if len(judges) > 1:
        raise Refused(f"the records were judged by different local judges: {sorted(map(str, judges))}")
    loaded: dict[str, tuple[list[Scenario], str]] = {}
    held: list[Held] = []
    for rec in recs:
        source = rec["scenario_source"]
        if source not in SET_OF:
            raise Refused(f"{rec['run_id']} is of source {source!r}; only {sorted(SET_OF)} rows hold the cards to rebuild claims from")
        if source not in loaded:
            loaded[source] = load_verified(source, SPLIT, root)
        if rec["dataset_hash"] != loaded[source][1]:
            raise Refused(f"{rec['run_id']} was made on a different dev scenario file than the one on disk")
        by_id = {s.id: s for s in loaded[source][0]}
        for system, rel in rec["predictions"].items():
            if system not in systems:
                continue
            if any(h.source == source and h.system == system for h in held):
                raise Refused(f"system {system} of the {source} set is in two of the named records, so its scores would be ambiguous")
            rows = [Prediction.model_validate_json(x) for x in (root / rel).read_text(encoding="utf-8").splitlines() if x.strip()]
            claims: list[Claim] = []
            for p in rows:
                if p.card is None or not p.card.findings:
                    continue
                if p.claim_supported is None or len(p.claim_supported) != len(p.card.findings):
                    raise Refused(f"{rec['run_id']} {system} {p.scenario_id}: the saved rows do not hold one local verdict per claim")
                sc = by_id.get(p.scenario_id)
                if sc is None:
                    raise Refused(f"{rec['run_id']} {system}: case {p.scenario_id} is not in the dev split of {source}")
                claims += [Claim(claim_id(rec["run_id"], system, p.scenario_id, i), rec["run_id"], rec["experiment"], system,
                                 p.scenario_id, sc.category, sc.origin, v, f)
                           for i, (f, v) in enumerate(zip(p.card.findings, p.claim_supported, strict=True))]
            held.append(Held(rec["run_id"], source, system, rows, claims))
    absent = [s for s in systems if not any(h.system == s for h in held)]
    if absent:
        raise Refused(f"no named record holds the system {absent}")
    overlap = set.intersection(*({s.id for s in scs} for scs, _ in loaded.values())) if len(loaded) > 1 else set()
    if overlap:
        raise Refused(f"the sets share case ids, so rows of the two could not be pooled: {sorted(overlap)[:3]}")
    coverage = {"records": {r["run_id"]: {"source": r["scenario_source"], "set": SET_OF[r["scenario_source"]],
                                          "systems_graded": [h.system for h in held if h.run_id == r["run_id"]],
                                          "systems_not_graded": sorted(set(r["predictions"]) - set(systems))} for r in recs},
                "claims": {f"{h.system}/{SET_OF[h.source]}": len(h.claims) for h in held}}
    return held, loaded, coverage


def claim_id(run_id: str, system: str, scenario_id: str, i: int) -> str:
    return f"{run_id}/{system}/{scenario_id}/{i}"


def _set_scores(scs: list[Scenario], rows: list[Prediction]) -> dict[str, Any]:
    m = recompute_metrics(scs, rows)
    base = {k: m.get(k) for k in ("grounded_rate", "answer_when_answerable", "abstain_when_insufficient")}
    return {"cases": len(scs), "claims": sum(len(p.card.findings) for p in rows if p.card), **base,
            "cloud_grounded_rate": m.get("cloud_grounded_rate"), "judge_agreement": m.get("judge_agreement"),
            "judge_kappa": m.get("judge_kappa"), "written_score_local": e3.score(base),
            "written_score_cloud": e3.score({**base, "grounded_rate": m.get("cloud_grounded_rate")})}


def scores_table(held: list[Held], loaded: dict[str, tuple[list[Scenario], str]]) -> dict[str, Any]:
    """Per system: the written score on the old set, the new set and both, from rows that carry the cloud verdicts."""
    out: dict[str, Any] = {}
    for h in held:
        entry = out.setdefault(h.system, {"covers": {"sets": [], "records": []}, "old": None, "new": None, "both": None})
        entry["covers"]["sets"].append(SET_OF[h.source])
        entry["covers"]["records"].append(h.run_id)
        entry[SET_OF[h.source]] = _set_scores(loaded[h.source][0], h.rows)
    for system, entry in out.items():
        mine = {h.source: h for h in held if h.system == system}
        if len(mine) == len(SET_OF):
            entry["both"] = _set_scores([s for src in SET_OF for s in loaded[src][0]], [p for src in SET_OF for p in mine[src].rows])
    return out


def _with_cloud(h: Held, verdicts: dict[str, bool]) -> list[Prediction]:
    out = []
    for p in h.rows:
        if p.card is not None and p.card.findings:
            cloud = [verdicts[claim_id(h.run_id, h.system, p.scenario_id, i)] for i in range(len(p.card.findings))]
            p = p.model_copy(update={"cloud_supported": cloud})
        out.append(p)
    return out


def run_all(run_ids: list[str], systems: list[str], *, llm: LLMClient, grader_name: str, root: Path = ROOT,
            counter_path: Path = CLAUDE_CALLS_PATH, cap: int = CLAUDE_CALL_CAP, say: Callable[[str], None] = print) -> list[Path]:
    held, loaded, coverage = load_systems(run_ids, systems, root)
    claims = [c for h in held for c in h.claims]
    if not claims:
        raise Refused("no claim in the named systems; see the per-system counts: " + json.dumps(coverage["claims"]))
    if is_dirty(root):
        raise Refused("working tree has uncommitted changes; commit first, so the rows and the record share a commit")
    commit = head_commit(root)
    batch = hashlib.sha256(json.dumps(sorted(c.id for c in claims)).encode()).hexdigest()[:12]
    header = {"kind": "judge-check-all-claims-rows", "batch_id": batch, "git_commit": commit, "run_ids": run_ids,
              "systems": systems, "n_claims": len(claims)}
    path = root / PARTIAL_REL / f"all-{batch}.rows.jsonl"
    rows, asks = open_rows(path, header)
    pending = [c for c in claims if rows.get(c.id, {}).get("status") != "ok"]
    say(f"{len(claims)} claims in the systems {systems}; {len(claims) - len(pending)} already graded in saved rows; "
        f"this run makes {len(pending)} grader calls (up to {len(pending) * CALLS_PER_ASK_MAX} if replies must be retried); "
        f"shared count {_counter(counter_path)} of {cap}")
    before = preflight(len(pending), counter_path, cap)
    started = _now()
    asks = _grade(pending, Judge(llm, tries=1), path, rows, asks)
    failed = [c.id for c in claims if rows[c.id]["status"] != "ok"]
    if failed:
        raise Refused(f"{len(failed)} claims got no cloud verdict; rows kept, rerun to resume, and no record is written")
    verdicts = {c.id: rows[c.id]["cloud"] for c in claims}
    graded = [Held(h.run_id, h.source, h.system, _with_cloud(h, verdicts), h.claims) for h in held]
    return _write_all(graded, loaded, coverage, systems, run_ids, commit, asks, before, counter_path, grader_name, started, root,
                      _asks_by_source(path, graded))


def _asks_by_source(path: Path, held: list[Held]) -> dict[str, int]:
    source_of = {c.id: h.source for h in held for c in h.claims}
    out: dict[str, int] = defaultdict(int)
    for ln in path.read_text(encoding="utf-8").splitlines()[1:]:
        try:
            out[source_of[json.loads(ln)["claim_id"]]] += 1
        except (ValueError, KeyError):
            continue
    return dict(out)


def _write_all(held: list[Held], loaded: dict[str, tuple[list[Scenario], str]], coverage: dict[str, Any], systems: list[str],
               run_ids: list[str], commit: str, asks: int, before: int, counter_path: Path, grader_name: str, started: str,
               root: Path, asks_by_source: dict[str, int]) -> list[Path]:
    table = scores_table(held, loaded)
    judge = _read_record(run_ids[0], root)["config"]["judge"]
    paths = []
    for source in (s for s in SET_OF if any(h.source == s for h in held)):
        scs, digest = loaded[source]
        mine = {h.system: h.rows for h in held if h.source == source}
        metrics = {n: {k: v for k, v in recompute_metrics(scs, r).items() if series.base_name(k) in SAVED} for n, r in mine.items()}
        intervals = {n: {k: v for k, v in recompute_intervals(scs, r, INTERVALS).items() if v is not None} for n, r in mine.items()}
        config = {
            "experiment": EXPERIMENT, "mode": "all_claims", "provisional": True, "commit_of_rows": commit, "set": SET_OF[source],
            "judge": {"model": judge["model"], "rubric_version": RUBRIC_VERSION,
                      "same_model_and_prompt_as_writer": bool(judge.get("same_model_and_prompt_as_writer"))},
            "cloud_grader": {"model": grader_name, "role": "grades every claim of the named systems, blind to the local verdict, "
                             "the writing system and the experiment"},
            "source_runs": run_ids, "systems": systems, "this_record_covers": {"set": SET_OF[source], "systems": sorted(mine)},
            "scores": table, "scores_note": SCORE_NOTE, "coverage": coverage,
            "claude_calls": asks_by_source.get(source, 0),
            "claude_calls_note": "asks made for this set's claims, one call each; the client may retry a reply that is not valid "
                                 "JSON, so calls used are between claude_calls and twice that. The whole run made "
                                 f"{asks} asks; the other set's record holds the rest.",
            "shared_counter_before": before, "shared_counter_after": _counter(counter_path), "started_at": started,
            "notes": ["Provisional: no claim comes from this record alone.",
                      "Every claim of each named system in this set was graded; no sample was drawn.",
                      "If the cloud grader also wrote some of the graded answers, its verdicts on those may favour its own wording."],
        }
        attempts = [{"setting": "cloud_grader", "kind": "grading", "ok": True, "error": None, "asks": asks_by_source.get(source, 0)}]
        paths.append(write_run(EXPERIMENT, SPLIT, config, {"judge": judge["model"], "cloud_grader": grader_name},
                               {"judge_rubric": RUBRIC_VERSION}, metrics, intervals, attempts, mine, digest,
                               scenario_source=source, root=root))
    return paths
