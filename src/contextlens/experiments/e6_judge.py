"""E6, the judging stage: judge the cards a `--generate-only` run saved, and write the E6 record. No local model, no Ollama.

    python -m contextlens.experiments.e6_judge --from-commit <sha> [--system single] [--rows PATH] [--sonnet-check]

`--from-commit` names the commit that generated the cards (the saved rows are keyed by it). The stage refuses unless the prompts
and the code that shaped the cards are the same at that commit and at HEAD (`CARD_CODE`, compared with git). Every claim of every
saved version is judged by Haiku through `claude -p`, one claim per call, in parallel, with rubric e2-v1 (batch mode was not
calibrated). Answers are saved as they arrive under results/partial/e6-judge/, so a stop resumes without regrading. Then
SELECTION_RULE is applied unchanged and the record is written through the same function the local run uses.

`--rows` defaults to the rows file under the working tree's root; the generating run's rows are git-ignored and stay on the disk
of the worktree that ran it, so pass its path when this runs elsewhere.

`--sonnet-check` also has Sonnet grade every claim of the chosen version and of v1 on the same rows. Those verdicts go in
`cloud_supported` (as in judge_check_all), never in `claim_supported`, and `config.sonnet_check` says whether SELECTION_RULE picks
the same version under them. The test split is never read."""

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contextlens.data.scenarios import load_scenarios
from contextlens.eval.records import ROOT, SCENARIO_SOURCES, head_commit, is_dirty, split_hash
from contextlens.experiments import e2_context as e2
from contextlens.experiments import e6_prompts as e6
from contextlens.experiments import parallel_grader as pg
from contextlens.experiments.judge_calibrate import UsageGuard, UsageUnreadable
from contextlens.experiments.judge_check import Refused, preflight
from contextlens.llm.client import CLAUDE_CALL_CAP, CLAUDE_CALLS_PATH, FakeLLM
from contextlens.replay.runner import Prediction
from contextlens.schemas import Scenario

CARD_CODE = (  # what shapes a card; e6_prompts.py is not here because its edits since the run do not touch how cards are made
    "src/contextlens/prompts", "src/contextlens/agents", "src/contextlens/memory", "src/contextlens/retrieval",
    "src/contextlens/llm", "src/contextlens/replay", "src/contextlens/data", "src/contextlens/schemas.py",
    "src/contextlens/experiments/e2_context.py", e2.CONFIG_REL.as_posix(),
)
JUDGE_DIR_REL = Path("results/partial/e6-judge")
MAX_WEEKLY_PCT = 90.0
CALIBRATION_NOTE = (
    "Haiku failed the written calibration test (docs/DECISIONS.md entry 56, record JCAL-20261105T044839474906Z, 300 claims): "
    "agreement with Sonnet 0.813 against the local judge's 0.870, a paired difference of -0.057 (95% interval -0.110 to -0.006, "
    "wholly below zero), so it counts as clearly worse under the rule as written. It is still this record's judge by the "
    "coordinator's decision, because local judging was ruled out. The explanation (the local judge and Sonnet both say "
    "'supported' to nearly everything, so the base rate helps the local agreement) is a reading, not proof. Sonnet is a "
    "reference, not ground truth. grounded_rate and unsupported_claim_rate come from Haiku through `claude -p`, one claim per "
    "call; no person has reviewed a sample, so they are not claims."
)


class JudgeStopped(RuntimeError):
    """The usage guard, a rate limit or the call cap stopped grading. Every answer so far is saved; rerun to resume."""


def resolve_commit(root: Path, ref: str) -> str:
    out = subprocess.run(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=root, capture_output=True, text=True)
    if out.returncode != 0:
        raise Refused(f"{ref!r} is not a commit of this repository")
    return out.stdout.strip()


def card_code_changed(root: Path, commit: str) -> list[str]:
    """Files of CARD_CODE that differ between `commit` and HEAD."""
    out = subprocess.run(["git", "diff", "--name-only", commit, "HEAD", "--", *CARD_CODE], cwd=root, capture_output=True, text=True)
    if out.returncode != 0:
        raise Refused(f"git could not compare {commit[:12]} with HEAD: {out.stderr.strip()[:200]}")
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def all_saved_rows(path: Path, commit: str, system: str, ids: set[str]) -> dict[str, dict[str, Prediction]]:
    """Every saved row of this commit and system by version and case, broken ones included, the last line of a case winning.
    A run that was not interrupted keeps a broken row and `_health` counts it; dropping it here would flatter the version.
    `e6._saved_rows`, which drops broken rows so a resume retries them, is unchanged."""
    have: dict[str, dict[str, Prediction]] = {}
    if not path.exists():
        return have
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
            row = Prediction.model_validate(e["row"])
            if (e["commit"], e["system"]) == (commit, system) and row.scenario_id in ids:
                have.setdefault(e["version"], {})[row.scenario_id] = row
        except (ValueError, KeyError, TypeError):
            continue
    return have


def claim_id(system: str, version: str, scenario_id: str, i: int) -> str:
    return f"{system}/{version}/{scenario_id}/{i}"


def items_of(system: str, rows: dict[str, list[Prediction]]) -> list[pg.Item]:
    return [pg.Item(claim_id(system, v, p.scenario_id, i), f.statement, tuple((e.speaker, e.quote) for e in f.evidence))
            for v, ps in rows.items() for p in ps if p.card is not None for i, f in enumerate(p.card.findings)]


def grade(items: list[pg.Item], llm: Any, path: Path, header: dict[str, Any], guard: UsageGuard, *, counter_path: Path,
          cap: int, concurrency: int, say: Callable[[str], None], sleep: Callable[[float], None] = time.sleep) -> dict[str, bool]:
    """Verdicts by claim id, from saved rows plus calls for the claims that have none. Raises JudgeStopped if grading stops
    early and Refused if the call cap could be passed or a claim ends with no valid verdict."""
    store = pg.RowStore(path, header)
    todo = pg.pending(items, store)
    say(f"{llm.model}: {len(items)} claims, {len(items) - len(todo)} already graded in saved rows, {len(todo)} calls to make")
    preflight(len(todo), counter_path, cap)
    out = pg.grade_single(items, llm, store, concurrency=concurrency, should_stop=guard.check, sleep=sleep)
    if out.stop_reason:
        raise JudgeStopped(f"{out.stop_reason}; {len(items) - len(pg.pending(items, store))} of {len(items)} claims graded and saved, rerun to resume")
    left = pg.ungraded(items, store)
    if left:
        raise JudgeStopped(f"{len(left)} claims got no valid verdict; rows kept, rerun to resume")
    return {i.id: store.rows[i.id]["cloud"] for i in items}


def with_verdicts(system: str, rows: dict[str, list[Prediction]], verdicts: dict[str, bool], field: str) -> dict[str, list[Prediction]]:
    """Rows where `field` holds one verdict per finding."""
    out: dict[str, list[Prediction]] = {}
    for v, ps in rows.items():
        out[v] = [p.model_copy(update={field: [verdicts[claim_id(system, v, p.scenario_id, i)] for i in range(len(p.card.findings))]})
                  if p.card is not None and p.card.findings else p for p in ps]
    return out


def sonnet_check(scenarios: list[Scenario], rows: dict[str, list[Prediction]], choice: dict[str, Any]) -> dict[str, Any]:
    """SELECTION_RULE on v1 and the chosen version with Sonnet's verdicts in place of Haiku's. Needs `cloud_supported` on both."""
    pair = {v: [p.model_copy(update={"claim_supported": p.cloud_supported}) for p in rows[v]] for v in (e6.BASELINE, choice["selected"])}
    metrics, _, _, again = e6.measure(scenarios, pair)
    claims = {v: [x for p in ps if p.claim_supported for x in zip(p.claim_supported, p.cloud_supported or [], strict=True)]
              for v, ps in ((v, rows[v]) for v in pair)}
    both = [x for xs in claims.values() for x in xs]
    return {"versions": list(pair), "selected_under_sonnet": again["selected"], "selected_under_haiku": choice["selected"],
            "same_version": again["selected"] == choice["selected"], "qualifies_under_sonnet": again["qualifies"],
            "metrics_under_sonnet": metrics, "claims_graded": len(both),
            "haiku_agrees_with_sonnet": (sum(a == b for a, b in both) / len(both)) if both else None,
            "note": "Sonnet graded every claim of these two versions only; the selection among all versions is Haiku's. "
                    "Sonnet's verdicts are in `cloud_supported` of those rows and do not replace `claim_supported`."}


def run_judge(*, root: Path = ROOT, from_commit: str, system: str = e6.DEFAULT_SYSTEM, rows_path: Path | None = None,
              haiku: Any, sonnet: Any | None = None, guard: UsageGuard, counter_path: Path = CLAUDE_CALLS_PATH,
              cap: int = CLAUDE_CALL_CAP, concurrency: int = pg.DEFAULT_CONCURRENCY, say: Callable[[str], None] = print,
              sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    if system not in e6.ALL_SYSTEMS:
        raise Refused(f"unknown system {system!r}; expected one of {e6.ALL_SYSTEMS}")
    if is_dirty(root):
        raise Refused("the working tree has uncommitted changes; commit first, or the record cannot be written")
    commit = resolve_commit(root, from_commit)
    changed = card_code_changed(root, commit)
    if changed:
        raise Refused(f"the prompts or card code changed between {commit[:12]} and HEAD, so the saved cards are not what this code "
                      f"would make: {changed[:8]}")
    head = head_commit(root)
    rows_path = rows_path or root / e6.PARTIAL_REL
    manifest_rel = SCENARIO_SOURCES[e2.SCENARIO_SOURCE][0]
    entry = json.loads((root / manifest_rel).read_text(encoding="utf-8"))["splits"][e6.SPLIT]
    digest = split_hash(e6.SPLIT, root, e2.SCENARIO_SOURCE)
    if digest != entry["hash"]:
        raise Refused("the dev scenario file does not match its manifest")
    scenarios = load_scenarios(e6.SPLIT, root / SCENARIO_SOURCES[e2.SCENARIO_SOURCE][1])
    ids = [s.id for s in scenarios]
    saved = all_saved_rows(rows_path, commit, system, set(ids))
    versions = [v for v in e6.declared_versions() if v.version in saved]
    if not versions or versions[0].version != e6.BASELINE:
        raise Refused(f"{rows_path} holds no saved cards of {e6.BASELINE} for commit {commit[:12]} on {system}")
    short = {v.version: [i for i in ids if i not in saved[v.version]] for v in versions}
    if any(short.values()):
        raise Refused("saved cards are missing for: " + ", ".join(f"{v} ({len(m)} cases)" for v, m in short.items() if m))
    by_version = {v.version: [saved[v.version][i] for i in ids] for v in versions}

    effective = e6.effective_digests(system, versions, scenarios[0], FakeLLM([]), object(), e2.load_retrieval_config(root / e2.CONFIG_REL))
    health: dict[str, dict[str, Any]] = {}
    failed: dict[str, str] = {}
    measured: dict[str, list[Prediction]] = {}
    for v in versions:
        health[v.version], why = e6._health(v.version, by_version[v.version])
        if why and v.version == e6.BASELINE:
            raise Refused(f"the baseline cannot be measured: {why}")
        if why:
            failed[v.version] = why
        else:
            measured[v.version] = by_version[v.version]
    not_run = [n for n in e6.VERSIONS if n not in saved]

    start_pct = guard.start()
    stem = f"{commit[:12]}.{system}"
    header = {"kind": "e6-judge-rows", "commit": commit, "system": system, "rubric": e2.RUBRIC_VERSION}
    haiku_items = items_of(system, measured)
    if not haiku_items:
        raise Refused("no card made a claim, so there is nothing to judge")
    verdicts = grade(haiku_items, haiku, root / JUDGE_DIR_REL / f"{stem}.haiku.rows.jsonl", {**header, "model": haiku.model}, guard,
                     counter_path=counter_path, cap=cap, concurrency=concurrency, say=say, sleep=sleep)
    judged = with_verdicts(system, measured, verdicts, "claim_supported")
    choice = e6.measure(scenarios, judged)[3]
    say(f"selected under {haiku.model}: {choice['selected']}")

    extra: dict[str, Any] = {}
    sonnet_calls = 0
    if sonnet is not None and choice["selected"] != e6.BASELINE:
        pair = {v: judged[v] for v in (e6.BASELINE, choice["selected"])}
        s_verdicts = grade(items_of(system, pair), sonnet, root / JUDGE_DIR_REL / f"{stem}.sonnet.rows.jsonl",
                           {**header, "model": sonnet.model}, guard, counter_path=counter_path, cap=cap, concurrency=concurrency,
                           say=say, sleep=sleep)
        judged = {**judged, **with_verdicts(system, pair, s_verdicts, "cloud_supported")}
        extra["sonnet_check"] = {"model": sonnet.model, **sonnet_check(scenarios, judged, choice)}
        sonnet_calls = len(s_verdicts)
        say(f"selected under {sonnet.model} for {list(pair)}: {extra['sonnet_check']['selected_under_sonnet']}")
    elif sonnet is not None:
        extra["sonnet_check"] = {"model": sonnet.model, "versions": [e6.BASELINE], "same_version": None,
                                 "note": "Haiku kept the baseline v1, so there is no chosen version to compare it with; Sonnet was not asked."}
    end_pct = guard.final()

    claims = len(haiku_items)
    names = e2.Models()
    models = {"writer": names.writer, "judge": haiku.model, "embedder": names.embedder}
    if sonnet is not None:
        models["cloud_grader"] = sonnet.model
    extra["judged_after_generation"] = {
        "generating_commit": commit, "head_commit": head, "system": system, "card_code_compared": list(CARD_CODE),
        "card_code_changed": [], "rows_file": rows_path.name,
        "note": "the cards were generated by an earlier `--generate-only` run at generating_commit and judged here; the writer and "
                "embedder names are those of e2_context at that commit (unchanged at HEAD); cold-start and GPU figures were "
                "not recorded in the saved rows",
    }
    extra["judge_calibration"] = {"entry": 56, "record": "JCAL-20261105T044839474906Z", "failed": True,
                                  "explanation_is": "a reading, not proof"}
    extra["usage"] = {"start_pct": start_pct, "end_pct": end_pct, "max_weekly_pct": MAX_WEEKLY_PCT}
    out = e6.write_e6_record(
        root=root, system=system, scenarios=scenarios, digest=digest, versions=versions, effective=effective, systems=judged,
        health=health, failed=failed, not_run=not_run,
        judge={"model": haiku.model, "rubric_version": e2.RUBRIC_VERSION, "same_model_and_prompt_as_writer": False,
               "calls": claims, "share_supported": sum(verdicts.values()) / claims, "via": "claude -p, one claim per call",
               "calibration": "failed the written test (entry 56); see notes"},
        judge_note=CALIBRATION_NOTE, models=models, retrieval_config=e2.load_retrieval_config(root / e2.CONFIG_REL), cold={}, gpu={},
        memory_build={"cases": len(scenarios), "failed": None, "mean_ms": None,
                      "note": "built in the generating run, not recorded in the saved rows"},
        started=started, claude_calls=claims + sonnet_calls, extra_config=extra, say=say)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-commit", required=True, help="the commit that generated the saved cards")
    ap.add_argument("--system", choices=e6.ALL_SYSTEMS, default=e6.DEFAULT_SYSTEM)
    ap.add_argument("--rows", type=Path, help="the saved rows file; default results/partial/e6-rows.jsonl under this root")
    ap.add_argument("--sonnet-check", action="store_true", help="Sonnet also grades the chosen version and v1")
    ap.add_argument("--concurrency", type=int, default=pg.DEFAULT_CONCURRENCY)
    args = ap.parse_args(argv)
    try:
        result = run_judge(from_commit=args.from_commit, system=args.system, rows_path=args.rows, haiku=pg.CountedClaude(),
                           sonnet=pg.CountedClaude(model=e2.CLOUD_MODEL) if args.sonnet_check else None,
                           guard=UsageGuard(MAX_WEEKLY_PCT), concurrency=args.concurrency)
    except UsageUnreadable as exc:
        print(f"refused: weekly usage cannot be read ({exc})", file=sys.stderr)
        return 1
    except (Refused, e2.RunBroken) as exc:
        print(f"E6 judging refused, nothing written: {exc}", file=sys.stderr)
        return 1
    except JudgeStopped as exc:
        print(f"E6 judging stopped: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"system": args.system, "selected": result["selected"], "record": result["record"].as_posix(),
                      "qualifies": result["choice"]["qualifies"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
