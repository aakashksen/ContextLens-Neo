"""E7's judging stage. Writing the cards holds the GPU; judging runs after it is released.

    python -m contextlens.experiments.e7_judge E7-<run id> --concurrency 8 --max-weekly-pct 98

Each claim (a finding of a card, with its cited quotes) is graded once by Haiku through `claude -p`, many at a time,
with the E2 rubric (`e2-v1`) and `experiments/parallel_grader.py`. No local model judges. The weekly usage guard stops
the run at 98% (the person's decision, 2026-11-06). Answers are saved as they arrive under results/partial/e7-judge/ (ids and verdicts, never text), so a
stopped run resumes. The result, ids and counts only, goes to results/e7-judge/<run id>.json.

Haiku failed the written calibration (docs/DECISIONS.md, entry 56); the explanation given there is a reading, not proof.
The output says so beside every number."""

import argparse
import json
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from contextlens.eval.records import ROOT
from contextlens.experiments import parallel_grader as pg
from contextlens.experiments.e2_context import RUBRIC_VERSION
from contextlens.experiments.judge_calibrate import Stopped, UsageGuard, UsageUnreadable
from contextlens.experiments.judge_check import Refused, preflight
from contextlens.llm.client import CLAUDE_CALL_CAP, CLAUDE_CALLS_PATH
from contextlens.replay.runner import Prediction

PARTIAL_REL = Path("results/partial/e7-judge")
OUT_REL = Path("results/e7-judge")
NOTE = ("Haiku failed the written calibration (docs/DECISIONS.md, entry 56) and is used as the dev judge anyway; the "
        "explanation given there is a reading, not proof. These rates are Haiku's, not ground truth.")


def load_items(run_id: str, root: Path) -> tuple[list[pg.Item], dict[str, str]]:
    """Every claim of every system of the run, and which system each belongs to."""
    path = root / "results/runs" / f"{run_id}.json"
    if not path.is_file():
        raise Refused(f"no run record {run_id}")
    rec = json.loads(path.read_text(encoding="utf-8"))
    items: list[pg.Item] = []
    system_of: dict[str, str] = {}
    for system, rel in rec["predictions"].items():
        for line in (root / rel).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            p = Prediction.model_validate_json(line)
            for i, f in enumerate(p.card.findings if p.card else []):
                cid = f"{run_id}/{system}/{p.scenario_id}/{i}"
                items.append(pg.Item(cid, f.statement, tuple((e.speaker, e.quote) for e in f.evidence)))
                system_of[cid] = system
    return items, system_of


def grade_claims(items: list[pg.Item], *, llm, partial_path: Path, header: dict, guard: UsageGuard | None,
                 use_guard: bool = True, concurrency: int = pg.DEFAULT_CONCURRENCY,
                 counter_path: Path = CLAUDE_CALLS_PATH, cap: int = CLAUDE_CALL_CAP, say: Callable[[str], None] = print,
                 sleep: Callable[[float], None] = time.sleep) -> tuple[dict[str, bool], list[str], dict]:
    """One call per claim without a saved verdict, many at a time, under the weekly usage guard. Returns the verdicts,
    the ids left ungraded and a small account (calls made, weekly usage at start and end). Raises Stopped when a limit
    ended the run early; the saved rows stay and the same call resumes."""
    if use_guard and guard is None:
        raise Refused("the weekly usage guard needs a usage reader; pass none only with --no-usage-guard")
    start_pct = guard.start() if guard is not None and use_guard else None
    store = pg.RowStore(partial_path, header)
    todo = pg.pending(items, store)
    say(f"{len(items)} claims, {len(items) - len(todo)} already graded; {len(todo)} calls to {llm.model}, concurrency "
        f"{concurrency}. Weekly usage {'not guarded' if start_pct is None else f'{start_pct:g}%'}.")
    preflight(len(todo), counter_path, cap)
    calls_before = llm.calls_made
    stop = (lambda: guard.check()) if guard is not None and use_guard else (lambda: None)
    out = pg.grade_single(items, llm, store, concurrency=concurrency, should_stop=stop, sleep=sleep)
    end_pct = guard.final() if guard is not None and use_guard else None
    if out.stop_reason:
        raise Stopped(f"{out.stop_reason}; {len(store.rows)} rows kept, rerun the same command to resume", start_pct, end_pct)
    verdicts = {i.id: store.rows[i.id]["cloud"] for i in items if store.rows.get(i.id, {}).get("status") == "ok"}
    return verdicts, pg.ungraded(items, store), {"calls_made": llm.calls_made - calls_before,
                                                 "weekly_usage_pct": {"start": start_pct, "end": end_pct}}


def judge(run_id: str, *, llm, root: Path = ROOT, guard: UsageGuard | None, use_guard: bool = True,
          concurrency: int = pg.DEFAULT_CONCURRENCY, counter_path: Path = CLAUDE_CALLS_PATH,
          cap: int = CLAUDE_CALL_CAP, say: Callable[[str], None] = print,
          sleep: Callable[[float], None] = time.sleep) -> Path:
    if use_guard and guard is None:
        raise Refused("the weekly usage guard needs a usage reader; pass none only with --no-usage-guard")
    items, system_of = load_items(run_id, root)
    header = {"kind": "e7-judge-rows", "run_id": run_id, "model": llm.model, "rubric": RUBRIC_VERSION}
    verdicts, missing, account = grade_claims(
        items, llm=llm, partial_path=root / PARTIAL_REL / f"{run_id}.rows.jsonl", header=header, guard=guard,
        use_guard=use_guard, concurrency=concurrency, counter_path=counter_path, cap=cap, say=say, sleep=sleep)
    per_system: dict[str, dict] = {}
    for system in dict.fromkeys(system_of.values()):
        mine = [i for i in system_of if system_of[i] == system]
        graded = [verdicts[i] for i in mine if i in verdicts]
        per_system[system] = {"claims": len(mine), "graded": len(graded), "supported": sum(graded),
                              "grounded_rate": sum(graded) / len(graded) if graded else None}
    doc = {"kind": "e7-judge", "run_id": run_id, "model": llm.model, "rubric": RUBRIC_VERSION,
           "note": NOTE, "complete": not missing, "per_system": per_system, "ungraded": missing, "verdicts": verdicts,
           **account,
           "judged_at": datetime.now(timezone.utc).isoformat()}
    path = root / OUT_REL / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Judge the claims of an E7 run with Haiku, one claim per call")
    ap.add_argument("run_id")
    ap.add_argument("--concurrency", type=int, default=pg.DEFAULT_CONCURRENCY)
    ap.add_argument("--max-weekly-pct", type=float, default=98.0)
    ap.add_argument("--no-usage-guard", action="store_true", help="run without reading weekly usage")
    a = ap.parse_args(argv)
    guard = None if a.no_usage_guard else UsageGuard(a.max_weekly_pct)
    try:
        path = judge(a.run_id, llm=pg.CountedClaude(), guard=guard, use_guard=not a.no_usage_guard, concurrency=a.concurrency)
    except UsageUnreadable as exc:
        print(f"refused: weekly usage cannot be read ({exc}); pass --no-usage-guard to run without the guard", file=sys.stderr)
        return 1
    except Refused as exc:
        print("refused:", exc, file=sys.stderr)
        return 1
    except Stopped as exc:
        print(f"stopped: {exc}. Weekly usage at start {exc.usage_start_pct}%, at end {exc.usage_end_pct}%.", file=sys.stderr)
        return 2
    print(path.as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())
