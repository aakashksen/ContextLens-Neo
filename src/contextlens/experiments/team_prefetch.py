"""T9.1's precondition (the coordinator, 2026-11-06): `team_shipping_changed` with retrieval before the memory model call
and the specialists' sub-deadline (its own `FIXES`; both fixes were then dropped and production ships neither), once, on the new dev cases (`scenarios_extra`, 108), under entry 51's rules, then
compared with the earlier arm's rows by one judge.

Two stages. The first holds the GPU and judges nothing; the second runs after the GPU is released.

  python -m contextlens.experiments.team_prefetch write --run-conditions "<what else was running>"
  python -m contextlens.experiments.team_prefetch judge E3TP-<run id> [--also-sonnet] [--max-weekly-pct 98]

`write` builds each case's memory with the writer, then presses the shipping team (a `PromptedChecker` on the writer,
`memory_gap_abstains` on) with the memory specialist prefetched, one case at a time, saving every finished row as it is made
so a stopped run resumes at the same commit. It refuses a dirty tree at the start. It writes an E3TP record with no
judging in it. `judge` grades every claim of both arms (the earlier record's saved rows and the new rows) with Haiku through
`claude -p`, one claim per call, with rubric e2-v1 and the weekly usage guard, so both arms have the same judge; with
`--also-sonnet` Sonnet grades them too and is a second figure. Local judging is not used. Haiku failed the written
calibration (docs/DECISIONS.md, entry 56); the explanation there is a reading, not proof.

The written score is entry 51's: mean(grounded_rate, min(answer_when_answerable, abstain_when_insufficient)). The output
also counts the cards the 8 s deadline cut in each arm."""

import argparse
import json
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contextlens.api import production
from contextlens.eval.records import ROOT, head_commit, is_dirty, label_text, load_verified, write_run
from contextlens.eval.row_report import row_report
from contextlens.experiments import e2_context as e2
from contextlens.experiments import e3_team as e3
from contextlens.experiments import team_attempt as ta
from contextlens.experiments.e2_context import DEADLINE_S, PROMPT_VERSION, STREAK_LOCAL, Models, RunBroken
from contextlens.experiments.e7_judge import NOTE as HAIKU_NOTE
from contextlens.experiments.e7_judge import grade_claims
from contextlens.experiments.e7_realtime import deadline_report
from contextlens.experiments.judge_calibrate import UsageGuard
from contextlens.experiments.judge_check import Refused
from contextlens.experiments.parallel_grader import DEFAULT_CONCURRENCY, Item
from contextlens.llm.client import CLAUDE_CALL_CAP, CLAUDE_CALLS_PATH, LLMClient, LLMError
from contextlens.llm.ollama_ctl import OllamaCtl, full_name
from contextlens.memory.extract import MemoryStore
from contextlens.replay.runner import Prediction

EXPERIMENT = "E3TP"
SOURCE = "scenarios_extra"
ARM = "team_shipping_changed_prefetch_subdeadline"
EARLIER_RECORD, EARLIER_ARM = "E3TS-20261104T173526710888Z", ta.SHIPPING_CHANGED
PARTIAL_REL = Path("results/partial/e3tp-rows.jsonl")  # git-ignored; one line per finished row, keyed by commit
OUT_REL = Path("results/e3tp-judge")
SONNET = e2.CLOUD_MODEL
# The two team fixes this check measured, which were then dropped (docs/DECISIONS.md, entries 64 and 70): production's
# TEAM_SETTINGS is now off, so the check names them itself and stays a record of what was measured.
FIXES = {"prefetch_memory": True, "specialist_deadline_s": production.SPECIALIST_DEADLINE_S}
CHANGE = ("both team fixes (since dropped, see docs/DECISIONS.md entry 70), with the values from api/production.py: retrieval before the memory model call (the "
          "memory specialist's first search is run on the question and shown to the model as its own first tool step, so it "
          "needs one model call less) and a sub-deadline for the memory and consistency specialists' first pass, so assistance "
          "always has the rest of the deadline. Everything else is `team_shipping_changed`.")


def make_assistant(sc, writer: LLMClient, embedder: Any, retrieval_config: dict[str, Any], memory: MemoryStore):
    team = ta.make_assistant(ta.SHIPPING_CHANGED, sc, writer, embedder, retrieval_config, memory)
    team.deadline_s = production.DEADLINE_S
    for name, value in FIXES.items():
        setattr(team, name, value)
    return team


def run_prefetch(*, root: Path = ROOT, writer: LLMClient, embedder: Any, ctl: OllamaCtl, names: Models | None = None,
                 retrieval_config: dict[str, Any], run_conditions: str, say: Callable[[str], None] = print) -> Path:
    names = names or Models()
    try:
        if not run_conditions.strip():
            raise RunBroken("run_conditions is required: say what else was running while latency was measured")
        if is_dirty(root):
            raise RunBroken("the working tree has uncommitted changes; commit first, or the record cannot be written")
        return _run(root, writer, embedder, ctl, names, retrieval_config, run_conditions, say)
    except LLMError as exc:
        raise RunBroken(f"a model server call failed: {exc}") from exc
    finally:
        e2._unload_models(ctl, [(names.writer, False), (names.embedder, True)], say)


def _run(root: Path, writer: LLMClient, embedder: Any, ctl: OllamaCtl, names: Models, retrieval_config: dict[str, Any],
         run_conditions: str, say: Callable[[str], None]) -> Path:
    started = datetime.now(timezone.utc).isoformat()
    if retrieval_config.get("rerank") or retrieval_config.get("chunk_turns", 1) != 1:
        raise RunBroken("the retrieval config needs a one-turn chunk and no reranker")
    commit = head_commit(root)
    try:
        scenarios, digest = load_verified(SOURCE, ta.SPLIT, root)
    except RuntimeError as exc:
        raise RunBroken(str(exc)) from exc
    missing = [m for m in (names.writer, names.embedder) if full_name(m) not in ctl.installed()]
    if missing:
        raise RunBroken(f"models not installed in Ollama: {missing}")
    cold = {"writer": ctl.load(names.writer), "embedder": ctl.load(names.embedder, embedding=True)}
    devices = {"writer": e3.ollama_device(ctl, names.writer), "embedder": e3.ollama_device(ctl, names.embedder)}
    have = e2._saved_rows(root / PARTIAL_REL, commit, SOURCE, ARM, {s.id for s in scenarios}, names.writer)[0]
    if have:
        say(f"reusing {len(have)} saved rows")
    todo = [s for s in scenarios if s.id not in have]
    say(f"building memory for {len(todo)} cases")
    memories: dict[str, MemoryStore] = {}
    failed: list[str] = []
    reasons: list[dict[str, str]] = []
    for sc in todo:
        memories[sc.id], cost = e2.build_memory(sc, writer, lambda cid, why: reasons.append({"id": cid, "reason": why}))
        if cost is None:
            failed.append(sc.id)
    if len(failed) / len(scenarios) > e2.BROKEN_MAX:
        raise RunBroken(f"memory extraction failed on {len(failed)} of {len(scenarios)} cases")
    say(f"running {ARM}")

    def save(row: Prediction) -> None:
        entry = {"commit": commit, "subset": SOURCE, "winner": ARM, "model": names.writer, "calls": 0,
                 "row": json.loads(row.model_dump_json())}
        e2._append_durable(root / PARTIAL_REL, json.dumps(entry))

    rows = e2.run_system(ARM, scenarios, lambda sc: make_assistant(sc, writer, embedder, retrieval_config,
                                                                  memories.get(sc.id, MemoryStore())),
                         STREAK_LOCAL, say=say, have=have, on_row=save)
    health = e2.check_health(ARM, rows)
    metrics, intervals, comparisons = e3.card_stats(scenarios, {ARM: rows})
    report = row_report(rows, DEADLINE_S)
    config: dict[str, Any] = {
        "experiment": EXPERIMENT, "part": "cards", "arm": ARM, "judged": False, "source": SOURCE,
        "based_on_rows": EARLIER_RECORD, "change": CHANGE, "deadline_s": DEADLINE_S,
        "precondition": "T9.1 (the coordinator, 2026-11-06): the fixed arm runs once on the new dev cases under entry 51's rules, "
                        "before T9.1; the comparison with the earlier rows is made by one judge for both, in the judge step",
        "no_further_team_variants": True, "run_conditions": run_conditions, "devices": devices,
        "consistency_checker": {"class": "PromptedChecker", "model": names.writer}, "score": e3.SELECTION_RULE,
        "retrieval": {k: retrieval_config.get(k) for k in ("winner", "mode", "k", "rerank", "chunk_turns", "run_ids")},
        "row_report": {ARM: report}, "health": {ARM: health}, "cold_start_s": cold,
        "deadline": deadline_report({ARM: rows}),
        "memory_build": {"cases": len(scenarios), "failed": len(failed), "failed_ids": failed, "failed_reasons": reasons},
        "rows_reused_from_partial": len(have),
        "notes": [e3.conditions_note(run_conditions), label_text(root, SOURCE)], "claude_calls": 0, "started_at": started,
    }
    attempts = [{"setting": ARM, "kind": "card system", "ok": True, "error": None, "health": health}]
    path = write_run(EXPERIMENT, ta.SPLIT, config, {"writer": names.writer, "embedder": names.embedder},
                     {"recall": PROMPT_VERSION}, metrics, intervals, attempts, {ARM: rows}, digest,
                     comparisons=comparisons, scenario_source=SOURCE, root=root)
    say(f"wrote {path.name}")
    return path


# -- judging --------------------------------------------------------------------------------------------------------

def written_score(grounded_rate: float | None, metrics: dict[str, float | None]) -> float | None:
    return e3.score({**metrics, "grounded_rate": grounded_rate})


def _rows(root: Path, record_id: str, system: str) -> tuple[list[Prediction], dict[str, float | None]]:
    rec = json.loads((root / "results/runs" / f"{record_id}.json").read_text(encoding="utf-8"))
    lines = (root / rec["predictions"][system]).read_text(encoding="utf-8").splitlines()
    return [Prediction.model_validate_json(x) for x in lines if x.strip()], rec["metrics"][system]


def _items(arm: str, rows: list[Prediction]) -> list[Item]:
    return [Item(f"{arm}/{p.scenario_id}/{i}", f.statement, tuple((e.speaker, e.quote) for e in f.evidence))
            for p in rows for i, f in enumerate(p.card.findings if p.card else [])]


def judge(new_id: str, *, judges: list, root: Path = ROOT, guard: UsageGuard | None, use_guard: bool = True,
          concurrency: int = DEFAULT_CONCURRENCY, counter_path: Path = CLAUDE_CALLS_PATH, cap: int = CLAUDE_CALL_CAP,
          say: Callable[[str], None] = print, sleep: Callable[[float], None] | None = None) -> Path:
    """Every claim of both arms graded by each judge in `judges`; one file with each judge's rates and written scores."""
    import time

    arms = {"earlier": (EARLIER_RECORD, EARLIER_ARM), "fixed": (new_id, ARM)}
    loaded = {a: _rows(root, rid, system) for a, (rid, system) in arms.items()}
    items = [i for a, (rows, _) in loaded.items() for i in _items(a, rows)]
    out_judges: dict[str, Any] = {}
    for llm in judges:
        header = {"kind": "e3tp-judge-rows", "earlier": EARLIER_RECORD, "fixed": new_id, "model": llm.model,
                  "rubric": e2.RUBRIC_VERSION}
        safe = "".join(c if c.isalnum() else "-" for c in llm.model)
        verdicts, missing, account = grade_claims(
            items, llm=llm, partial_path=root / "results/partial/e3tp-judge" / f"{new_id}.{safe}.rows.jsonl", header=header,
            guard=guard, use_guard=use_guard, concurrency=concurrency, counter_path=counter_path, cap=cap, say=say,
            sleep=sleep or time.sleep)
        per = {}
        for a, (rows, metrics) in loaded.items():
            mine = [i.id for i in _items(a, rows)]
            graded = [verdicts[i] for i in mine if i in verdicts]
            rate = sum(graded) / len(graded) if graded else None
            per[a] = {"claims": len(mine), "graded": len(graded), "grounded_rate": rate,
                      "answer_when_answerable": metrics.get("answer_when_answerable"),
                      "abstain_when_insufficient": metrics.get("abstain_when_insufficient"),
                      "written_score": written_score(rate, metrics)}
        out_judges[llm.model] = {"arms": per, "complete": not missing, "ungraded": missing, **account}
    doc = {"kind": "e3tp-judge", "earlier_record": EARLIER_RECORD, "fixed_record": new_id, "rubric": e2.RUBRIC_VERSION,
           "note": HAIKU_NOTE + " One judge grades both arms; earlier local-judge figures (0.792 local, 0.712 Sonnet) are "
                   "not comparable with these.",
           "deadline": deadline_report({a: rows for a, (rows, _) in loaded.items()}),
           "latency_ms": {a: {k: m.get(k) for k in ("latency_p50_ms", "latency_p95_ms")} for a, (_, m) in loaded.items()},
           "judges": out_judges, "judged_at": datetime.now(timezone.utc).isoformat()}
    path = root / OUT_REL / f"{new_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def main(argv: list[str] | None = None) -> int:
    from contextlens.experiments.judge_calibrate import Stopped, UsageUnreadable

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--run-conditions", required=True)
    j = sub.add_parser("judge")
    j.add_argument("record")
    j.add_argument("--also-sonnet", action="store_true", help="Sonnet grades the same claims too, as a second figure")
    j.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    j.add_argument("--max-weekly-pct", type=float, default=98.0)
    a = ap.parse_args(argv)
    if a.cmd == "write":
        from contextlens.llm.client import Embedder, OllamaClient

        try:
            path = run_prefetch(writer=OllamaClient(e2.WRITER_MODEL), embedder=Embedder(e2.EMBED_MODEL), ctl=OllamaCtl(),
                                retrieval_config=e2.load_retrieval_config(ROOT / e2.CONFIG_REL), run_conditions=a.run_conditions)
        except (RunBroken, OSError, ValueError, KeyError) as exc:
            print(f"{EXPERIMENT} stopped, nothing more written: {exc}", file=sys.stderr)
            return 2
        print(path.as_posix())
        return 0
    from contextlens.experiments.parallel_grader import HAIKU_MODEL, CountedClaude

    judges = [CountedClaude(HAIKU_MODEL), *([CountedClaude(SONNET)] if a.also_sonnet else [])]
    try:
        path = judge(a.record, judges=judges, guard=UsageGuard(a.max_weekly_pct), concurrency=a.concurrency)
    except UsageUnreadable as exc:
        print(f"refused: weekly usage cannot be read ({exc})", file=sys.stderr)
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
