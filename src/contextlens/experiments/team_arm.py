"""One run of the single arm `team_shipping_changed` (docs/DECISIONS.md, entry 51). Dev only.

The arm is the shipping team (a `PromptedChecker` on the writer) with `memory_gap_abstains` on, both pieces already written.
Only this arm runs. The other systems come from the saved rows of the two E3T records in PRIOR, loaded from disk, and are
judged by the same local judge, so the record holds one system (its metrics recompute from its rows) and the tables
beside the other arms sit in `config.cross_record` with the source record ids: a record's comparisons must name two
systems of that same record. Nothing here picks a winner under entry 51; that needs both judges.

Run (uses Ollama; the person starts it, on a committed tree):
  .venv/Scripts/python -m contextlens.experiments.team_attempt --arm team_shipping_changed --run-conditions "<what else was running>"
"""

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contextlens.eval.records import ROOT, RUNS_REL, head_commit, is_dirty, label_text, load_verified, write_run
from contextlens.eval.row_report import row_report
from contextlens.experiments import e2_context as e2
from contextlens.experiments import e3_team as e3
from contextlens.experiments import team_attempt as ta
from contextlens.experiments.e2_context import DEADLINE_S, PROMPT_VERSION, STREAK_LOCAL, Judge, JudgeFailed, Models, RunBroken
from contextlens.llm.client import LLMClient, LLMError
from contextlens.llm.ollama_ctl import OllamaCtl, full_name
from contextlens.memory.extract import MemoryStore
from contextlens.replay.runner import Prediction

EXPERIMENT = "E3TS"  # its own name, so `check_run E3T` does not see these records
ARM = ta.SHIPPING_CHANGED
DECISION_ENTRY = 51
PRIOR = {"scenarios": "E3T-20261104T131120203662Z", "scenarios_extra": "E3T-20261104T131417960789Z"}
ALL_SYSTEMS = (*ta.RUN_SYSTEMS, ARM)
PARTIAL_REL = Path("results/partial/e3ts-rows.jsonl")  # git-ignored; one line per finished row, keyed by commit and system
COMPARED = (ARM, "team_changed")  # entry 51 looks at the first; the second is reported beside it
WRITTEN = ("Written by entry 51 before any result. A one-time exception to entry 48's one-attempt rule, made so that what could "
           "ship gets measured; no further team variants follow, whatever the result. No new logic, prompt or threshold: the "
           "arm is the two pieces entries 48 and 49 already wrote.")
NEW_LABELS = ("The labels of the new set (scenarios_extra) are unreviewed by a person. An agent's read of at least 30 of them "
              "is a separate step under entry 51 and does not make them reviewed.")
RULE_LOOKS_AT = ("Entry 51's rule for T9.1's primary looks at `team_shipping_changed` (not `team_changed`, which cannot ship as "
                 "measured) on the new set (scenarios_extra), which must beat `single` on the written score under both the "
                 "local and the cloud judge. This record has the local judge only; the cloud grader's half comes from the "
                 "judge check, so this record decides nothing.")
CROSS_NOTE = ("The other systems' rows are the saved rows of the E3T records named in `source_records`, loaded from disk. A "
              "record's comparisons must name two systems of that record, so every comparison, score and twin table that sets "
              "this arm beside the others is here, recomputed from those rows and this record's rows. They were run in an "
              "earlier session: timing and load differ, and the judge model is the same one.")


def _load_prior(root: Path, source: str, run_id: str, digest: str, judge_model: str) -> dict[str, list[Prediction]]:
    path = root / RUNS_REL / f"{run_id}.json"
    if not path.is_file():
        raise RunBroken(f"the saved record {run_id} is not under {RUNS_REL.as_posix()}")
    rec = json.loads(path.read_text(encoding="utf-8"))
    what = (rec.get("scenario_source"), rec.get("split"), rec.get("dataset_hash"))
    if what != (source, ta.SPLIT, digest):
        raise RunBroken(f"{run_id} is not the dev record of {source} on the files on disk")
    if ((rec.get("config") or {}).get("judge") or {}).get("model") != judge_model:
        raise RunBroken(f"{run_id} was judged by a different model than {judge_model}, so scores would not be comparable")
    if not set(ta.RUN_SYSTEMS) <= set(rec["predictions"]):
        raise RunBroken(f"{run_id} lacks some of the systems {list(ta.RUN_SYSTEMS)}")
    return {n: rows for n, rows in e3.load_rows(root, rec).items() if n in ta.RUN_SYSTEMS}


def entry51_local(sets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Whether each compared arm has a strictly higher written score than `single` on the new set, under the local judge.
    A fact about this judge only: it names no winner."""
    base = sets["new"]["scores"].get("single")
    out: dict[str, Any] = {}
    for name in COMPARED:
        score = sets["new"]["scores"].get(name)
        out[name] = {"score_new": score, "single_score_new": base,
                     "beats_single_on_new_by_written_score": None if score is None or base is None else score > base,
                     "scores_old_both": {k: sets[k]["scores"].get(name) for k in ("old", "both")}}
    return {"judge": "local", "set": "new (scenarios_extra)", "rule": "the written score is strictly higher than single's",
            "arms": out, "winner": "not chosen here: entry 51 needs both judges"}


def memory_empty_by_gold(scenarios: list[Any], rows: list[Prediction]) -> dict[str, int]:
    """Cards the change withheld, by what the case really is. Withheld answerable cases are the cost of the change."""
    gold = {s.id: s.gold_label == "insufficient" for s in scenarios}
    fired = {p.scenario_id for p in rows if any(d.reason == "memory_empty" for d in p.drops)}
    return {"insufficient_cases": sum(gold[i] for i in fired), "answerable_cases": sum(not gold[i] for i in fired)}


def run_arm(*, root: Path = ROOT, writer: LLMClient, judge_llm: LLMClient, embedder: Any, ctl: OllamaCtl,
            names: Models | None = None, retrieval_config: dict[str, Any], run_conditions: str,
            prior: dict[str, str] | None = None, say: Callable[[str], None] = print) -> dict[str, Path]:
    """Run the arm on both dev sources, judge it, and write one record per source. Models are unloaded at the end."""
    names = names or Models()
    mine = [(names.writer, False), (names.judge, False), (names.embedder, True)]
    try:
        if not run_conditions.strip():
            raise RunBroken("run_conditions is required: say what else was running while latency was measured")
        return _arm(root, writer, judge_llm, embedder, ctl, names, retrieval_config, run_conditions, prior or PRIOR, say)
    except LLMError as exc:
        raise RunBroken(f"a model server call failed: {exc}") from exc
    finally:
        e2._unload_models(ctl, mine, say)


def _arm(root: Path, writer: LLMClient, judge_llm: LLMClient, embedder: Any, ctl: OllamaCtl, names: Models,
         retrieval_config: dict[str, Any], run_conditions: str, prior: dict[str, str], say: Callable[[str], None]) -> dict[str, Path]:
    started = datetime.now(timezone.utc).isoformat()
    if is_dirty(root):
        raise RunBroken("the working tree has uncommitted changes; commit first, or the record cannot be written")
    if retrieval_config.get("rerank") or retrieval_config.get("chunk_turns", 1) != 1:
        raise RunBroken("the retrieval config needs a one-turn chunk and no reranker")
    commit = head_commit(root)
    loaded = {}
    for source in ta.SOURCES:
        try:
            loaded[source] = load_verified(source, ta.SPLIT, root)
        except RuntimeError as exc:
            raise RunBroken(str(exc)) from exc
    others = {s: _load_prior(root, s, prior[s], loaded[s][1], names.judge) for s in ta.SOURCES}
    for s in ta.SOURCES:
        ids = [p.scenario_id for p in others[s]["single"]]
        if ids != [sc.id for sc in loaded[s][0]]:
            raise RunBroken(f"the saved rows of {prior[s]} do not cover the {s} dev cases in order")
    missing = [m for m in (names.writer, names.judge, names.embedder) if full_name(m) not in ctl.installed()]
    if missing:
        raise RunBroken(f"models not installed in Ollama: {missing}")

    cold: dict[str, float] = {"judge_check": ctl.load(names.judge)}
    Judge(judge_llm).canary()
    ctl.unload(names.judge)
    cold["writer"] = ctl.load(names.writer)
    cold["embedder"] = ctl.load(names.embedder, embedding=True)
    devices = {"writer": e3.ollama_device(ctl, names.writer), "embedder": e3.ollama_device(ctl, names.embedder)}

    rows: dict[str, list[Prediction]] = {}
    health: dict[str, dict[str, Any]] = {}
    memory_failed: dict[str, list[str]] = {}
    memory_reasons: dict[str, list[dict[str, str]]] = {}
    reused: dict[str, int] = {}
    for source in ta.SOURCES:
        scenarios = loaded[source][0]
        have = e2._saved_rows(root / PARTIAL_REL, commit, source, ARM, {s.id for s in scenarios}, names.writer)[0]
        reused[source] = len(have)
        if have:
            say(f"{source}: reusing {len(have)} saved rows")
        todo = [s for s in scenarios if s.id not in have]
        say(f"{source}: building memory for {len(todo)} cases")
        memories: dict[str, MemoryStore] = {}
        memory_failed[source], memory_reasons[source] = [], []
        for sc in todo:
            memories[sc.id], cost = e2.build_memory(
                sc, writer, lambda cid, why, source=source: memory_reasons[source].append({"id": cid, "reason": why}))
            if cost is None:
                memory_failed[source].append(sc.id)
        if len(memory_failed[source]) / len(scenarios) > e2.BROKEN_MAX:
            raise RunBroken(f"memory extraction failed on {len(memory_failed[source])} of {len(scenarios)} {source} cases")
        say(f"{source}: running {ARM}")

        def save(row: Prediction, source: str = source) -> None:
            entry = {"commit": commit, "subset": source, "winner": ARM, "model": names.writer, "calls": 0,
                     "row": json.loads(row.model_dump_json())}
            e2._append_durable(root / PARTIAL_REL, json.dumps(entry))

        rows[source] = e2.run_system(
            ARM, scenarios,
            lambda sc: ta.make_assistant(ARM, sc, writer, embedder, retrieval_config, memories.get(sc.id, MemoryStore())),
            STREAK_LOCAL, say=say, have=have, on_row=save)
        health[source] = e2.check_health(ARM, rows[source])
    ctl.unload(names.writer)

    say("judging")
    cold["judge"] = ctl.load(names.judge)
    devices["judge"] = e3.ollama_device(ctl, names.judge)
    judge = Judge(judge_llm)
    for source in ta.SOURCES:
        rows[source] = e2.judge_rows(judge, rows[source])
    if judge.calls == 0:
        raise JudgeFailed("no finding was judged, so no card made a claim")
    judged = {"model": names.judge, "rubric_version": e2.RUBRIC_VERSION, "same_model_and_prompt_as_writer": False,
              "calls": judge.calls, "share_supported": judge.yes / judge.calls}

    labels = {s: label_text(root, s) for s in ta.SOURCES}
    by_source = {s: (loaded[s][0], {**others[s], ARM: rows[s]}) for s in ta.SOURCES}
    full = ta.all_reports(by_source, labels, ALL_SYSTEMS, ())
    cross = {"source_records": prior, "systems": list(ALL_SYSTEMS), "note": CROSS_NOTE,
             "scores": {k: {n: v["scores"].get(n) for n in ALL_SYSTEMS} for k, v in {**full["sets"], **full["slices"]}.items()},
             "sets": full["sets"], "slices": full["slices"], "twins": full["twins"], "entry_51_local": entry51_local(full["sets"])}
    models = {"writer": names.writer, "judge": names.judge, "embedder": names.embedder}
    paths: dict[str, Path] = {}
    for source in ta.SOURCES:
        scenarios, digest = loaded[source]
        metrics, intervals, comparisons = e3.card_stats(scenarios, {ARM: rows[source]})
        report = {ARM: row_report(rows[source], DEADLINE_S)}
        config: dict[str, Any] = {
            "experiment": EXPERIMENT, "part": "cards", "decision_entry": DECISION_ENTRY, "arm": ARM,
            "written_before_results": WRITTEN, "new_set_labels": NEW_LABELS, "entry_51_rule_looks_at": RULE_LOOKS_AT,
            "no_further_team_variants": True, "source": source, "sources_run": list(ta.SOURCES),
            "run_conditions": run_conditions,
            "devices": {**devices, "note": "from the Ollama process list right after each model loaded; unknown means it did not say"},
            "systems": [ARM], "compared_with": list(ta.RUN_SYSTEMS), "deadline_s": DEADLINE_S, "judge": judged,
            "same_for_all": "the writer, retrieval config, cases, deadline and local judge model of the E3T records named in "
                            "cross_record.source_records; the arm adds a PromptedChecker on the writer and memory_gap_abstains",
            "consistency_checker": {"class": "PromptedChecker", "model": names.writer},
            "change": ta.CHANGE, "hypothesis": ta.HYPOTHESIS, "score": e3.SELECTION_RULE,
            "cross_record": cross,
            "retrieval": {k: retrieval_config.get(k) for k in ("winner", "mode", "k", "rerank", "chunk_turns", "run_ids")},
            "row_report": report, "health": {ARM: health[source]}, "cold_start_s": cold,
            "drops": {ARM: e3.drop_report(scenarios, rows[source])},
            "memory_empty_by_case": memory_empty_by_gold(scenarios, rows[source]),
            "memory_build": {"cases": len(scenarios), "failed": len(memory_failed[source]), "failed_ids": memory_failed[source],
                             "failed_reasons": memory_reasons[source]},
            "rows_reused_from_partial": reused[source],
            "prompt_note": f"prompts are unchanged, label {PROMPT_VERSION}",
            "notes": [e3.conditions_note(run_conditions), e3.deadline_sentence(report), ta.ANSWER_RATE_NOTE,
                      "grounded_rate comes from one local judge; no person has reviewed a sample.",
                      NEW_LABELS, RULE_LOOKS_AT],
            "claude_calls": 0, "started_at": started,
        }
        attempts = [{"setting": ARM, "kind": "card system", "ok": True, "error": None, "health": health[source]}]
        paths[source] = write_run(EXPERIMENT, ta.SPLIT, config, models, {"recall": PROMPT_VERSION}, metrics, intervals, attempts,
                                  {ARM: rows[source]}, digest, comparisons=comparisons, scenario_source=source, root=root)
        say(f"wrote {paths[source].name}")
    return paths
