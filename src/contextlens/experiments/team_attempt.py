"""One written attempt to fix how the team merges its specialists (docs/DECISIONS.md, entry 48). Dev only.

Three systems on the same cases and settings: `single`, `team`, and `team_changed` (the team with `memory_gap_abstains`:
a clean empty memory pass withholds the other agent's findings). A fourth arm, `team_shipping` (docs/DECISIONS.md, entry 49),
is the team built as production builds it, with a consistency checker; it is reported apart and cannot change entry 48's
selection. Each of the two dev sources, the 148 old cases
(`scenarios`) and the 108 new twins (`scenarios_extra`), gets its own record, because a record holds one source. Each
record also carries the old set, the new set and both together, and the new set split into the 24 pairs a plain
retrieval score could separate and the other 30. Nothing is run here without the person starting it.

Run (uses Ollama; the person starts it, on a committed tree):
  .venv/Scripts/python -m contextlens.experiments.team_attempt --run-conditions "<what else was running>"
  .venv/Scripts/python -m contextlens.experiments.team_attempt --arm team_shipping_changed --run-conditions "<...>"   (entry 51)
The finished run keeps results/partial/e3t-rows.jsonl: its rows are keyed by commit, subset, system and model, so only a rerun
at the same commit reuses them.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contextlens.agents.consistency_check import PromptedChecker
from contextlens.data.extra_cases import SHAPES, _shape_of
from contextlens.eval import series
from contextlens.eval.records import ROOT, head_commit, is_dirty, label_text, load_verified, write_run
from contextlens.eval.row_report import row_report
from contextlens.eval.stats import mcnemar_exact, paired_bootstrap_diff
from contextlens.experiments import e2_context as e2
from contextlens.experiments import e3_team as e3
from contextlens.experiments.e2_context import DEADLINE_S, PROMPT_VERSION, STREAK_LOCAL, Judge, JudgeFailed, Models, RunBroken
from contextlens.llm.client import LLMClient, LLMError
from contextlens.llm.ollama_ctl import OllamaCtl, full_name
from contextlens.memory.extract import MemoryStore
from contextlens.replay.runner import Assistant, Prediction
from contextlens.schemas import Scenario

EXPERIMENT = "E3T"  # its own name, so E3's records and checks are not mixed with these
SPLIT = "dev"
SOURCES = ("scenarios", "scenarios_extra")  # the old 148 dev cases, then the new 108
SYSTEMS = ("single", "team", "team_changed")  # entry 48's attempt; unchanged by the shipping arm
SHIPPING = "team_shipping"
RUN_SYSTEMS = (*SYSTEMS, SHIPPING)
SHIPPING_CHANGED = "team_shipping_changed"  # entry 51; run alone by experiments/team_arm.py, never part of RUN_SYSTEMS
PAIRS = (("team", "single"), ("team_changed", "single"), ("team_changed", "team"), (SHIPPING, "single"), (SHIPPING, "team"),
         (SHIPPING_CHANGED, "single"), (SHIPPING_CHANGED, "team"), (SHIPPING_CHANGED, "team_changed"), (SHIPPING_CHANGED, SHIPPING))
CHALLENGERS = ("team", "team_changed")
DECISION_ENTRY = 40
ATTEMPT = "team-merge-memory-gap-1"
HYPOTHESIS = ("How the team merges its specialists loses their 'not enough evidence' signal: memory returns no finding when the turns "
              "do not support one, but any finding from the other specialist makes the card answered.")
CHANGE = ("memory_gap_abstains: when the memory specialist's first pass finished cleanly with no finding, the card abstains and the "
          "other findings are withheld (saved in drops with reason memory_empty). Nothing else differs from `team`.")
SHIPPING_DECISION_ENTRY = 49
SHIPPING_DECLARED = ("The team built as production builds it (api/production.py `_TeamPerPress`): a PromptedChecker on the model that "
                     "production uses for llms['consistency'], which also gives the consistency agent its checker prompt and tool. "
                     "Added after phase-4 review D1. Reported apart from entry 48's verdict; it cannot change it.")
SELECTION_RULE = e3.SELECTION_RULE
SELECTION_APPLIED = ("single stays selected unless a challenger has a strictly higher score than single on the old set and on the new "
                     "set, and is not clearly worse than single on grounded_rate, answer_when_answerable or abstain_when_insufficient "
                     "on both sets together; if both qualify the higher combined score wins")
WEAK_SHAPES = ("topic-no-detail", "after-cut", "never-mentioned", "different-detail")  # lower question-word overlap when insufficient
PARTIAL_REL = Path("results/partial/e3t-rows.jsonl")  # git-ignored; one line per finished row
SETS = ("old", "new", "both")
SLICES = ("new_weak_24", "new_other_30")
ANSWER_RATE_NOTE = ("answer_when_answerable is an answer rate: it counts cards that did not abstain, not cards that answered correctly")
_TWIN = re.compile(r"^(syn-xtra-.+)-(ins|ans)$")


def make_assistant(system: str, sc: Scenario, writer: LLMClient, embedder: Any, retrieval_config: dict[str, Any],
                   memory: MemoryStore) -> Assistant:
    """`team_changed` is E3's team build with the one switch on, so the three systems share tools, writer and deadline.
    `team_shipping` is the plain team with a `PromptedChecker` on the writer, as production builds it.
    `team_shipping_changed` is that build with the same switch on (entry 51)."""
    if system in (SHIPPING, SHIPPING_CHANGED):
        team = e3.make_card_assistant("team", sc, writer, embedder, retrieval_config, memory, checker=PromptedChecker(writer))
        if system == SHIPPING_CHANGED:
            team.memory_gap_abstains = True  # type: ignore[attr-defined]
        return team
    if system != "team_changed":
        return e3.make_card_assistant(system, sc, writer, embedder, retrieval_config, memory)
    team = e3.make_card_assistant("team", sc, writer, embedder, retrieval_config, memory)
    team.memory_gap_abstains = True  # type: ignore[attr-defined]
    return team


# ---------------------------------------------------------------- twin pairs

def pair_key(scenario_id: str) -> tuple[str, bool] | None:
    """(pair id, is the insufficient side) for a `scenarios_extra` case id, else None."""
    m = _TWIN.match(scenario_id)
    return (m.group(1), m.group(2) == "ins") if m else None


def pair_shape(key: str) -> str:
    return _shape_of(key + "-ins")


def twin_outcomes(scs: list[Scenario], rows: list[Prediction]) -> dict[str, tuple[bool, bool]]:
    """Per pair: (the insufficient case abstained, the twin did not abstain), the two outcomes the selection metrics count."""
    by_id = {p.scenario_id: p for p in rows}
    got: dict[str, list[bool | None]] = defaultdict(lambda: [None, None])
    for s in scs:
        k = pair_key(s.id)
        card = by_id[s.id].card
        if k and card:
            got[k[0]][0 if k[1] else 1] = (card.status == "abstained") if k[1] else (card.status != "abstained")
    return {k: (bool(v[0]), bool(v[1])) for k, v in got.items() if None not in v}


def twin_table(outcomes: dict[str, tuple[bool, bool]]) -> dict[str, Any]:
    """How the pairs fall: right on both sides, answers both, abstains both, or reversed (answers the insufficient case and
    abstains on its twin). Only the first separates the two cases of a pair."""
    n = len(outcomes)
    both = sum(a and b for a, b in outcomes.values())
    return {"pairs": n, "both_right": both, "answers_both": sum(not a and b for a, b in outcomes.values()),
            "abstains_both": sum(a and not b for a, b in outcomes.values()),
            "reversed": sum(not a and not b for a, b in outcomes.values()),
            "pair_right_rate": both / n if n else None}


def pair_verdict(diff: float, lo: float, hi: float, p: float | None, binary: bool = True, higher_is_better: bool = True) -> str:
    """Better or worse only when the interval excludes zero and, for a yes/no metric, the exact McNemar test agrees."""
    if (lo > 0 or hi < 0) and (not binary or (p is not None and p < e3.ALPHA)):
        return "better" if (diff > 0) == higher_is_better else "worse"
    return "no clear difference"


def twin_compare(outcomes: dict[str, dict[str, tuple[bool, bool]]], a: str, b: str) -> dict[str, Any] | None:
    """a minus b on 'both sides of the pair right': the difference, a bootstrap interval resampling pairs, exact McNemar."""
    keys = sorted(set(outcomes[a]) & set(outcomes[b]))
    if not keys:
        return None
    ya = [all(outcomes[a][k]) for k in keys]
    yb = [all(outcomes[b][k]) for k in keys]
    diff, lo, hi = paired_bootstrap_diff([float(x) for x in ya], [float(x) for x in yb], keys, n=series.RESAMPLES, seed=series.SEED)
    p = mcnemar_exact(ya, yb)
    return {"a": a, "b": b, "pairs": len(keys), "diff": diff, "ci": [lo, hi], "mcnemar_p": p,
            f"{a}_only": sum(x and not y for x, y in zip(ya, yb, strict=True)),
            f"{b}_only": sum(y and not x for x, y in zip(ya, yb, strict=True)), "verdict": pair_verdict(diff, lo, hi, p)}


def twin_report(scs: list[Scenario], rows: dict[str, list[Prediction]]) -> dict[str, Any]:
    """Per-twin outcomes for each system over all pairs, the 24 weaker-shape pairs and the other 30, with comparisons."""
    mine = [s for s in scs if pair_key(s.id)]
    full = {n: twin_outcomes(mine, rows[n]) for n in rows}
    out: dict[str, Any] = {}
    for name, keep in (("all", lambda k: True), ("weak_24", lambda k: pair_shape(k) in WEAK_SHAPES),
                       ("other_30", lambda k: pair_shape(k) not in WEAK_SHAPES)):
        part = {n: {k: v for k, v in o.items() if keep(k)} for n, o in full.items()}
        cmp_ = [c for a, b in PAIRS if a in part and b in part and (c := twin_compare(part, a, b))]
        out[name] = {"by_system": {n: twin_table(o) for n, o in part.items()}, "comparisons": cmp_}
    out["by_shape"] = {sh: {n: twin_table({k: v for k, v in o.items() if pair_shape(k) == sh}) for n, o in full.items()}
                       for sh in SHAPES}
    return out


# ---------------------------------------------------------------- reports on a set of cases (pure, from rows)

def set_report(scs: list[Scenario], rows: dict[str, list[Prediction]], labels: str) -> dict[str, Any]:
    """Metrics, intervals and comparisons of the three systems on `scs`, McNemar counts for each pair, and a finding."""
    metrics, intervals, comparisons = e3.card_stats(scs, rows)
    paired = {f"{a}_vs_{b}": e3.paired_counts(scs, rows, (a, b)) for a, b in PAIRS if a in rows and b in rows}
    insufficient = sum(s.gold_label == "insufficient" for s in scs)
    return {"cases": len(scs), "insufficient_cases": insufficient, "answerable_cases": len(scs) - insufficient,
            "metrics": metrics, "intervals": intervals, "comparisons": comparisons, "paired_binary": paired,
            "scores": {n: e3.score(m) for n, m in metrics.items()}, "finding": finding(comparisons, paired, insufficient, labels)}


def finding(comparisons: list[dict[str, Any]], paired: dict[str, Any], insufficient: int, labels: str) -> str:
    """In words, from the saved numbers. Better or worse only when the interval excludes zero and, for a yes/no metric,
    McNemar agrees; otherwise 'no clear difference' with the counts."""
    lines = []
    for c in comparisons:
        if c["metric"] not in e3.COMPARE or "[" in c["metric"]:
            continue
        binary = c["metric"] in e3.BINARY_METRICS
        v = pair_verdict(c["diff"], *c["ci"], c["mcnemar_p"], binary, c["metric"] in e3.HIGHER_IS_BETTER)
        d = paired.get(f"{c['a']}_vs_{c['b']}", {}).get(c["metric"]) if binary else None
        counts = f", {c['a']} only {d[c['a'] + '_only']} and {c['b']} only {d[c['b'] + '_only']} of {d['cases']}" if d else ""
        test = "" if c["mcnemar_p"] is None else f", exact McNemar p={c['mcnemar_p']:.3f}"
        lines.append(f"{c['a']} vs {c['b']} on {c['metric']}: {v} (difference {c['diff']:+.3f}, 95% interval "
                     f"{c['ci'][0]:+.3f} to {c['ci'][1]:+.3f}{counts}{test})")
    return (f"{'; '.join(lines)}. {ANSWER_RATE_NOTE}. The abstention side rests on {insufficient} insufficient-evidence cases. "
            f"{labels}")


def select(sets: dict[str, dict[str, Any]], challengers: tuple[str, ...] = CHALLENGERS) -> dict[str, Any]:
    """The written rule (entry 48): `single` unless a challenger beats its score on the old set and on the new set and is
    not clearly worse on grounded_rate, answer_when_answerable or abstain_when_insufficient on both together."""
    qualified: dict[str, dict[str, Any]] = {}
    for ch in challengers:
        score = {k: sets[k]["scores"].get(ch) for k in SETS}
        base = {k: sets[k]["scores"].get("single") for k in SETS}
        if any(score[k] is None or base[k] is None for k in SETS):
            qualified[ch] = {"qualifies": False, "why": "a part of the score is undefined"}
            continue
        beats = score["old"] > base["old"] and score["new"] > base["new"]
        worse = [c["metric"] for c in sets["both"]["comparisons"]
                 if (c["a"], c["b"]) == (ch, "single") and c["metric"] in e3.HIGHER_IS_BETTER
                 and pair_verdict(c["diff"], *c["ci"], c["mcnemar_p"], c["metric"] in e3.BINARY_METRICS) == "worse"]
        qualified[ch] = {"qualifies": beats and not worse, "beats_single_on_old_and_new": beats,
                         "clearly_worse_on": worse, "score": score}
    winners = [c for c in challengers if qualified[c]["qualifies"]]
    chosen = max(winners, key=lambda c: sets["both"]["scores"][c]) if winners else "single"
    return {"rule": SELECTION_RULE, "applied": SELECTION_APPLIED, "selected": chosen, "single_score": {k: sets[k]["scores"].get("single") for k in SETS},
            "challengers": qualified,
            "note": "`single` stays selected unless the rule says otherwise; no set alone decides it. " + ANSWER_RATE_NOTE + "."}


def all_reports(by_source: dict[str, tuple[list[Scenario], dict[str, list[Prediction]]]], labels: dict[str, str],
                systems: tuple[str, ...] = SYSTEMS, challengers: tuple[str, ...] = CHALLENGERS) -> dict[str, Any]:
    """Everything derived from rows of `systems`: the old set, the new set, both together, the two slices of the new set,
    the twins. The default is entry 48's three systems, so the shipping arm's rows never enter its numbers."""
    old_s, old_r = by_source["scenarios"]
    new_s, new_r = by_source["scenarios_extra"]
    old_r = {n: old_r[n] for n in systems}
    new_r = {n: new_r[n] for n in systems}
    both_s = old_s + new_s
    both_r = {n: old_r[n] + new_r[n] for n in old_r}
    weak = [s for s in new_s if (k := pair_key(s.id)) and pair_shape(k[0]) in WEAK_SHAPES]
    other = [s for s in new_s if (k := pair_key(s.id)) and pair_shape(k[0]) not in WEAK_SHAPES]
    text = f"old: {labels['scenarios']} new: {labels['scenarios_extra']}"
    sets = {"old": set_report(old_s, old_r, labels["scenarios"]), "new": set_report(new_s, new_r, labels["scenarios_extra"]),
            "both": set_report(both_s, both_r, text)}
    slices = {"new_weak_24": set_report(weak, new_r, labels["scenarios_extra"]), "new_other_30": set_report(other, new_r, labels["scenarios_extra"])}
    return {"sets": sets, "slices": slices, "twins": twin_report(new_s, new_r), "selection": select(sets, challengers)}


def shipping_report(by_source: dict[str, tuple[list[Scenario], dict[str, list[Prediction]]]], labels: dict[str, str],
                    entry48_selected: str) -> dict[str, Any]:
    """The shipping arm beside the other three, apart from entry 48's verdict. The stop signal only states two facts:
    whether `team_shipping` beats `single` on dev under the written rule, and whether adding it would change which
    system that rule selects. Nothing acts on it."""
    full = all_reports(by_source, labels, RUN_SYSTEMS, (*CHALLENGERS, SHIPPING))
    alone = select(full["sets"], (SHIPPING,))["challengers"][SHIPPING]
    with_it = full["selection"]["selected"]
    scores = {k: {n: v["scores"].get(n) for n in RUN_SYSTEMS} for k, v in {**full["sets"], **full["slices"]}.items()}
    return {"declared": SHIPPING_DECLARED, "decision_entry": SHIPPING_DECISION_ENTRY, "systems": list(RUN_SYSTEMS),
            "scores": scores, "sets": full["sets"], "slices": full["slices"], "twins": full["twins"],
            "selection_with_it": full["selection"],
            "stop_signal": {"team_shipping_beats_single_on_dev": alone["qualifies"],
                            "detail": {k: v for k, v in alone.items() if k != "qualifies"},
                            "selected_by_entry_48": entry48_selected, "selected_with_team_shipping": with_it,
                            "adding_it_would_change_the_selection": with_it != entry48_selected,
                            "rule": "the written rule of entry 48, applied to `team_shipping` as the only challenger",
                            "action": "none in code: T9.1 is held and goes to the coordinator if either is true"}}


# ---------------------------------------------------------------- the run

def _saved(root: Path, commit: str, source: str, system: str, ids: set[str], model: str) -> dict[str, Prediction]:
    return e2._saved_rows(root / PARTIAL_REL, commit, source, system, ids, model)[0]


def run_attempt(*, root: Path = ROOT, writer: LLMClient, judge_llm: LLMClient, embedder: Any, ctl: OllamaCtl,
                names: Models | None = None, retrieval_config: dict[str, Any], run_conditions: str,
                say: Callable[[str], None] = print) -> dict[str, Path]:
    """Run the four arms on both dev sources, judge, and write one record per source. Models are unloaded at the end."""
    names = names or Models()
    mine = [(names.writer, False), (names.judge, False), (names.embedder, True)]
    try:
        if not run_conditions.strip():
            raise RunBroken("run_conditions is required: say what else was running while latency was measured")
        return _attempt(root, writer, judge_llm, embedder, ctl, names, retrieval_config, run_conditions, say)
    except LLMError as exc:
        raise RunBroken(f"a model server call failed: {exc}") from exc
    finally:
        e2._unload_models(ctl, mine, say)


def _attempt(root: Path, writer: LLMClient, judge_llm: LLMClient, embedder: Any, ctl: OllamaCtl, names: Models,
             retrieval_config: dict[str, Any], run_conditions: str, say: Callable[[str], None]) -> dict[str, Path]:
    started = datetime.now(timezone.utc).isoformat()
    if is_dirty(root):
        raise RunBroken("the working tree has uncommitted changes; commit first, or the record cannot be written")
    if retrieval_config.get("rerank") or retrieval_config.get("chunk_turns", 1) != 1:
        raise RunBroken("the retrieval config needs a one-turn chunk and no reranker")
    commit = head_commit(root)
    loaded = {}
    for source in SOURCES:
        try:
            loaded[source] = load_verified(source, SPLIT, root)
        except RuntimeError as exc:
            raise RunBroken(str(exc)) from exc
    missing = [m for m in (names.writer, names.judge, names.embedder) if full_name(m) not in ctl.installed()]
    if missing:
        raise RunBroken(f"models not installed in Ollama: {missing}")

    cold: dict[str, float] = {"judge_check": ctl.load(names.judge)}
    Judge(judge_llm).canary()
    ctl.unload(names.judge)
    cold["writer"] = ctl.load(names.writer)
    cold["embedder"] = ctl.load(names.embedder, embedding=True)
    devices = {"writer": e3.ollama_device(ctl, names.writer), "embedder": e3.ollama_device(ctl, names.embedder)}

    rows: dict[str, dict[str, list[Prediction]]] = {}
    health: dict[str, dict[str, dict[str, Any]]] = {}
    memory_failed: dict[str, list[str]] = {}
    memory_reasons: dict[str, list[dict[str, str]]] = {}
    for source in SOURCES:
        scenarios = loaded[source][0]
        have = {n: _saved(root, commit, source, n, {s.id for s in scenarios}, names.writer) for n in RUN_SYSTEMS}
        if any(have.values()):
            say(f"{source}: reusing {sum(map(len, have.values()))} saved rows")
        todo = [s for s in scenarios if any(s.id not in have[n] for n in RUN_SYSTEMS)]
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
        rows[source], health[source] = {}, {}
        for name in RUN_SYSTEMS:
            say(f"{source}: running {name}")

            def save(row: Prediction, source: str = source, name: str = name) -> None:
                entry = {"commit": commit, "subset": source, "winner": name, "model": names.writer, "calls": 0,
                         "row": json.loads(row.model_dump_json())}
                e2._append_durable(root / PARTIAL_REL, json.dumps(entry))

            rows[source][name] = e2.run_system(
                name, scenarios,
                lambda sc, name=name: make_assistant(name, sc, writer, embedder, retrieval_config, memories.get(sc.id, MemoryStore())),
                STREAK_LOCAL, say=say, have=have[name], on_row=save)
            health[source][name] = e2.check_health(name, rows[source][name])
    ctl.unload(names.writer)

    say("judging")
    cold["judge"] = ctl.load(names.judge)
    devices["judge"] = e3.ollama_device(ctl, names.judge)
    judge = Judge(judge_llm)
    for source in SOURCES:
        rows[source] = {n: e2.judge_rows(judge, p) for n, p in rows[source].items()}
    if judge.calls == 0:
        raise JudgeFailed("no finding was judged, so no card made a claim")
    judged = {"model": names.judge, "rubric_version": e2.RUBRIC_VERSION, "same_model_and_prompt_as_writer": False,
              "calls": judge.calls, "share_supported": judge.yes / judge.calls}

    labels = {s: label_text(root, s) for s in SOURCES}
    by_source = {s: (loaded[s][0], rows[s]) for s in SOURCES}
    derived = all_reports(by_source, labels)
    shipping = shipping_report(by_source, labels, derived["selection"]["selected"])
    models = {"writer": names.writer, "judge": names.judge, "embedder": names.embedder}
    paths: dict[str, Path] = {}
    for source in SOURCES:
        scenarios, digest = loaded[source]
        metrics, intervals, comparisons = e3.card_stats(scenarios, rows[source])
        report = {n: row_report(p, DEADLINE_S) for n, p in rows[source].items()}
        config: dict[str, Any] = {
            "experiment": EXPERIMENT, "part": "cards", "attempt": ATTEMPT, "hypothesis": HYPOTHESIS, "change": CHANGE,
            "decision_entry": DECISION_ENTRY, "source": source, "sources_run": list(SOURCES), "run_conditions": run_conditions,
            "devices": {**devices, "note": "from the Ollama process list right after each model loaded; unknown means it did not say"},
            "systems": list(RUN_SYSTEMS), "entry_48_systems": list(SYSTEMS), "deadline_s": DEADLINE_S, "judge": judged,
            "selection_rule": SELECTION_RULE, "selection": derived["selection"], "selection_applied": SELECTION_APPLIED,
            "sets": derived["sets"], "slices": derived["slices"], "twins": derived["twins"],
            "same_for_all": "one local writer, the same retrieval config, the same cases and deadline; single, team and team_changed "
                            "share one Tools build with no consistency checker; team_shipping adds a PromptedChecker on the same writer",
            "team_shipping": {**shipping, "consistency_checker": {
                "class": "PromptedChecker", "model": names.writer,
                "production_source": "contextlens.api.production.local_llms()['consistency'], the one OllamaClient(MODEL)",
                "also_changes": "the consistency agent's prompt and tool list (agents/graph.py, has_checker)"}},
            "retrieval": {k: retrieval_config.get(k) for k in ("winner", "mode", "k", "rerank", "chunk_turns", "run_ids")},
            "row_report": report, "health": health[source], "cold_start_s": cold,
            "drops": {n: e3.drop_report(scenarios, p) for n, p in rows[source].items()},
            "memory_build": {"cases": len(scenarios), "failed": len(memory_failed[source]), "failed_ids": memory_failed[source],
                             "failed_reasons": memory_reasons[source]},
            "prompt_note": f"prompts are unchanged, label {PROMPT_VERSION}",
            "notes": [derived["sets"]["both"]["finding"], e3.conditions_note(run_conditions), e3.deadline_sentence(report),
                      ANSWER_RATE_NOTE, "grounded_rate comes from one local judge; no person has reviewed a sample.",
                      f"Selected by the written rule: {derived['selection']['selected']}.",
                      "team_shipping (entry 49) is reported apart under config.team_shipping and is not part of that verdict; "
                      f"stop signal: {json.dumps(shipping['stop_signal']['team_shipping_beats_single_on_dev'])} (beats single), "
                      f"{json.dumps(shipping['stop_signal']['adding_it_would_change_the_selection'])} (would change the selection)."],
            "claude_calls": 0, "started_at": started,
        }
        attempts = [{"setting": n, "kind": "card system", "ok": True, "error": None, "health": health[source][n]} for n in RUN_SYSTEMS]
        paths[source] = write_run(EXPERIMENT, SPLIT, config, models, {"recall": PROMPT_VERSION}, metrics, intervals, attempts,
                                  rows[source], digest, comparisons=comparisons, scenario_source=source, root=root)
        say(f"wrote {paths[source].name}")
    return paths  # the partial rows stay: they are keyed by commit, so a later commit never reuses them


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-conditions", required=True, help="what else was running while latency was measured")
    ap.add_argument("--arm", choices=[SHIPPING_CHANGED],
                    help="run only this arm (entry 51), beside the saved rows of the two E3T records; omit to run entry 48's four systems")
    args = ap.parse_args(argv)
    if not args.run_conditions.strip():
        ap.error("--run-conditions must say something")
    from contextlens.experiments import team_arm
    from contextlens.llm.client import Embedder, OllamaClient

    run, name = (team_arm.run_arm, team_arm.EXPERIMENT) if args.arm else (run_attempt, EXPERIMENT)
    try:
        paths = run(writer=OllamaClient(e2.WRITER_MODEL), judge_llm=OllamaClient(e2.JUDGE_MODEL), embedder=Embedder(e2.EMBED_MODEL),
                    ctl=OllamaCtl(), retrieval_config=e2.load_retrieval_config(ROOT / e2.CONFIG_REL), run_conditions=args.run_conditions)
    except (RunBroken, OSError, ValueError, KeyError) as exc:
        print(f"{name} stopped, nothing more written: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: v.as_posix() for k, v in paths.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
