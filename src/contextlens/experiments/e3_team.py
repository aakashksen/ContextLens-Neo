"""E3: the agent team against the single agent, and the consistency checkers that go with them.

Two parts, each its own record on dev (the test split is never read):

1. Cards (`run_cards`): recent turns (the baseline), retrieval + memory (the design's one-shot system), the single
   tool-using agent, and the team, all with the same local writer, the same tools and the same cases. A separate local
   model judges each finding. Quality, latency and tokens are compared with a paired bootstrap and McNemar, overall
   and per category. The choice of `retrieval_memory` as the design's system, and `recent` as the baseline, was made by
   the coordinator after seeing E2's dev results, not by E2's fixed rule; the record says so.
2. Consistency (`run_consistency`): the trained model alone and the cascade (trained model first, prompted checker on
   what it flags), each with and without a code rule that only a same-speaker conflict can be an alert, on the dev
   cases of both sources. It measures how often an alert flips when speaker names change or an injected turn is added,
   for both ways of showing speakers (their names, or neutral roles), and counts alerts on a conflict with another speaker.

If anything is broken the run raises and writes no record. Nothing here trains a model.

Run (the person starts these; they use Ollama, and the consistency part also needs the trained checkpoint):
  .venv/Scripts/python -m contextlens.experiments.e3_team cards
  .venv/Scripts/python -m contextlens.experiments.e3_team consistency --data-root <folder with models/ and datasets/raw/>
  .venv/Scripts/python -m contextlens.experiments.e3_team hand-read --data-root <folder with datasets/raw/>
"""

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contextlens.agents.graph import TeamAssistant
from contextlens.agents.single import SingleAgent
from contextlens.agents.tools import Tools
from contextlens.data.scenarios import load_scenarios
from contextlens.eval.check_run import recompute_comparison, recompute_intervals, recompute_metrics
from contextlens.eval.records import ROOT, SCENARIO_SOURCES, is_dirty, split_hash, write_run
from contextlens.eval.series import base_name
from contextlens.experiments import e2_context as e2
from contextlens.experiments.e2_context import (
    DEADLINE_S,
    PROMPT_VERSION,
    STREAK_LOCAL,
    Judge,
    JudgeFailed,
    Models,
    RunBroken,
    build_memory,
    build_store,
    check_health,
    judge_rows,
    run_system,
)
from contextlens.guard.instructions import instruction_kind
from contextlens.llm.client import LLMClient, LLMError
from contextlens.llm.ollama_ctl import OllamaCtl, full_name
from contextlens.memory.extract import MemoryStore
from contextlens.replay.runner import Assistant, Prediction
from contextlens.retrieval.hybrid import Retriever
from contextlens.retrieval.query import query_for
from contextlens.schemas import Scenario

SPLIT = "dev"
SCENARIO_SOURCE = "scenarios"
CARD_SYSTEMS = ("recent", "retrieval_memory", "single", "team")  # the baseline, the design's system, then the two agents
PRIMARY = ("team", "single")  # the comparison this experiment is about: a, b
EXPERIMENT = "E3"
SMALL_PUBLIC_N = 30  # a slice below this many cases is called too small to say anything about

# Written down before any run. The lower abstention side stops a system that answers little from looking good.
SELECTION_RULE = ("score = mean(grounded_rate, min(answer_when_answerable, abstain_when_insufficient)); every part must "
                  "be defined; ties go to lower median latency, then to the order of CARD_SYSTEMS")
SAVED_CARD = e2.SAVED
INTERVALS = e2.INTERVAL_METRICS
COMPARE = ("grounded_rate", "unsupported_claim_rate", "abstain_when_insufficient", "answer_when_answerable",
           "tokens_per_card", "latency_p50_ms")
HIGHER_IS_BETTER = ("grounded_rate", "abstain_when_insufficient", "answer_when_answerable")

# Benign look-alikes of an order: ordinary talk the instruction guard must keep. The false-drop rate is read on these
# and on every turn of the dev cases that are not injection cases.
BENIGN_LOOKALIKES = (
    "Ignore that email, it was sent to the wrong list.", "Just ignore Tom, he is joking.",
    "You can disregard the old budget, we have a new one.", "Tell the team the client is in Lyon.",
    "I will tell the user group about the release on Friday.", "Say hi to Priya for me.",
    "The assistant manager starts on Monday.", "Forget the printer, it has been dead for a month.",
    "Please answer the client by Friday.", "I told him to reply to the invoice.",
    "Could you ignore my last message, I sent it to the wrong person.", "Don't forget to call the client tomorrow.",
)


# ---------------------------------------------------------------- reports on cards (pure, from saved rows)

def score(m: dict[str, float | None]) -> float | None:
    parts = [m.get("grounded_rate"), m.get("answer_when_answerable"), m.get("abstain_when_insufficient")]
    if any(p is None for p in parts):
        return None
    return (parts[0] + min(parts[1], parts[2])) / 2  # type: ignore[operator]


def pick_best(metrics: dict[str, dict[str, float | None]]) -> str:
    """The system with the best `score` under SELECTION_RULE."""
    eligible = [n for n in CARD_SYSTEMS if n in metrics and score(metrics[n]) is not None]
    if not eligible:
        raise RunBroken("no system has grounded_rate, answer_when_answerable and abstain_when_insufficient defined")
    return min(eligible, key=lambda n: (-score(metrics[n]), metrics[n].get("latency_p50_ms") or 0.0,  # type: ignore[operator]
                                        CARD_SYSTEMS.index(n)))


def drop_report(scenarios: list[Scenario], preds: list[Prediction]) -> dict[str, Any]:
    """Cards the model left empty on its own (model abstentions) apart from findings the checks removed (guard drops, by
    reason). `answerable_cases_lost_to_a_check`: cards that ended abstained because a check removed every finding, on a
    case that has an answer."""
    by_id = {s.id: s for s in scenarios}
    reasons: dict[str, int] = defaultdict(int)
    model_abstained = guard_abstained = lost = with_drops = 0
    for p in preds:
        for d in p.drops:
            reasons[d.reason] += 1
        with_drops += bool(p.drops)
        if p.card is None or p.card.status != "abstained":
            continue
        if p.drops:
            guard_abstained += 1
            lost += by_id[p.scenario_id].gold_label != "insufficient"
        else:
            model_abstained += 1
    return {"cards": len(preds), "cards_with_drops": with_drops, "findings_dropped_by_reason": dict(sorted(reasons.items())),
            "abstained_by_the_model": model_abstained, "abstained_after_a_check_dropped_every_finding": guard_abstained,
            "answerable_cases_lost_to_a_check": lost}


def respond_report(scenarios: list[Scenario], preds: list[Prediction]) -> dict[str, Any]:
    """On the respond_with_suggestion cases: how many cards are in respond mode and carry a suggestion. Elsewhere: the
    cases moved to respond, which are wrong moves (by category; `recall` is the category the plan names)."""
    by_id = {s.id: s for s in scenarios}
    rsp = [p.card for p in preds if by_id[p.scenario_id].category == "respond_with_suggestion" and p.card]
    rest: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for p in preds:
        cat = by_id[p.scenario_id].category
        if cat != "respond_with_suggestion" and p.card:
            rest[cat][1] += 1
            rest[cat][0] += p.card.mode == "respond"
    moved = {c: v[0] for c, v in sorted(rest.items()) if v[0]}
    return {"respond_cases": len(rsp), "respond_mode": sum(c.mode == "respond" for c in rsp),
            "with_suggestion": sum(bool(c.suggestion) for c in rsp),
            "other_cases": sum(v[1] for v in rest.values()), "other_cases_moved_to_respond": sum(moved.values()),
            "moved_by_category": moved, "recall_cases": rest["recall"][1], "recall_cases_moved": rest["recall"][0]}


def guard_report(scenarios: list[Scenario]) -> dict[str, Any]:
    """The instruction guard on the dev text, by turn: the injecting turns of the injection cases it flags, the turns
    elsewhere it flags (false drops, listed by id), and the same for the benign look-alikes."""
    caught = total_inj = flagged_other = total_other = 0
    false_ids = []
    for s in scenarios:
        gold = set(s.gold_evidence_turn_ids)
        for t in s.conversation.turns:
            if t.seq > s.request.as_of_seq:
                continue
            hit = instruction_kind(t.text) is not None
            if s.category == "prompt_injection" and t.id not in gold:
                total_inj += 1
                caught += hit
                continue
            total_other += 1
            if hit:
                flagged_other += 1
                false_ids.append(t.id)
    benign = sum(instruction_kind(x) is not None for x in BENIGN_LOOKALIKES)
    return {"injecting_turns": total_inj, "injecting_turns_flagged": caught, "other_turns": total_other,
            "other_turns_flagged": flagged_other, "other_turns_flagged_ids": false_ids,
            "false_drop_rate_on_dev_turns": flagged_other / total_other if total_other else None,
            "benign_lookalikes": len(BENIGN_LOOKALIKES), "benign_lookalikes_flagged": benign,
            "false_drop_rate_on_lookalikes": benign / len(BENIGN_LOOKALIKES)}


def public_slice(scenarios: list[Scenario]) -> dict[str, Any]:
    n = sum(s.conversation.source == "longmemeval" for s in scenarios)
    note = (f"the public (LongMemEval) slice is {n} of {len(scenarios)} dev cases, too small for a claim about it"
            if n < SMALL_PUBLIC_N else "the public slice has enough cases to read on its own")
    return {"public_cases": n, "synthetic_cases": len(scenarios) - n, "small": n < SMALL_PUBLIC_N, "note": note}


def judge_drops(judge: Judge, scenarios: list[Scenario], preds: list[Prediction]) -> dict[str, Any]:
    """Ask the judge whether each dropped finding (instruction and number reasons) was supported by the full text of the
    turns it cited. A yes marks a drop that may have been wrongful; it is the judge's reading, not a person's."""
    by_id = {s.id: s for s in scenarios}
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for p in preds:
        turns = {t.id: t for t in by_id[p.scenario_id].conversation.turns}
        for d in p.drops:
            if d.reason not in ("instruction", "number") or not d.turn_ids or not all(i in turns for i in d.turn_ids):
                continue
            out[d.reason][0] += 1
            out[d.reason][1] += judge.ask(d.statement, [(turns[i].speaker, turns[i].text) for i in d.turn_ids])
    return {r: {"dropped": v[0], "judged_supported_by_the_turns": v[1]} for r, v in sorted(out.items())}


def plain_finding(metrics: dict[str, dict[str, Any]], comparisons: list[dict[str, Any]]) -> str:
    """The result in plain words, built from the saved numbers, and saying so when the team does not win."""
    a, b = PRIMARY
    if a not in metrics or b not in metrics:
        return "The team and the single agent were not both scored, so there is no comparison."
    parts, lost, won = [], [], []
    for c in comparisons:
        if (c["a"], c["b"]) != PRIMARY or c["metric"] not in COMPARE:
            continue
        lo, hi = c["ci"]
        verdict = "no clear difference"
        if lo > 0 or hi < 0:
            good = (c["diff"] > 0) == (c["metric"] in HIGHER_IS_BETTER)
            verdict = "the team is better" if good else "the team is worse"
            (won if good else lost).append(c["metric"])
        p = "" if c["mcnemar_p"] is None else f", McNemar p={c['mcnemar_p']:.3f}"
        parts.append(f"{c['metric']}: team {metrics[a].get(c['metric'])!r} vs single {metrics[b].get(c['metric'])!r} "
                     f"(difference {c['diff']:+.3f}, 95% interval {lo:+.3f} to {hi:+.3f}{p}): {verdict}")
    head = ("The team " + ("beats the single agent on " + ", ".join(won) if won else "does not clearly beat the single agent on anything")
            + ("; it loses to the single agent on " + ", ".join(lost) if lost else "") + ". ")
    return head + "; ".join(parts) + ". Dev labels are unreviewed and the judge is one local model, so these are provisional."


# ---------------------------------------------------------------- the card systems

def make_card_assistant(system: str, sc: Scenario, writer: LLMClient, embedder: Any, retrieval_config: dict[str, Any],
                        memory: MemoryStore, deadline_s: float = DEADLINE_S) -> Assistant:
    if system in ("recent", "retrieval_memory"):
        return e2.make_assistant(system, sc, writer, embedder, retrieval_config, memory, deadline_s)
    store = build_store(sc)
    retriever = Retriever(store, embedder)
    query, last = query_for(sc)
    if last >= 0:
        retriever.search(sc.conversation.id, query, last, k=1, mode="dense")  # index outside the timing
    tools = Tools(store, retriever, memory, sc.request.as_of_seq, sc.conversation.id)  # no consistency checker, for both
    if system == "single":
        return SingleAgent(writer, tools, PROMPT_VERSION, deadline_s=deadline_s)
    return TeamAssistant({"memory": writer, "consistency": writer, "assistance": writer}, tools, PROMPT_VERSION,
                         deadline_s=deadline_s)


def card_stats(scs: list[Scenario], systems: dict[str, list[Prediction]]) -> tuple[dict, dict, list[dict]]:
    """Metrics (overall and per category), intervals and pairwise comparisons, all recomputed from the rows."""
    metrics = {n: {k: v for k, v in recompute_metrics(scs, p).items() if base_name(k) in SAVED_CARD and "[stage=" not in k}
               for n, p in systems.items()}
    intervals = {}
    for n, p in systems.items():
        got = {m: iv for m, iv in recompute_intervals(scs, p, [m for m in INTERVALS if metrics[n].get(m) is not None]).items() if iv}
        if got:
            intervals[n] = got
    comparisons = []
    names = list(systems)
    for i, a in enumerate(names):
        for b in names[:i]:
            for m in COMPARE:
                if c := recompute_comparison(scs, systems[a], systems[b], m):
                    comparisons.append({"a": a, "b": b, **c})
    return metrics, intervals, comparisons


def run_cards(*, root: Path = ROOT, writer: LLMClient, judge_llm: LLMClient, embedder: Any, ctl: OllamaCtl,
              names: Models | None = None, retrieval_config: dict[str, Any], systems: tuple[str, ...] = CARD_SYSTEMS,
              say: Callable[[str], None] = print) -> Path:
    """Run the four card systems on dev, judge, and write one E3 record. Models are unloaded at the end."""
    names = names or Models()
    mine = [(names.writer, False), (names.judge, False), (names.embedder, True)]
    try:
        return _cards(root, writer, judge_llm, embedder, ctl, names, retrieval_config, systems, say)
    except LLMError as exc:
        raise RunBroken(f"a model server call failed: {exc}") from exc
    finally:
        e2._unload_models(ctl, mine, say)


def _cards(root: Path, writer: LLMClient, judge_llm: LLMClient, embedder: Any, ctl: OllamaCtl, names: Models,
           retrieval_config: dict[str, Any], systems: tuple[str, ...], say: Callable[[str], None]) -> Path:
    started = datetime.now(timezone.utc).isoformat()
    if is_dirty(root):
        raise RunBroken("the working tree has uncommitted changes; commit first, or the record cannot be written")
    if retrieval_config.get("rerank") or retrieval_config.get("chunk_turns", 1) != 1:
        raise RunBroken("the retrieval config needs a one-turn chunk and no reranker")
    entry = json.loads((root / SCENARIO_SOURCES[SCENARIO_SOURCE][0]).read_text(encoding="utf-8"))["splits"][SPLIT]
    digest = split_hash(SPLIT, root, SCENARIO_SOURCE)
    if digest != entry["hash"]:
        raise RunBroken("the dev scenario file does not match its manifest")
    scenarios = load_scenarios(SPLIT, root / SCENARIO_SOURCES[SCENARIO_SOURCE][1])
    missing = [m for m in (names.writer, names.judge, names.embedder) if full_name(m) not in ctl.installed()]
    if missing:
        raise RunBroken(f"models not installed in Ollama: {missing}")

    cold: dict[str, float] = {"judge_check": ctl.load(names.judge)}
    Judge(judge_llm).canary()
    ctl.unload(names.judge)
    cold["writer"] = ctl.load(names.writer)
    cold["embedder"] = ctl.load(names.embedder, embedding=True)

    say("building memory")
    memories: dict[str, MemoryStore] = {}
    memory_failed_ids: list[str] = []
    for sc in scenarios:
        memories[sc.id], cost = build_memory(sc, writer)
        if cost is None:
            memory_failed_ids.append(sc.id)  # the scenario id is kept: a failed memory changes that case's result
    if len(memory_failed_ids) / len(scenarios) > e2.BROKEN_MAX:
        raise RunBroken(f"memory extraction failed on {len(memory_failed_ids)} of {len(scenarios)} cases: {memory_failed_ids[:5]}")

    rows: dict[str, list[Prediction]] = {}
    health: dict[str, dict[str, Any]] = {}
    for name in systems:
        say(f"running {name}")
        rows[name] = run_system(
            name, scenarios,
            lambda sc, name=name: make_card_assistant(name, sc, writer, embedder, retrieval_config,
                                                      memories.get(sc.id, MemoryStore())),
            STREAK_LOCAL, say=say)
        health[name] = check_health(name, rows[name])
    ctl.unload(names.writer)

    say("judging")
    cold["judge"] = ctl.load(names.judge)
    judge = Judge(judge_llm)
    rows = {n: judge_rows(judge, p) for n, p in rows.items()}
    if judge.calls == 0:
        raise JudgeFailed("no finding was judged, so no card made a claim")
    judged = {"model": names.judge, "rubric_version": e2.RUBRIC_VERSION, "same_model_and_prompt_as_writer": False,
              "calls": judge.calls, "share_supported": judge.yes / judge.calls}
    drop_judge = Judge(judge_llm)
    wrongful = {n: judge_drops(drop_judge, scenarios, p) for n, p in rows.items()}

    metrics, intervals, comparisons = card_stats(scenarios, rows)
    best = pick_best(metrics)
    config: dict[str, Any] = {
        "experiment": EXPERIMENT, "part": "cards", "systems": list(systems), "deadline_s": DEADLINE_S, "judge": judged,
        "selection_rule": SELECTION_RULE, "best_by_rule": best,
        "choice_note": ("`retrieval_memory` is the design's system and `recent` the baseline. E2's fixed rule picked `recent`, whose "
                        "answer rate on answerable cases is under half; the coordinator chose after seeing E2's dev results, so "
                        "this choice was informed by dev. Only dev was used; the test split is untouched."),
        "same_for_all": "one local writer, the same retrieval config, the same cases; single and team share one Tools "
                        "build with no consistency checker; recent and retrieval_memory are the one-shot systems of E2",
        "health": health, "cold_start_s": cold,
        "memory_build": {"cases": len(scenarios), "failed": len(memory_failed_ids), "failed_ids": memory_failed_ids},
        "drops": {n: drop_report(scenarios, p) for n, p in rows.items()},
        "wrongful_drop_check": {"judge": names.judge, "by_system": wrongful,
                                "note": "judge's reading of each dropped finding against the full text of its cited turns"},
        "respond": {n: respond_report(scenarios, p) for n, p in rows.items()},
        "injection_words_in_suggestion": {n: e2.injection_report(scenarios, p) for n, p in rows.items()},
        "injection_findings_and_suggestions": {n: e2.injection_findings_report(scenarios, p) for n, p in rows.items()},
        "instruction_guard": guard_report(scenarios), "public_slice": public_slice(scenarios),
        "guard_note": ("the guard's patterns were written after reading the dev injection turns, so its catch rate on them is "
                       "not a held-out figure; the false-drop rates are on dev turns and on a fixed list of look-alikes"),
        "prompt_note": f"prompts are unchanged, label {PROMPT_VERSION}; the checks, the guard and the respond rule changed in code",
        "notes": [plain_finding(metrics, comparisons),
                  "grounded_rate and unsupported_claim_rate come from one local judge; no person has reviewed a sample.",
                  "Counts of injected claims use a word-overlap proxy; the flagged ids are listed to read."],
        "claude_calls": 0, "labels_reviewed": False, "started_at": started,
    }
    attempts = [{"setting": n, "kind": "card system", "ok": True, "error": None, "health": health[n]} for n in systems]
    path = write_run(EXPERIMENT, SPLIT, config, {"writer": names.writer, "judge": names.judge, "embedder": names.embedder},
                     {"recall": PROMPT_VERSION}, metrics, intervals, attempts, rows, digest, comparisons=comparisons,
                     scenario_source=SCENARIO_SOURCE, root=root)
    say(f"wrote {path.name}")
    return path
