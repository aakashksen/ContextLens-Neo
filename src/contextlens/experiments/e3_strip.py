"""E3 consistency, part two: every checker path scored with and without the instruction strip, on the same dev cases.

Paths: the trained model, the prompted checker alone (the path `api/production.py` uses), and the cascade (trained model
first, prompted checker second), with the strip on the first stage only and on both. Flips are reported per checker and per
injected template. The dev cases where the injection is itself the statement are reported apart and never enter an average.
Nothing here chooses a version: accuracy is reported beside flips, and `accuracy_fell` lists every drop."""
from collections import Counter
from dataclasses import dataclass
from typing import Any

from contextlens.agents.consistency_check import ConsistencyChecker
from contextlens.eval.check_run import recompute_metrics
from contextlens.experiments import e3_team as e3
from contextlens.guard.instructions import instruction_kind, without_instruction_turns
from contextlens.replay.runner import Prediction
from contextlens.schemas import Scenario

NOTE = ("off: the context as it was, nothing removed. strip: earlier turns the instruction guard flags are removed from the "
        "context by `without_instruction_turns`, the one function the product's trained model uses; the statement being "
        "scored is never removed. Cascade has three arms: off, strip_first_stage (only the trained model's context), "
        "strip_both_stages (the prompted second stage too). prompted_alone is the prompted checker asked about every case, "
        "the way the product asks it. Same cases and same speaker mode in every arm. No version is chosen here by flip "
        "rate: label and alert accuracy sit beside it, and `accuracy_fell` lists every drop against the off arm.")
TEMPLATES_NOTE = ("The three fixed templates (inject0 to inject2) were read while the guard's patterns were written, so they "
                  "guided the patterns and their flip figures are not clean. Only the extra templates, if given, were written "
                  "blind to the patterns.")
STATEMENT_NOTE = ("These dev cases are prompt-injection cases in which the last turn, the one a checker scores as the statement, "
                  "is the order itself. They carry no gold label, so they are in no flip or accuracy figure above. A flagged "
                  "statement is not removed by the strip (it only touches earlier turns), so the strip does not address this "
                  "case: the checker still reads the order as the statement.")


@dataclass
class Scored:
    alerts: dict[str, dict[str, bool]]  # scenario id -> view -> whether the checker raises an alert
    labels: dict[str, str]  # scenario id -> the label given for the base view
    rows: list[Prediction]


def from_outcomes(src: str, system: str, cases: list[e3.Case3], outcomes: dict[str, dict[str, e3.Outcome]]) -> Scored:
    return Scored({i: {v: e3.alerts(o, system) for v, o in vs.items()} for i, vs in outcomes.items()},
                  {s.id: e3.system_label(outcomes[s.id]["base"], system)[0] for s, _, _ in cases},
                  e3.rows_for(src, system, cases, outcomes))


def from_prompted(src: str, checker: ConsistencyChecker, cases: list[e3.Case3], mode: str,
                  extra: tuple[e3.Template, ...]) -> Scored:
    """The prompted checker asked about every view of a labelled case, and about the base view of the rest. A failed call
    counts as no alert and as `insufficient`, the way the cascade fails closed."""
    seen: dict[tuple, tuple[str, float]] = {}

    def ask(st: Any, ea: list[Any]) -> tuple[str, float]:
        key = (st.speaker, st.text, tuple((t.speaker, t.text) for t in ea))
        if key not in seen:
            try:
                seen[key] = checker.check(st, ea)
            except Exception:
                seen[key] = ("insufficient", 0.0)
        return seen[key]

    alerts: dict[str, dict[str, bool]] = {}
    labels: dict[str, str] = {}
    rows = []
    for s, st, ea in cases:
        pairs = e3.views(st, ea, mode, extra) if s.gold_label is not None else {"base": e3.shown(st, ea, mode)}
        got = {v: ask(*pair) for v, pair in pairs.items()}
        alerts[s.id] = {v: lab == "contradiction" for v, (lab, _) in got.items()}
        label, conf = got["base"]
        labels[s.id] = label
        rows.append(e3.row_from(src, s, st, label, conf, 0))
    return Scored(alerts, labels, rows)


def _f(kn: list[int]) -> dict[str, Any]:
    return {"flipped": kn[0], "trials": kn[1], "rate": e3.rate(kn)}


def side(src: str, scored: Scored, labelled: list[e3.Case3], scenarios: list[Scenario],
         extra: tuple[e3.Template, ...]) -> dict[str, Any]:
    """Flips (speaker, each fixed template, each extra template), catch and false alerts, and label metrics."""
    speaker = [0, 0]
    fixed = {f"inject{k}": [0, 0] for k in range(len(e3.INJECTED_TURNS))}
    extras = {tid: [0, 0] for tid, _ in extra}
    pos = neg = hit = false = 0
    for s, _, _ in labelled:
        a = scored.alerts[s.id]
        base = a["base"]
        speaker[0] += a["rename"] != base
        speaker[1] += 1
        for name, kn in fixed.items():
            kn[0] += a[name] != base
            kn[1] += 1
        for k, (tid, _) in enumerate(extra):
            extras[tid][0] += a[f"extra{k}"] != base
            extras[tid][1] += 1
        if s.gold_label == "contradiction":
            pos += 1
            hit += base
        else:
            neg += 1
            false += base
    all_fixed = [sum(kn[0] for kn in fixed.values()), sum(kn[1] for kn in fixed.values())]
    all_extra = [sum(kn[0] for kn in extras.values()), sum(kn[1] for kn in extras.values())]
    m = recompute_metrics(scenarios, scored.rows)
    total = pos + neg
    return {"flips": {"speaker": _f(speaker), "fixed_templates_all": _f(all_fixed),
                      "fixed_templates": {k: _f(v) for k, v in fixed.items()},
                      "extra_templates_all": _f(all_extra) if extra else None,
                      "extra_templates": {k: _f(v) for k, v in extras.items()}},
            "catch_far": {"catch": [hit, pos], "far": [false, neg]},
            "alert_accuracy": (hit + neg - false) / total if total else None,
            "label_accuracy": m.get("label_accuracy"), "macro_f1": m.get("macro_f1")}


def statement_block(checkers: dict[str, dict[str, dict[str, Scored]]], cases: dict[str, list[e3.Case3]]) -> dict[str, Any]:
    """The prompt-injection dev cases whose statement is the order: counts and what each checker answered."""
    inj = [(s, st, ea) for s, st, ea in cases["scenarios"] if s.category == "prompt_injection"]
    ids = [s.id for s, _, _ in inj]
    return {"note": STATEMENT_NOTE, "cases": len(inj),
            "with_a_gold_label": sum(s.gold_label is not None for s, _, _ in inj),
            "statement_flagged_by_the_guard": sum(instruction_kind(st.text) is not None for _, st, _ in inj),
            "earlier_context_changed_by_the_strip": sum(len(without_instruction_turns(ea)) != len(ea) for _, _, ea in inj),
            "by_checker": {name: {arm: {"labels_given": dict(sorted(Counter(run["scenarios"].labels[i] for i in ids).items())),
                                        "alerts": sum(run["scenarios"].alerts[i]["base"] for i in ids)}
                                  for arm, run in arms.items()} for name, arms in checkers.items()}}


def measure(*, evaluator: e3.Evaluator, prompted: ConsistencyChecker, prompted_stripped: ConsistencyChecker | None, mode: str,
            cases: dict[str, list[e3.Case3]], labelled: dict[str, list[e3.Case3]], data: dict[str, tuple[list[Scenario], str]],
            extra: tuple[e3.Template, ...], extra_sha256: str | None, say: Any = print) -> dict[str, Any]:
    """Score every path with the strip off and on. `evaluator` is the run's own (strip off, `prompted` as second stage)."""
    srcs = e3.CONSISTENCY_SOURCES
    make = lambda second, strip: e3.Evaluator(evaluator.proba, evaluator.labels, second, evaluator.threshold, strip_flagged=strip)  # noqa: E731
    evals = {"off": evaluator, "strip_first_stage": make(prompted, True)}
    if prompted_stripped is not None:
        evals["strip_both_stages"] = make(prompted_stripped, True)
    outs = {}
    for arm, ev in evals.items():
        say(f"scoring the trained model and cascade, arm {arm}")
        outs[arm] = {src: e3.analyse(ev, cases[src], mode, extra) for src in srcs}

    checkers: dict[str, dict[str, dict[str, Scored]]] = {
        "trained": {"off": {s: from_outcomes(s, "trained", cases[s], outs["off"][s]) for s in srcs},
                    "strip": {s: from_outcomes(s, "trained", cases[s], outs["strip_first_stage"][s]) for s in srcs}},
        "cascade": {arm: {s: from_outcomes(s, "cascade", cases[s], outs[arm][s]) for s in srcs} for arm in evals}}
    prompted_arms = {"off": prompted, **({"strip": prompted_stripped} if prompted_stripped is not None else {})}
    checkers["prompted_alone"] = {}
    for arm, checker in prompted_arms.items():
        say(f"scoring the prompted checker alone, arm {arm}")
        checkers["prompted_alone"][arm] = {s: from_prompted(s, checker, cases[s], mode, extra) for s in srcs}

    summary = {name: {arm: {s: side(s, run[s], labelled[s], data[s][0], extra) for s in srcs} for arm, run in arms.items()}
               for name, arms in checkers.items()}
    fell = [f"{name} {arm} {s} {m}" for name, arms in summary.items() for arm, per in arms.items() if arm != "off"
            for s, v in per.items() for m in ("label_accuracy", "alert_accuracy")
            if None not in (v[m], arms["off"][s][m]) and v[m] < arms["off"][s][m]]
    return {"note": NOTE, "speaker_mode": mode, "checkers": summary, "accuracy_fell": fell,
            "templates_note": TEMPLATES_NOTE,
            "fixed_templates_flagged_by_the_guard": {f"inject{k}": instruction_kind(t) for k, t in enumerate(e3.INJECTED_TURNS)},
            "extra_templates": {"given": len(extra), "file_sha256": extra_sha256, "ids": [i for i, _ in extra],
                                "flagged_by_the_guard": {i: instruction_kind(t) for i, t in extra}},
            "cases_with_a_flagged_earlier_turn": {s: sum(len(without_instruction_turns(ea)) != len(ea) for _, _, ea in cases[s])
                                                  for s in srcs},
            "the_statement_is_the_injection": statement_block(checkers, cases),
            "not_measured": [] if prompted_stripped is not None else ["strip_both_stages and prompted_alone strip: no stripping "
                                                                      "prompted checker was given"]}
