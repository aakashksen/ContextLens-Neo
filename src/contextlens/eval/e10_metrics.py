"""E10's code-scored metrics. Every number comes from a saved card and the case's gold; no model judges anything here.

A card is scored per case (`case_scores`), the per-case table is what a record keeps (ids, booleans, numbers, no text), and
every metric below is a function of that table alone, so a record's numbers recompute from the record.

Metrics (name: cases it covers, what counts):
  cite_any / cite_all     qmsum_long: the card cites at least one / every gold evidence turn (P1)
  catch_rate              conflict_contradiction and conflict_update: a finding carries the gold label and cites both gold turns (P2)
  false_alert_rate        conflict_consistent: a finding labelled contradiction or update (any citation)
  abstain_when_insufficient / answer_when_answerable   twin_insufficient / twin_answerable (P3)
  twin_pair_ok            both twins of a pair right: abstained on the insufficient one, answered with a gold citation on the other
  precision               every case with gold: share of cited turns that are gold (cards that cite nothing are left out)
  label_accuracy          conflict cases and control cases with a gold label: the card's label equals it
  deadline_cut_rate       every case: the card says the deadline cut it
  tokens, latency         mean/p50/p95 per card
"""

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from contextlens.eval import series
from contextlens.eval.stats import bootstrap_ci, mcnemar_exact, paired_bootstrap_diff, percentile
from contextlens.replay.runner import Prediction
from contextlens.schemas import Card, Scenario

ALERTS = ("contradiction", "update")
CONFLICT = ("conflict_contradiction", "conflict_update", "conflict_consistent")
EXPECTED_RESPOND = "control_respond_with_suggestion"
CUT_MS = 7700  # a press this long that ended in a timeout was cut by the 8 s deadline


def cited_turns(card: Card) -> set[str]:
    return {e.turn_id for f in card.findings for e in f.evidence}


def deadline_cut(p: Prediction) -> bool:
    """The card says the deadline cut it: a stage note "deadline reached", or a model call that timed out at the deadline
    (the single agent reports that as an llm error, with the press at about 8 s)."""
    if p.card is None:
        return False
    if any("deadline reached" in n for n in p.card.missing):
        return True
    return p.stage_latency_ms.get("total", 0) >= CUT_MS and any("timed out" in n for n in p.card.missing)


def case_scores(sc: Scenario, p: Prediction, group: str) -> dict[str, Any]:
    """One row of the per-case table. Fields that do not apply to the case are None."""
    card = p.card
    gold = set(sc.gold_evidence_turn_ids)
    row: dict[str, Any] = {
        "id": sc.id, "category": sc.category, "group": group, "errored": bool(p.errored),
        "deadline_cut": deadline_cut(p), "tokens": card.tokens if card else None,
        "latency_ms": p.stage_latency_ms.get("total", card.latency_ms if card else None),
    }
    if card is None:
        return {**row, "status": None, "label": None}
    cited = cited_turns(card)
    label = series.label_of(card)
    abstained = card.status == "abstained"
    row.update(status=card.status, mode=card.mode, label=label, cited_n=len(cited), n_findings=len(card.findings))
    row["precision"] = len(cited & gold) / len(cited) if gold and cited else None
    row["cite_any"] = bool(cited & gold) if gold else None
    row["cite_all"] = gold <= cited if gold else None
    alert_findings = [f for f in card.findings if f.label in ALERTS]
    row["alert"] = bool(alert_findings)
    row["answered"] = not abstained
    cat = sc.category
    if cat in CONFLICT:
        both = [f for f in card.findings if gold <= {e.turn_id for e in f.evidence}]
        row["label_ok"] = label == sc.gold_label
        if cat == "conflict_consistent":
            row["false_alert"] = bool(alert_findings)
        else:
            row["caught"] = any(f.label == sc.gold_label for f in both)
            row["flagged_both"] = any(f.label in ALERTS for f in both)
    elif cat == "twin_insufficient":
        row["abstained_ok"] = abstained
    elif cat == "twin_answerable":
        row["answered_ok"] = not abstained
    elif cat.startswith("control_") and sc.gold_label:
        row["label_ok"] = label == sc.gold_label
    if cat.startswith("control_"):
        row["mode_ok"] = (card.mode == "respond") == (cat == EXPECTED_RESPOND)
        if sc.gold_label == "insufficient":
            row["abstained_ok"] = abstained
        elif sc.gold_label is None and sc.request.question:
            row["answered_ok"] = not abstained
    return row


def score_arm(scenarios: Sequence[Scenario], rows: Sequence[Prediction], groups: dict[str, str]) -> list[dict[str, Any]]:
    by_id = {p.scenario_id: p for p in rows}
    return [case_scores(s, by_id[s.id], groups[s.id]) for s in scenarios if s.id in by_id]


# ---------------------------------------------------------------- metric definitions

def _pair_ok(table: list[dict[str, Any]]) -> list[tuple[str, str, float]]:
    """(unit id, group, 1.0 when both twins are right) per twin pair that has both rows."""
    pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in table:
        if r["category"] in ("twin_insufficient", "twin_answerable"):
            pairs[r["id"].rsplit("-", 1)[0]][r["category"]] = r
    out = []
    for key, d in sorted(pairs.items()):
        if len(d) == 2:
            a, i = d["twin_answerable"], d["twin_insufficient"]
            out.append((key, i["group"], float(bool(i.get("abstained_ok")) and bool(a.get("answered_ok")) and bool(a.get("cite_any")))))
    return out


# name -> (categories, field, kind). `kind` binary: a per-case 0/1; `mean`: a per-case number.
BINARY: dict[str, tuple[tuple[str, ...], str]] = {
    "cite_any": (("qmsum_long",), "cite_any"),
    "cite_all": (("qmsum_long",), "cite_all"),
    "catch_rate": (("conflict_contradiction", "conflict_update"), "caught"),
    "catch_contradiction": (("conflict_contradiction",), "caught"),
    "catch_update": (("conflict_update",), "caught"),
    "flag_both_rate": (("conflict_contradiction", "conflict_update"), "flagged_both"),
    "false_alert_rate": (("conflict_consistent",), "false_alert"),
    "conflict_label_accuracy": (CONFLICT, "label_ok"),
    "abstain_when_insufficient": (("twin_insufficient",), "abstained_ok"),
    "answer_when_answerable": (("twin_answerable",), "answered_ok"),
    "twin_answerable_cite_any": (("twin_answerable",), "cite_any"),
    "deadline_cut_rate": ((), "deadline_cut"),
    "errored_rate": ((), "errored"),
}
CONTROL_BINARY: dict[str, str] = {"control_label_accuracy": "label_ok", "control_abstain": "abstained_ok",
                                  "control_answer": "answered_ok", "control_mode_accuracy": "mode_ok"}
CONTINUOUS = ("precision", "tokens", "latency_ms")


def units(table: list[dict[str, Any]], metric: str) -> list[tuple[str, str, float]]:
    """(unit id, group, value) for each case or pair the metric covers; values are 0/1 or a number."""
    if metric == "twin_pair_ok":
        return _pair_ok(table)
    if metric in BINARY:
        cats, field = BINARY[metric]
        pick = [r for r in table if (not cats or r["category"] in cats) and r.get(field) is not None]
    elif metric in CONTROL_BINARY:
        pick = [r for r in table if r["category"].startswith("control_") and r.get(CONTROL_BINARY[metric]) is not None]
        field = CONTROL_BINARY[metric]
    elif metric in CONTINUOUS:
        pick = [r for r in table if r.get(metric) is not None]
        field = metric
    elif metric.startswith(("tokens[", "latency_ms[", "precision[")):  # tokens[qmsum_long] ... by category prefix
        base, cat = metric[:-1].split("[", 1)
        pick = [r for r in table if r["category"].startswith(cat) and r.get(base) is not None]
        field = base
    else:
        raise KeyError(metric)
    return [(r["id"], r["group"], float(r[field])) for r in pick]


METRICS = (*BINARY, *CONTROL_BINARY, "twin_pair_ok", *CONTINUOUS)
SLICES = ("qmsum_long", "conflict", "twin", "control")  # cost metrics are also reported per case type


def summarize(table: list[dict[str, Any]], resamples: int = 2000) -> dict[str, dict[str, Any]]:
    """Per metric: n, value (a rate or mean), 95% interval over whole meetings; cost metrics add p50 and p95."""
    out: dict[str, dict[str, Any]] = {}
    names = [*METRICS, *(f"{b}[{c}]" for b in ("tokens", "latency_ms") for c in SLICES)]
    for name in names:
        u = units(table, name)
        if not u:
            continue
        vals = [v for _, _, v in u]
        entry: dict[str, Any] = {"n": len(vals), "value": sum(vals) / len(vals),
                                 "ci": list(bootstrap_ci(vals, [g for _, g, _ in u], n=resamples))}
        if name.split("[")[0] in ("tokens", "latency_ms"):
            entry.update(p50=percentile(vals, 50), p95=percentile(vals, 95))
        out[name] = entry
    return out


def compare(a: list[dict[str, Any]], b: list[dict[str, Any]], metric: str, resamples: int = 5000) -> dict[str, Any] | None:
    """Paired comparison of a against b on the cases both have: mean difference with a 95% interval resampling whole
    meetings, and for a 0/1 metric the exact McNemar p and the discordant counts. None when no unit is shared."""
    ua, ub = {i: (g, v) for i, g, v in units(a, metric)}, {i: (g, v) for i, g, v in units(b, metric)}
    ids = sorted(set(ua) & set(ub))
    if not ids:
        return None
    va, vb, groups = [ua[i][1] for i in ids], [ub[i][1] for i in ids], [ua[i][0] for i in ids]
    diff, lo, hi = paired_bootstrap_diff(va, vb, groups, n=resamples)
    out: dict[str, Any] = {"metric": metric, "n": len(ids), "a_value": sum(va) / len(va), "b_value": sum(vb) / len(vb),
                           "diff": diff, "ci": [lo, hi], "mcnemar_p": None}
    if metric not in CONTINUOUS and not metric.startswith(("tokens[", "latency_ms[")):
        x, y = [v == 1.0 for v in va], [v == 1.0 for v in vb]
        out.update(mcnemar_p=mcnemar_exact(x, y), a_only=sum(p and not q for p, q in zip(x, y)),
                   b_only=sum(q and not p for p, q in zip(x, y)))
    return out


# ---------------------------------------------------------------- the claim rule (written before any run)

PRIMARIES = ("cite_any", "catch_rate", "twin_pair_ok")  # P1, P2, P3; Holm over three
HOLM_ALPHA = 0.05 / 3
ARMS = {"A": "single", "B": "single_plus", "C": "team_shipping_changed", "D": "team_routed"}
CUT_MAX = 0.05
LATENCY_RATIO_MAX = 1.25
TOKEN_RATIO_MAX = 1.5

CLAIM_RULE = (
    "Written before any run. 'The team catches contradictions or long-range evidence a single agent misses' may be said if "
    "team_routed (D) beats single_plus (B) on P1 (cite_any on qmsum_long) or P2 (catch_rate on conflict cases) with a 95% interval "
    "over meetings that excludes zero and exact McNemar p below 0.0167 (three primaries, Holm), D is not clearly worse than B on "
    "the other primaries or on false_alert_rate (a clearly worse metric is one whose interval lies wholly below zero, or above "
    "for false alerts), and D's deadline_cut_rate is at most 5%. 'At single-agent cost' may be added only if D's median latency "
    "on the control cases is within 1.25x of single's (A) and its mean tokens within 1.5x. If B matches D the finding is that "
    "the checker and retrieval help and the multi-agent structure does not. If C (team_shipping_changed) loses to A again, that "
    "is reported.")


def _sig_win(c: dict[str, Any] | None) -> bool:
    return bool(c and c["diff"] > 0 and c["ci"][0] > 0 and c["mcnemar_p"] is not None and c["mcnemar_p"] < HOLM_ALPHA)


def claim_verdict(tables: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Apply CLAIM_RULE to per-arm tables keyed by arm name. Needs all four arms, else says which are missing."""
    missing = [n for n in ARMS.values() if n not in tables]
    if missing:
        return {"decided": False, "reason": f"arms missing: {missing}"}
    d, b, a = tables[ARMS["D"]], tables[ARMS["B"]], tables[ARMS["A"]]
    cmp = {m: compare(d, b, m) for m in (*PRIMARIES, "false_alert_rate")}
    wins = [m for m in ("cite_any", "catch_rate") if _sig_win(cmp[m])]
    worse = [m for m in PRIMARIES if cmp[m] and cmp[m]["diff"] < 0 and cmp[m]["ci"][1] < 0]
    fa = cmp["false_alert_rate"]
    if fa and fa["diff"] > 0 and fa["ci"][0] > 0:
        worse.append("false_alert_rate")
    cut = summarize(d).get("deadline_cut_rate", {}).get("value")
    cost: dict[str, Any] = {}
    lat_d = [v for _, _, v in units([r for r in d if r["category"].startswith("control_")], "latency_ms")]
    lat_a = [v for _, _, v in units([r for r in a if r["category"].startswith("control_")], "latency_ms")]
    tok_d = [v for _, _, v in units([r for r in d if r["category"].startswith("control_")], "tokens")]
    tok_a = [v for _, _, v in units([r for r in a if r["category"].startswith("control_")], "tokens")]
    if lat_d and lat_a and tok_d and tok_a and percentile(lat_a, 50) > 0 and sum(tok_a) > 0:  # a zero base has no ratio
        cost = {"latency_ratio": percentile(lat_d, 50) / percentile(lat_a, 50), "token_ratio": (sum(tok_d) / len(tok_d)) / (sum(tok_a) / len(tok_a))}
    team_claim = bool(wins) and not worse and cut is not None and cut <= CUT_MAX
    return {"decided": True, "team_beats_single_plus_on": wins, "team_clearly_worse_on": worse, "d_deadline_cut_rate": cut,
            "may_claim_team_catches_more": team_claim, "cost": cost,
            "may_add_single_agent_cost": bool(team_claim and cost and cost["latency_ratio"] <= LATENCY_RATIO_MAX
                                              and cost["token_ratio"] <= TOKEN_RATIO_MAX),
            "d_vs_b": cmp, "c_vs_a": {m: compare(tables[ARMS["C"]], a, m) for m in (*PRIMARIES, "false_alert_rate")},
            "rule": CLAIM_RULE}
