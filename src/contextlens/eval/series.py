"""Per-unit values behind a metric, so an interval or a paired comparison can be recomputed from
saved prediction rows. A series is None when a source field is missing on any row, or when no unit
exists: that value is then not recomputable."""
import re
from collections.abc import Sequence

from contextlens.eval import metrics as M
from contextlens.eval.stats import bootstrap_ci, mcnemar_exact, paired_bootstrap_diff
from contextlens.replay.runner import Prediction
from contextlens.schemas import Card, Scenario

Rows = Sequence[tuple[Scenario, Prediction]]
SEED = 13
RESAMPLES = 10_000
Unit = tuple[str, str, float]  # scenario id, conversation id, value

PER_SCENARIO = ("label_accuracy", "mrr", "recall@", "tokens_per_card", "cost_usd_per_card", "schema_valid_rate")
BINARY = ("label_accuracy", "schema_valid_rate", "recall@")
JUDGED = ("grounded_rate", "unsupported_claim_rate")


def label_of(card: Card) -> str:
    for f in card.findings:
        if f.label:
            return f.label
    return "insufficient"


def schema_valid(raw: str) -> bool:
    try:
        Card.model_validate_json(raw)
    except ValueError:
        return False
    return True


def grounded_per_card(rows: Rows) -> list[list[bool]] | None:
    """Per row, one flag per finding. None when a card with findings lacks a judgement for each."""
    if not all(p.claim_supported is not None and len(p.claim_supported) == len(p.card.findings)
               for _, p in rows if p.card is not None and p.card.findings):
        return None
    out: list[list[bool]] = []
    for s, p in rows:
        if p.card is None:
            out.append([])
            continue
        judged = {id(f): j for f, j in zip(p.card.findings, p.claim_supported or [])}
        turn_seq = {t.id: t.seq for t in s.conversation.turns}
        out.append(M.grounded_flags([p.card], turn_seq, s.request.as_of_seq, lambda f, judged=judged: judged[id(f)]))
    return out


def base_name(name: str) -> str:
    return name.split("[", 1)[0]


def _is(name: str, group: tuple[str, ...]) -> bool:
    return any(name == g or (g.endswith("@") and re.fullmatch(re.escape(g) + r"\d+", name)) for g in group)


def is_per_scenario(name: str) -> bool:
    return _is(name, PER_SCENARIO)


def is_binary(name: str) -> bool:
    return _is(name, BINARY)


def units(rows: Rows, name: str) -> list[Unit] | None:
    carded = all(p.card is not None for _, p in rows)
    out: list[Unit] = []
    if name == "label_accuracy":
        if not carded:
            return None
        out = [(s.id, s.conversation.id, float(label_of(p.card) == s.gold_label))  # type: ignore[arg-type]
               for s, p in rows if s.gold_label]
    elif name in ("tokens_per_card", "cost_usd_per_card"):
        if not carded:
            return None
        out = [(s.id, s.conversation.id, float(p.card.tokens if name == "tokens_per_card" else p.card.cost_usd))  # type: ignore[union-attr]
               for s, p in rows]
    elif name == "schema_valid_rate":
        if any(p.raw_output is None for _, p in rows):
            return None
        out = [(s.id, s.conversation.id, float(schema_valid(p.raw_output or ""))) for s, p in rows]
    elif name == "mrr" or re.fullmatch(r"recall@\d+", name):
        need = [(s, p) for s, p in rows if s.gold_evidence_turn_ids]
        if any(p.ranked_turn_ids is None for _, p in need):
            return None
        for s, p in need:
            ranked = p.ranked_turn_ids or []
            if name == "mrr":
                rank = M.first_gold_rank(ranked, s.gold_evidence_turn_ids)
                v = 1.0 / rank if rank else 0.0
            else:
                v = M.recall_at_k(ranked, s.gold_evidence_turn_ids, int(name.split("@")[1])) or 0.0
            out.append((s.id, s.conversation.id, v))
    elif name in JUDGED:
        flags = grounded_per_card(rows)
        if flags is None:
            return None
        for (s, _), fl in zip(rows, flags):
            if name == "grounded_rate":
                out += [(s.id, s.conversation.id, float(x)) for x in fl]
            elif fl:
                out.append((s.id, s.conversation.id, sum(not x for x in fl) / len(fl)))
    elif name == "speaker_accuracy":
        for s, p in rows:
            speaker_of = {t.id: t.speaker for t in s.conversation.turns}
            for f in (p.card.findings if p.card else []):
                out += [(s.id, s.conversation.id, float(ev.speaker == speaker_of[ev.turn_id]))
                        for ev in f.evidence if ev.turn_id in speaker_of]
    else:
        return None
    return out or None


def interval(rows: Rows, name: str) -> tuple[float, float] | None:
    u = units(rows, name)
    if u is None:
        return None
    return bootstrap_ci([v for _, _, v in u], [g for _, g, _ in u], RESAMPLES, SEED)


def comparison(a_rows: Rows, b_rows: Rows, name: str) -> dict[str, object] | None:
    """a minus b on one per-scenario series: mean difference, its 95% interval, and, for a
    correct/incorrect series, the exact McNemar p-value."""
    if not is_per_scenario(name):
        return None
    ua, ub = units(a_rows, name), units(b_rows, name)
    if ua is None or ub is None or [x[0] for x in ua] != [x[0] for x in ub]:
        return None
    va, vb = [v for _, _, v in ua], [v for _, _, v in ub]
    diff, lo, hi = paired_bootstrap_diff(va, vb, [g for _, g, _ in ua], RESAMPLES, SEED)
    p = mcnemar_exact([v == 1.0 for v in va], [v == 1.0 for v in vb]) if is_binary(name) else None
    return {"metric": name, "diff": diff, "ci": [lo, hi], "mcnemar_p": p}
