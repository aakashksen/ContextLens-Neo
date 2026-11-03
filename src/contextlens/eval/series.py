"""Metric values recomputed from saved prediction rows, and the intervals and paired comparisons
built on them. An interval or comparison resamples whole conversations and recomputes the actual
metric on each resample with the same functions as `base_metrics`; it is None when the metric is
undefined on the data or on any resample, because dropping those resamples would bias the interval
toward the defined cases."""
import random
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence

from contextlens.eval import metrics as M
from contextlens.eval.stats import mcnemar_exact, percentile
from contextlens.replay.runner import Prediction
from contextlens.schemas import Card, Scenario

Metrics = dict[str, float | None]
Rows = Sequence[tuple[Scenario, Prediction]]
SEED = 13
RESAMPLES = 10_000
Unit = tuple[str, float]  # scenario id, 1.0 when correct

BINARY = ("label_accuracy", "schema_valid_rate", "recall@")
F1_LABELS = tuple(f"f1[label={x}]" for x in M.LABELS)  # per-label F1; undefined (None) without gold cases
JUDGED = ("grounded_rate", "unsupported_claim_rate")
GROUP_SUFFIX = re.compile(r"^(.*)\[(category|origin)=(.*)\]$")


def label_of(card: Card) -> str:
    for f in card.findings:
        if f.label:
            return f.label
    return "insufficient"


def predicted_label(p: Prediction) -> str | None:
    """The consistency label a row gives: from its card, else its bare label; None for a retrieval row."""
    if p.card is not None:
        return label_of(p.card)
    return p.label


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
        judged = {id(f): j for f, j in zip(p.card.findings, p.claim_supported or [], strict=False)}
        turn_seq = {t.id: t.seq for t in s.conversation.turns}
        out.append(M.grounded_flags([p.card], turn_seq, s.request.as_of_seq, lambda f, judged=judged: judged[id(f)]))
    return out


def base_name(name: str) -> str:
    return name.split("[", 1)[0]


def _is(name: str, group: tuple[str, ...]) -> bool:
    return any(name == g or (g.endswith("@") and re.fullmatch(re.escape(g) + r"\d+", name)) for g in group)


def is_binary(name: str) -> bool:
    """A series of correct/incorrect outcomes, which McNemar's test may be applied to."""
    return _is(name, BINARY)


def units(rows: Rows, name: str) -> list[Unit] | None:
    """Correct/incorrect outcome per scenario for a binary metric, or None when not recomputable."""
    if name == "label_accuracy":
        if any(predicted_label(p) is None for _, p in rows):
            return None
        out = [(s.id, float(predicted_label(p) == s.gold_label)) for s, p in rows if s.gold_label]
    elif name == "schema_valid_rate":
        if any(p.raw_output is None for _, p in rows):
            return None
        out = [(s.id, float(schema_valid(p.raw_output or ""))) for s, p in rows]
    elif re.fullmatch(r"recall@\d+", name):
        need = [(s, p) for s, p in rows if s.gold_evidence_turn_ids]
        if any(p.ranked_turn_ids is None for _, p in need):
            return None
        out = [(s.id, M.recall_at_k(p.ranked_turn_ids or [], s.gold_evidence_turn_ids,
                                    int(name.split("@")[1])) or 0.0) for s, p in need]
    else:
        return None
    return out or None


def metric_value(rows: Rows, name: str) -> float | None:
    """One metric on these rows, by the same functions `recompute_metrics` uses. `name[category=x]`
    and `name[origin=y]` restrict to those scenarios first. None when undefined or unknown."""
    found = GROUP_SUFFIX.match(name)
    if found:
        inner, key, value = found.groups()
        return metric_value([(s, p) for s, p in rows if getattr(s, key) == value], inner)
    if name == "label_accuracy":
        u = units(rows, name)
        return M.mean([v for _, v in u]) if u else None
    ks = [int(k) for k in re.findall(r"^recall@(\d+)$", name)]
    return base_metrics(rows, ks, only=name).get(name)


def _resamples(rows: Rows, n: int, seed: int) -> Iterator[list[int]]:
    """Whole-conversation resamples as row positions: draw as many conversations as there are,
    with replacement."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, (s, _) in enumerate(rows):
        groups[s.conversation.id].append(i)
    buckets = list(groups.values())
    rng = random.Random(seed)
    for _ in range(n):
        yield [i for b in [rng.randrange(len(buckets)) for _ in buckets] for i in buckets[b]]


def interval(rows: Rows, name: str) -> tuple[float, float] | None:
    """95% bootstrap interval of the metric (seed 13, 10,000 resamples of whole conversations).
    None when the metric is undefined on the rows or on any resample."""
    if not rows or metric_value(rows, name) is None:
        return None
    vals = []
    for pos in _resamples(rows, RESAMPLES, SEED):
        v = metric_value([rows[i] for i in pos], name)
        if v is None:
            return None
        vals.append(v)
    return percentile(vals, 2.5), percentile(vals, 97.5)


def comparison(a_rows: Rows, b_rows: Rows, name: str) -> dict[str, object] | None:
    """a minus b on one metric: the difference on all rows, a 95% interval from resampling the same
    conversations for both systems, and, for a correct/incorrect metric, the exact McNemar p-value.
    None when the systems cover different cases or the metric is undefined for either, on all rows
    or on any resample."""
    if [s.id for s, _ in a_rows] != [s.id for s, _ in b_rows] or not a_rows:
        return None
    va, vb = metric_value(a_rows, name), metric_value(b_rows, name)
    if va is None or vb is None:
        return None
    diffs = []
    for pos in _resamples(a_rows, RESAMPLES, SEED):
        x = metric_value([a_rows[i] for i in pos], name)
        y = metric_value([b_rows[i] for i in pos], name)
        if x is None or y is None:
            return None
        diffs.append(x - y)
    p = None
    if is_binary(name):
        ua, ub = units(a_rows, name), units(b_rows, name)
        if ua is None or ub is None or [i for i, _ in ua] != [i for i, _ in ub]:
            return None
        p = mcnemar_exact([v == 1.0 for _, v in ua], [v == 1.0 for _, v in ub])
    return {"metric": name, "diff": va - vb, "ci": [percentile(diffs, 2.5), percentile(diffs, 97.5)],
            "mcnemar_p": p}


def _complete(present: list[bool]) -> bool | None:
    """True when every row has the field, None when no row has it, False when only some do."""
    if all(present):
        return True
    return None if not any(present) else False


Want = Callable[..., bool]


def _label_metrics(rows: Rows, ks: tuple[int, ...], want: Want) -> Metrics:
    out: Metrics = {}
    if want("macro_f1", "false_alert_rate", *F1_LABELS):
        labelled = [(s.gold_label, q) for s, p in rows if s.gold_label and (q := predicted_label(p))]
        gold, pred = [g for g, _ in labelled], [q for _, q in labelled]
        out["macro_f1"] = M.macro_f1(gold, pred)
        out["false_alert_rate"] = M.false_alert_rate(gold, pred)
        for label, name in zip(M.LABELS, F1_LABELS, strict=True):
            out[name] = M.label_f1(gold, pred, label)
    if want("label_accuracy"):
        u = units(rows, "label_accuracy")
        out["label_accuracy"] = M.mean([v for _, v in u]) if u else None
    if want("ece"):
        scored = [(s, p, predicted_label(p)) for s, p in rows if s.gold_label and predicted_label(p)]
        if scored and all(p.probability is not None for _, p, _ in scored):  # else the source field is missing
            out["ece"] = M.ece([s.gold_label == q for s, _, q in scored], [p.probability for _, p, _ in scored])  # type: ignore[misc]
    return out


def _card_metrics(rows: Rows, ks: tuple[int, ...], want: Want) -> Metrics:
    out: Metrics = {}
    carded = [(s, p, p.card) for s, p in rows if p.card is not None]
    if want("abstain_when_insufficient", "answer_when_answerable"):
        out.update(M.abstention_accuracy(
            [c.status == "abstained" for _, _, c in carded],
            [s.gold_label == "insufficient" for s, _, _ in carded],
        ))
    if want("tokens_per_card"):
        out["tokens_per_card"] = M.mean([float(c.tokens) for _, _, c in carded])
    if want("cost_usd_per_card"):
        out["cost_usd_per_card"] = M.mean([c.cost_usd for _, _, c in carded])
    if want("speaker_accuracy"):
        pred_sp: list[str] = []
        gold_sp: list[str] = []
        for s, _, c in carded:
            speaker_of = {t.id: t.speaker for t in s.conversation.turns}
            for ev in (e for f in c.findings for e in f.evidence):
                if ev.turn_id in speaker_of:
                    pred_sp.append(ev.speaker)
                    gold_sp.append(speaker_of[ev.turn_id])
        out["speaker_accuracy"] = M.speaker_accuracy(pred_sp, gold_sp)
    return out


def _judged_metrics(rows: Rows, ks: tuple[int, ...], want: Want) -> Metrics:
    """Whether a finding supports its claim is a saved rubric judgement; without one per finding the metrics are omitted."""
    flags = grounded_per_card(rows) if want(*JUDGED) else None
    if flags is None:
        return {}
    flat = [x for fl in flags for x in fl]
    return {"grounded_rate": sum(flat) / len(flat) if flat else None,
            "unsupported_claim_rate": M.unsupported_claim_rate(flags)}


def _raw_and_retrieval_metrics(rows: Rows, ks: tuple[int, ...], want: Want) -> Metrics:
    out: Metrics = {}
    raws = _complete([p.raw_output is not None for _, p in rows]) if want("schema_valid_rate") else False
    if raws is not False:
        out["schema_valid_rate"] = (
            M.mean([float(schema_valid(p.raw_output)) for _, p in rows if p.raw_output]) if raws else None)
    need = [(s, p) for s, p in rows if s.gold_evidence_turn_ids]
    has_ranks = _complete([p.ranked_turn_ids is not None for _, p in need]) \
        if want("mrr", *(f"recall@{k}" for k in ks)) else False
    if has_ranks is not False:
        ranked = [(s, p.ranked_turn_ids or []) for s, p in need] if has_ranks else []
        out["mrr"] = M.mrr([M.first_gold_rank(r, s.gold_evidence_turn_ids) for s, r in ranked])
        for k in ks:
            out[f"recall@{k}"] = M.mean(
                [v for s, r in ranked if (v := M.recall_at_k(r, s.gold_evidence_turn_ids, k)) is not None])
    return out


def _latency_metrics(rows: Rows, ks: tuple[int, ...], want: Want) -> Metrics:
    out: Metrics = {}
    stage_names = {name for _, p in rows for name in p.stage_latency_ms}
    for name in sorted(stage_names | {"total"}):
        suffix = "" if name == "total" else f"[stage={name}]"
        if not want(f"latency_p50_ms{suffix}", f"latency_p95_ms{suffix}"):
            continue
        if name in stage_names and not all(name in p.stage_latency_ms for _, p in rows):
            continue
        vals = [float(p.stage_latency_ms[name]) for _, p in rows if name in p.stage_latency_ms]
        out[f"latency_p50_ms{suffix}"] = percentile(vals, 50) if vals else None
        out[f"latency_p95_ms{suffix}"] = percentile(vals, 95) if vals else None
    return out


def base_metrics(rows: list[tuple[Scenario, Prediction]], ks: Iterable[int], only: str | None = None) -> Metrics:
    """A metric whose source field is on only some rows is left out, so it cannot be saved.
    With `only`, just the metric of that name is computed."""
    def want(*names: str) -> bool:
        return only is None or only in names

    ks = tuple(ks)
    out: Metrics = {}
    for section in (_label_metrics, _card_metrics, _judged_metrics, _raw_and_retrieval_metrics, _latency_metrics):
        out.update(section(rows, ks, want))
    return out
