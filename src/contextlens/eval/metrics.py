from collections.abc import Callable, Sequence

from contextlens.schemas import Card, ConsistencyLabel, Finding

LABELS: tuple[ConsistencyLabel, ...] = ("consistent", "update", "contradiction", "insufficient")


def recall_at_k(ranked_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float:
    return 1.0 if set(ranked_ids[:k]) & set(gold_ids) else 0.0


def first_gold_rank(ranked_ids: Sequence[str], gold_ids: Sequence[str]) -> int | None:
    for i, tid in enumerate(ranked_ids, start=1):
        if tid in gold_ids:
            return i
    return None


def mrr(ranks: Sequence[int | None]) -> float:
    """Mean reciprocal rank; a rank of None means no gold turn was retrieved."""
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks if r) / len(ranks)


def macro_f1(gold: Sequence[str], pred: Sequence[str]) -> float:
    f1s = []
    for label in LABELS:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return sum(f1s) / len(f1s)


def false_alert_rate(gold: Sequence[str], pred: Sequence[str]) -> float:
    """Share of non-contradiction cases predicted as contradiction."""
    negatives = [p for g, p in zip(gold, pred) if g != "contradiction"]
    if not negatives:
        return 0.0
    return sum(1 for p in negatives if p == "contradiction") / len(negatives)


def _finding_grounded(
    f: Finding,
    turn_seq: dict[str, int],
    as_of_seq: int,
    supports: Callable[[Finding], bool] | None,
) -> bool:
    if not f.evidence:
        return False
    for ev in f.evidence:
        seq = turn_seq.get(ev.turn_id)
        if seq is None or seq > as_of_seq:
            return False
    return supports(f) if supports else True


def grounded_rate(
    cards: Sequence[Card],
    turn_seq: dict[str, int],
    as_of_seq: int,
    supports: Callable[[Finding], bool] | None = None,
) -> float:
    """Share of findings whose cited turns all exist at or before as_of_seq (and are
    judged supporting, when a `supports` judge is given)."""
    findings = [f for c in cards for f in c.findings]
    if not findings:
        return 0.0
    ok = sum(_finding_grounded(f, turn_seq, as_of_seq, supports) for f in findings)
    return ok / len(findings)


def abstention_accuracy(
    abstained: Sequence[bool], gold_insufficient: Sequence[bool]
) -> dict[str, float]:
    """Both sides: abstains when gold is insufficient, answers when gold has an answer."""
    on_insuff = [a for a, g in zip(abstained, gold_insufficient) if g]
    on_answer = [not a for a, g in zip(abstained, gold_insufficient) if not g]
    return {
        "abstain_when_insufficient": sum(on_insuff) / len(on_insuff) if on_insuff else 0.0,
        "answer_when_answerable": sum(on_answer) / len(on_answer) if on_answer else 0.0,
    }


def speaker_accuracy(pred_speakers: Sequence[str], gold_speakers: Sequence[str]) -> float:
    if not gold_speakers:
        return 0.0
    return sum(p == g for p, g in zip(pred_speakers, gold_speakers)) / len(gold_speakers)
