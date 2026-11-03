from collections.abc import Callable, Sequence

from contextlens.schemas import Card, ConsistencyLabel, Finding

LABELS: tuple[ConsistencyLabel, ...] = ("consistent", "update", "contradiction", "insufficient")


def _same_length(a: Sequence[object], b: Sequence[object]) -> None:
    if len(a) != len(b):
        raise ValueError(f"paired lists differ in length: {len(a)} vs {len(b)}")


def mean(values: Sequence[float]) -> float | None:
    """Plain mean; None when there is nothing to average."""
    return sum(values) / len(values) if values else None


def recall_at_k(ranked_ids: Sequence[str], gold_ids: Sequence[str], k: int) -> float | None:
    """None when the case has no gold evidence, because there is nothing to find."""
    if not gold_ids:
        return None
    return 1.0 if set(ranked_ids[:k]) & set(gold_ids) else 0.0


def first_gold_rank(ranked_ids: Sequence[str], gold_ids: Sequence[str]) -> int | None:
    for i, tid in enumerate(ranked_ids, start=1):
        if tid in gold_ids:
            return i
    return None


def mrr(ranks: Sequence[int | None]) -> float | None:
    """Mean reciprocal rank; a rank of None means no gold turn was retrieved.
    None when there are no cases."""
    if not ranks:
        return None
    return sum(1.0 / r for r in ranks if r) / len(ranks)


def _f1(gold: Sequence[str], pred: Sequence[str], label: str) -> float:
    tp = sum(1 for g, p in zip(gold, pred, strict=True) if g == label and p == label)
    fp = sum(1 for g, p in zip(gold, pred, strict=True) if g != label and p == label)
    fn = sum(1 for g, p in zip(gold, pred, strict=True) if g == label and p != label)
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def label_f1(gold: Sequence[str], pred: Sequence[str], label: str) -> float | None:
    """F1 of one label. None when no gold case has that label, so there is nothing to find."""
    _same_length(gold, pred)
    return _f1(gold, pred, label) if label in gold else None


def macro_f1(gold: Sequence[str], pred: Sequence[str]) -> float | None:
    """None when there are no cases."""
    _same_length(gold, pred)
    if not gold:
        return None
    return sum(_f1(gold, pred, label) for label in LABELS) / len(LABELS)


def false_alert_rate(gold: Sequence[str], pred: Sequence[str]) -> float | None:
    """Share of non-contradiction cases predicted as contradiction; None when there are none."""
    _same_length(gold, pred)
    negatives = [p for g, p in zip(gold, pred, strict=True) if g != "contradiction"]
    if not negatives:
        return None
    return sum(1 for p in negatives if p == "contradiction") / len(negatives)


def ece(correct: Sequence[bool], probabilities: Sequence[float], bins: int = 10) -> float | None:
    """Expected calibration error: equal-width probability bins, each bin's |accuracy - mean probability|
    weighted by its share of the cases. A probability of 1 falls in the last bin. None when there are no cases."""
    _same_length(correct, probabilities)
    if not correct:
        return None
    grouped: list[list[tuple[bool, float]]] = [[] for _ in range(bins)]
    for ok, p in zip(correct, probabilities, strict=True):
        grouped[min(int(p * bins), bins - 1)].append((ok, p))
    total = len(correct)
    return sum(len(g) / total * abs(sum(ok for ok, _ in g) / len(g) - sum(p for _, p in g) / len(g))
               for g in grouped if g)


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


def grounded_flags(
    cards: Sequence[Card],
    turn_seq: dict[str, int],
    as_of_seq: int,
    supports: Callable[[Finding], bool] | None = None,
) -> list[bool]:
    """One flag per finding, in order: cited turns all exist at or before as_of_seq (and are
    judged supporting, when a `supports` judge is given)."""
    return [_finding_grounded(f, turn_seq, as_of_seq, supports) for c in cards for f in c.findings]


def grounded_rate(
    cards: Sequence[Card],
    turn_seq: dict[str, int],
    as_of_seq: int,
    supports: Callable[[Finding], bool] | None = None,
) -> float | None:
    """Share of grounded findings. None when there are no findings."""
    flags = grounded_flags(cards, turn_seq, as_of_seq, supports)
    return sum(flags) / len(flags) if flags else None


def unsupported_claim_rate(flags_per_card: Sequence[Sequence[bool]]) -> float | None:
    """Per card, the share of its claims that are not grounded; averaged over cards that
    have claims. None when no card has a claim."""
    shares = [sum(not x for x in fl) / len(fl) for fl in flags_per_card if fl]
    return mean(shares)


def abstention_accuracy(
    abstained: Sequence[bool], gold_insufficient: Sequence[bool]
) -> dict[str, float | None]:
    """Both sides: abstains when gold is insufficient, answers when gold has an answer.
    A side with no cases is None."""
    _same_length(abstained, gold_insufficient)
    on_insuff = [a for a, g in zip(abstained, gold_insufficient, strict=True) if g]
    on_answer = [not a for a, g in zip(abstained, gold_insufficient, strict=True) if not g]
    return {
        "abstain_when_insufficient": sum(on_insuff) / len(on_insuff) if on_insuff else None,
        "answer_when_answerable": sum(on_answer) / len(on_answer) if on_answer else None,
    }


def speaker_accuracy(pred_speakers: Sequence[str], gold_speakers: Sequence[str]) -> float | None:
    """None when there are no claims with a gold speaker."""
    _same_length(pred_speakers, gold_speakers)
    if not gold_speakers:
        return None
    return sum(p == g for p, g in zip(pred_speakers, gold_speakers, strict=True)) / len(gold_speakers)
