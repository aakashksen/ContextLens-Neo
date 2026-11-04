import re
from collections.abc import Callable, Sequence
from functools import lru_cache

import numpy as np

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


_WORD = re.compile(r"[a-z0-9']+")


@lru_cache(maxsize=4096)
def word_errors(ref: str, hyp: str) -> tuple[int, int]:
    """(substitutions + insertions + deletions, words in ref) after lowercasing and dropping punctuation."""
    r, h = _WORD.findall(ref.lower()), _WORD.findall(hyp.lower())
    if not r or not h:
        return len(r) or len(h), len(r)
    ids = {w: i for i, w in enumerate(set(r) | set(h))}
    hv = np.array([ids[w] for w in h])
    cols = np.arange(len(h) + 1)
    prev = cols.copy()
    for i, w in enumerate(r, start=1):
        cand = np.empty_like(prev)
        cand[0] = i
        cand[1:] = np.minimum(prev[1:] + 1, prev[:-1] + (hv != ids[w]))
        prev = np.minimum.accumulate(cand - cols) + cols  # the left neighbour's +1 insertion
    return int(prev[-1]), len(r)


def word_error_rate(refs: Sequence[str], hyps: Sequence[str]) -> float | None:
    """Word errors over reference words, summed over all pairs. None when the references hold no words."""
    _same_length(refs, hyps)
    pairs = [word_errors(r, h) for r, h in zip(refs, hyps, strict=True)]
    total = sum(n for _, n in pairs)
    return sum(e for e, _ in pairs) / total if total else None


Span = tuple[int, int, str]  # start_ms, end_ms, speaker
FRAME_MS = 10


def _frames(spans: Sequence[Span], skip: str | None = None) -> dict[str, np.ndarray]:
    end = max((e for _, e, _ in spans), default=0) // FRAME_MS + 1
    out: dict[str, np.ndarray] = {}
    for start, stop, who in spans:
        if who != skip and stop > start:
            out.setdefault(who, np.zeros(end, dtype=bool))[start // FRAME_MS : -(-stop // FRAME_MS)] = True
    return out


def _best_match(overlap: list[list[int]]) -> int:
    """Largest total overlap when each row (reference speaker) takes a distinct column (hypothesis speaker)."""
    best: dict[int, int] = {0: 0}
    for row in overlap:
        nxt = dict(best)
        for used, score in best.items():
            for j, v in enumerate(row):
                if not used >> j & 1 and score + v > nxt.get(used | 1 << j, -1):
                    nxt[used | 1 << j] = score + v
        best = nxt
    return max(best.values())


def diarization_error(ref: Sequence[Span], hyp: Sequence[Span]) -> tuple[int, int]:
    """(error frames, reference speaker frames) at 10 ms. Per frame the error is max(reference speakers,
    hypothesis speakers) minus those matching under the best one-to-one mapping of hypothesis names to reference
    names, so missed speech, false alarms and confusion all count, overlap included, with no collar. Hypothesis
    spans named "unknown" are no speech at all."""
    r, h = _frames(ref), _frames(hyp, skip="unknown")
    n = max((len(a) for a in (*r.values(), *h.values())), default=0)
    pad = lambda a: np.pad(a, (0, n - len(a)))  # noqa: E731
    rm = np.array([pad(a) for a in r.values()]).reshape(len(r), n)
    hm = np.array([pad(a) for a in h.values()]).reshape(len(h), n)
    ref_total = int(rm.sum())
    overlap = (rm.astype(np.int64) @ hm.T.astype(np.int64)).tolist() if len(h) else [[] for _ in r]
    correct = _best_match(overlap) if len(h) else 0
    worst = int(np.maximum(rm.sum(axis=0), hm.sum(axis=0)).sum()) if n else 0
    return worst - correct, ref_total


def diarization_error_rate(refs: Sequence[Sequence[Span]], hyps: Sequence[Sequence[Span]]) -> float | None:
    """Error frames over reference speaker frames, summed over all pairs. None without reference speech."""
    _same_length(refs, hyps)
    pairs = [diarization_error(r, h) for r, h in zip(refs, hyps, strict=True)]
    total = sum(n for _, n in pairs)
    return sum(e for e, _ in pairs) / total if total else None


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> float | None:
    """Agreement of two yes/no judges beyond chance. None when there are no pairs or chance agreement is
    certain (both judges answered the same one way throughout), where kappa is undefined."""
    _same_length(a, b)
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)
