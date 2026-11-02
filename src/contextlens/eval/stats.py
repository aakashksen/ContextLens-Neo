import random
from collections import defaultdict
from collections.abc import Sequence
from math import comb


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile, p in [0, 100]."""
    if not values:
        raise ValueError("percentile of empty sequence")
    s = sorted(values)
    pos = (len(s) - 1) * p / 100
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _by_group(values: Sequence[float], groups: Sequence[str]) -> list[list[float]]:
    if len(values) != len(groups):
        raise ValueError("values and groups differ in length")
    d: dict[str, list[float]] = defaultdict(list)
    for v, g in zip(values, groups):
        d[g].append(v)
    return list(d.values())


def _resample_means(buckets: list[list[float]], n: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    k = len(buckets)
    sums = [sum(b) for b in buckets]
    counts = [len(b) for b in buckets]
    out = []
    for _ in range(n):
        idx = [rng.randrange(k) for _ in range(k)]
        out.append(sum(sums[i] for i in idx) / sum(counts[i] for i in idx))
    return out


def bootstrap_ci(
    values: Sequence[float], groups: Sequence[str], n: int = 10000, seed: int = 13
) -> tuple[float, float]:
    """95% interval of the mean, resampling whole conversations (groups)."""
    if not values:
        raise ValueError("bootstrap_ci got no values: there is nothing to resample")
    means = _resample_means(_by_group(values, groups), n, seed)
    return percentile(means, 2.5), percentile(means, 97.5)


def paired_bootstrap_diff(
    a: Sequence[float], b: Sequence[float], groups: Sequence[str], n: int = 10000, seed: int = 13
) -> tuple[float, float, float]:
    """(mean of a-b, lo, hi) with a 95% interval, resampling whole conversations."""
    if len(a) != len(b):
        raise ValueError("a and b differ in length")
    diffs = [x - y for x, y in zip(a, b)]
    means = _resample_means(_by_group(diffs, groups), n, seed)
    return sum(diffs) / len(diffs), percentile(means, 2.5), percentile(means, 97.5)


def mcnemar_exact(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> float:
    """Two-sided exact McNemar p-value on discordant pairs."""
    if len(a_correct) != len(b_correct):
        raise ValueError("a_correct and b_correct differ in length")
    b = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    c = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    m = b + c
    if m == 0:
        return 1.0
    tail = sum(comb(m, i) for i in range(min(b, c) + 1)) / 2**m
    return min(1.0, 2 * tail)
