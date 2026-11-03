"""One check every trainer shares: a non-finite loss stops the run at once instead of being averaged into a record."""
import math
from collections.abc import Iterable
from typing import Any


class NonFiniteLoss(FloatingPointError):
    pass


def require_finite(value: float, where: str) -> float:
    if not math.isfinite(value):
        raise NonFiniteLoss(f"non-finite loss ({value}) at {where}; training stopped before the update")
    return value


def non_finite_epochs(history: Iterable[dict[str, Any]]) -> list[int]:
    """Epochs (1-based position when a row has no `epoch`) whose `train_loss` is missing, not a number, NaN or infinite."""
    bad = []
    for i, row in enumerate(history, 1):
        x = row.get("train_loss")
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            bad.append(row.get("epoch", i))
    return bad
