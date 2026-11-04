from collections.abc import Callable

from contextlens.data.loaders import Pair
from contextlens.device import loaded_device, resolve_device
from contextlens.schemas import ConsistencyLabel, Turn

ZERO_SHOT_MODEL = "cross-encoder/nli-deberta-v3-xsmall"  # Apache-2.0, see docs/datasets.md
ZERO_SHOT_NEEDS_MB = 600  # a budget, not a measurement
_FOUR_LABELS = ("consistent", "update", "contradiction", "insufficient")


def _context(earlier: list[str]) -> str:
    return " ".join(earlier[-6:])


class TfidfChecker:
    """TF-IDF of the statement and of the recent earlier turns, then logistic regression over the labels seen in fit."""

    def __init__(self) -> None:
        self._vec = None
        self._clf = None

    def _features(self, statements: list[str], contexts: list[str], fit: bool = False):
        from scipy.sparse import hstack
        from sklearn.feature_extraction.text import TfidfVectorizer

        if fit:
            self._vec = (TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                         TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True))
            self._vec[0].fit(statements)
            self._vec[1].fit(contexts)
        assert self._vec is not None
        return hstack([self._vec[0].transform(statements), self._vec[1].transform(contexts)]).tocsr()

    def fit(self, pairs: list[Pair]) -> "TfidfChecker":
        from sklearn.linear_model import LogisticRegression

        bad = {p.label for p in pairs} - set(_FOUR_LABELS)
        if bad:
            raise ValueError(f"unmapped labels {sorted(bad)}; build the pairs with build_pairs")
        x = self._features([p.statement for p in pairs], [_context(p.earlier) for p in pairs], fit=True)
        self._clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(x, [p.label for p in pairs])
        return self

    def check(self, statement: Turn, earlier: list[Turn]) -> tuple[ConsistencyLabel, float]:
        if self._clf is None:
            raise RuntimeError("TfidfChecker is not fitted")
        x = self._features([statement.text], [_context([t.text for t in earlier])])
        proba = self._clf.predict_proba(x)[0]
        best = int(proba.argmax())
        return self._clf.classes_[best], float(proba[best])


class ZeroShotChecker:
    """Off-the-shelf NLI model: the statement against each earlier turn. It can say only contradiction or consistent.

    `scorer` maps (premise, hypothesis) pairs to [contradiction, entailment, neutral] probabilities; by default the
    NLI model loads on first use.
    """

    def __init__(self, model_name: str = ZERO_SHOT_MODEL, scorer: Callable[[list[tuple[str, str]]], list[list[float]]] | None = None,
                 threshold: float = 0.5) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self._scorer = scorer or self._load_scorer
        self.device = "not loaded"  # where the model actually runs, once loaded

    def _load_scorer(self, pairs: list[tuple[str, str]]) -> list[list[float]]:
        from sentence_transformers import CrossEncoder

        if not hasattr(self, "_model"):
            self._model = CrossEncoder(self.model_name, device=resolve_device(ZERO_SHOT_NEEDS_MB))
            self.device = loaded_device(self._model)
        return self._model.predict(pairs, apply_softmax=True).tolist()

    def check(self, statement: Turn, earlier: list[Turn]) -> tuple[ConsistencyLabel, float]:
        if not earlier:  # nothing to compare against
            return "insufficient", 1.0
        probs = self._scorer([(t.text, statement.text) for t in earlier])
        worst = max(p[0] for p in probs)
        if worst >= self.threshold:
            return "contradiction", float(worst)
        return "consistent", float(1.0 - worst)
