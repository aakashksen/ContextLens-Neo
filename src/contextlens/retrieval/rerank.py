import logging
from collections.abc import Callable

from contextlens.device import loaded_device, resolve_device
from contextlens.retrieval.hybrid import ScoredTurn

log = logging.getLogger(__name__)

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
NEEDS_MB = 500  # a budget, not a measurement


class NoReranker:
    def rerank(self, query: str, candidates: list[ScoredTurn], k: int) -> list[ScoredTurn]:
        return candidates[:k]


class Reranker:
    """Cross-encoder reranker. The model loads on first use; if it cannot load,
    the input order is kept and `fallback_reason` says why."""

    def __init__(self, model_name: str = DEFAULT_MODEL, loader: Callable[[], object] | None = None) -> None:
        self.model_name = model_name
        self._loader = loader or self._load_cross_encoder
        self._model: object | None = None
        self.fallback_reason: str | None = None
        self.device = "not loaded"  # where the model actually runs, once loaded

    def _load_cross_encoder(self):
        from sentence_transformers import CrossEncoder

        return CrossEncoder(self.model_name, device=resolve_device(NEEDS_MB))

    def rerank(self, query: str, candidates: list[ScoredTurn], k: int) -> list[ScoredTurn]:
        if not candidates:
            return []
        if self._model is None and self.fallback_reason is None:
            try:
                self._model = self._loader()
                self.device = loaded_device(self._model)
            except Exception as e:
                self.fallback_reason = str(e)
                log.warning("reranker %s failed to load, keeping input order: %s", self.model_name, e)
        if self._model is None:
            return candidates[:k]
        scores = self._model.predict([(query, c.turn.text) for c in candidates])
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda p: -float(p[1]))
        return [c.model_copy(update={"score": float(s)}) for c, s in ranked[:k]]
