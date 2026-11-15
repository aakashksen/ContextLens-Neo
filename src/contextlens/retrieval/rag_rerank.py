"""Cross-encoder scoring for `team_rag`: how well each candidate turn answers the question, from the cleaned turn text.

The model loads on first use and is shared by every press. If it cannot load, `score` raises and the caller falls back to
the hybrid order with the score gate switched off."""

import logging
import threading
from collections.abc import Callable

from contextlens.agents.rag_text import clean
from contextlens.device import loaded_device, resolve_device
from contextlens.schemas import Turn

log = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
NEEDS_MB = 1500  # a budget, not a measurement
MAX_CHARS = 2400  # of a cleaned turn handed to the model (it reads 512 tokens at most)


class CrossScorer:
    def __init__(self, model_name: str = DEFAULT_MODEL, loader: Callable[[], object] | None = None) -> None:
        self.model_name = model_name
        self._loader = loader or self._load
        self._model: object | None = None
        self._lock = threading.Lock()
        self.device = "not loaded"

    def _load(self):
        from sentence_transformers import CrossEncoder

        device = resolve_device(NEEDS_MB)
        model = CrossEncoder(self.model_name, device=device, max_length=512)
        if device != "cpu":
            model.model.half()
        return model

    def score(self, query: str, turns: list[Turn]) -> list[float]:
        if not turns:
            return []
        with self._lock:
            if self._model is None:
                self._model = self._loader()
                self.device = loaded_device(self._model)
        pairs = [(query, clean(t.text)[:MAX_CHARS]) for t in turns]
        return [float(s) for s in self._model.predict(pairs, batch_size=25, show_progress_bar=False)]
