"""Cross-encoder scoring for `team_rag`: how well each candidate turn answers the question, from the cleaned turn text.

The model loads on first use and is shared by every press. If it cannot load, `score` raises and the caller falls back to
the hybrid order with the score gate switched off."""

import logging
import threading
from collections.abc import Callable

from contextlens.agents.rag_text import clip_for_prompt, words
from contextlens.device import loaded_device, resolve_device
from contextlens.schemas import Turn

log = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
NEEDS_MB = 1500  # a budget, not a measurement
MAX_CHARS = 2400  # of a cleaned turn handed to the model, clipped around the sentences that match the query
MAX_TOKENS = 512  # what the model reads of each (query, turn) pair


class CrossScorer:
    def __init__(self, model_name: str = DEFAULT_MODEL, loader: Callable[[], object] | None = None,
                 max_chars: int = MAX_CHARS, max_tokens: int = MAX_TOKENS) -> None:
        self.model_name, self.max_chars, self.max_tokens = model_name, max_chars, max_tokens
        self._loader = loader or self._load
        self._model: object | None = None
        self._lock = threading.Lock()
        self.device = "not loaded"

    def _load(self):
        from sentence_transformers import CrossEncoder

        device = resolve_device(NEEDS_MB)
        model = CrossEncoder(self.model_name, device=device, max_length=self.max_tokens)
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
        focus = words(query)
        pairs = [(query, clip_for_prompt(t.text, focus, self.max_chars)) for t in turns]
        return [float(s) for s in self._model.predict(pairs, batch_size=25, show_progress_bar=False)]
