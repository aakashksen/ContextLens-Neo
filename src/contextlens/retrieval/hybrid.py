import math
import re
from collections import Counter
from typing import Literal

from pydantic import BaseModel

from contextlens.memory.store import TranscriptStore
from contextlens.schemas import Turn

RRF_K = 60
LOW_CONFIDENCE = 0.5
Mode = Literal["dense", "bm25", "hybrid"]


class ScoredTurn(BaseModel):
    turn: Turn
    score: float
    parts: dict[str, float]


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _cosine(a: list[float], b: list[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _bm25(query: str, docs: list[Turn], k1: float = 1.5, b: float = 0.75) -> list[float]:
    toks = [_tokens(d.text) for d in docs]
    avg = (sum(len(t) for t in toks) / len(toks)) or 1.0
    df = Counter(w for t in toks for w in set(t))
    scores = []
    for t in toks:
        tf = Counter(t)
        s = 0.0
        for w in set(_tokens(query)):
            if w in tf:
                idf = math.log(1 + (len(toks) - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(t) / avg))
        scores.append(s)
    return scores


class Retriever:
    def __init__(self, store: TranscriptStore, embedder) -> None:
        self.store, self.embedder = store, embedder
        self._cache: dict[tuple[str, str], list[float]] = {}

    def delete(self, conv_id: str) -> None:
        for t in self.store.turns(conv_id, 2**62):
            self._cache.pop((t.id, t.text), None)
        self.store.delete(conv_id)

    def _vectors(self, conv_id: str, turns: list[Turn]) -> list[list[float]]:
        stored = getattr(self.store, "vectors", None)
        if stored is not None:
            have = stored(conv_id, [t.id for t in turns])
        else:
            have = {t.id: self._cache[t.id, t.text] for t in turns if (t.id, t.text) in self._cache}
            missing = [t for t in turns if t.id not in have]
            if missing:
                for t, v in zip(missing, self.embedder.embed([t.text for t in missing])):
                    self._cache[t.id, t.text] = have[t.id] = v
        return [have[t.id] for t in turns]

    def search(
        self,
        conv_id: str,
        query: str,
        as_of_seq: int,
        k: int = 8,
        speaker: str | None = None,
        since_ms: int | None = None,
        mode: Mode = "hybrid",
    ) -> list[ScoredTurn]:
        turns = []
        for t in self.store.turns(conv_id, as_of_seq):
            if t.speaker_confidence < LOW_CONFIDENCE:
                t = t.model_copy(update={"speaker": "unknown"})
            if speaker is not None and t.speaker != speaker:
                continue
            if since_ms is not None and t.start_ms < since_ms:
                continue
            turns.append(t)
        if not turns:
            return []
        rankings: dict[str, list[int]] = {}
        if mode in ("dense", "hybrid"):
            qv = self.embedder.embed([query])[0]
            sims = [_cosine(qv, v) for v in self._vectors(conv_id, turns)]
            rankings["dense"] = sorted(range(len(turns)), key=lambda i: -sims[i])
        if mode in ("bm25", "hybrid"):
            sc = _bm25(query, turns)
            rankings["bm25"] = [i for i in sorted(range(len(turns)), key=lambda i: -sc[i]) if sc[i] > 0]
        fused: dict[int, dict[str, float]] = {}
        for name, order in rankings.items():
            for rank, i in enumerate(order):
                fused.setdefault(i, {})[name] = 1.0 / (RRF_K + rank + 1)
        out = [ScoredTurn(turn=turns[i], score=sum(p.values()), parts=p) for i, p in fused.items()]
        out.sort(key=lambda s: (-s.score, s.turn.seq))
        return out[:k]
