import logging
import subprocess
import sys

from contextlens.retrieval.hybrid import ScoredTurn
from contextlens.retrieval.rerank import NoReranker, Reranker
from contextlens.schemas import Turn


class StubModel:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def predict(self, pairs):
        return [self.scores[text] for _, text in pairs]


def cand(i: int, text: str) -> ScoredTurn:
    turn = Turn(id=f"t{i}", conversation_id="c", seq=i, speaker="A", text=text, start_ms=i, end_ms=i + 1, source="synthetic")
    return ScoredTurn(turn=turn, score=1.0 / (i + 1), parts={"dense": 1.0})


def make(scores: dict[str, float]) -> Reranker:
    return Reranker(loader=lambda: StubModel(scores))


def test_order_follows_scores():
    cands = [cand(0, "a"), cand(1, "b"), cand(2, "c")]
    out = make({"a": 0.1, "b": 0.9, "c": 0.5}).rerank("q", cands, 3)
    assert [s.turn.text for s in out] == ["b", "c", "a"]
    assert [s.score for s in out] == [0.9, 0.5, 0.1]


def test_k_limits_output():
    cands = [cand(0, "a"), cand(1, "b"), cand(2, "c")]
    out = make({"a": 0.1, "b": 0.9, "c": 0.5}).rerank("q", cands, 2)
    assert [s.turn.text for s in out] == ["b", "c"]
    assert [s.turn.text for s in NoReranker().rerank("q", cands, 2)] == ["a", "b"]


def test_empty_candidates():
    def boom():
        raise AssertionError("model must not load for no candidates")

    assert Reranker(loader=boom).rerank("q", [], 5) == []
    assert NoReranker().rerank("q", [], 5) == []


def test_load_failure_falls_back_and_says_so(caplog):
    def broken():
        raise OSError("no weights")

    r = Reranker(loader=broken)
    cands = [cand(0, "a"), cand(1, "b"), cand(2, "c")]
    with caplog.at_level(logging.WARNING):
        out = r.rerank("q", cands, 2)
    assert out == cands[:2]
    assert r.fallback_reason == "no weights"
    assert "no weights" in caplog.text


def test_import_does_not_load_torch():
    code = "import sys, contextlens.retrieval.rerank; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
