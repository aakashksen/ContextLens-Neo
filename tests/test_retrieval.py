import os

import pytest

from contextlens.memory.store import InMemoryTranscriptStore
from contextlens.retrieval.hybrid import Retriever
from contextlens.schemas import Turn

TEXTS = [
    "we should book the flights for june",
    "my sister Anneliese lives in Oslo",
    "the budget is about four thousand dollars",
    "I prefer the window seat",
    "let us meet at the station",
]


class StubEmbedder:
    """Each text gets a one-hot-ish vector by position; the query matches the topic word."""

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 6
            if "budget" in t or "money" in t:
                v[0] = 1.0
            elif "flights" in t:
                v[1] = 1.0
            elif "seat" in t:
                v[2] = 1.0
            elif "station" in t:
                v[3] = 1.0
            elif "sister" in t:
                v[4] = 1.0
            else:
                v[5] = 1.0
            out.append(v)
        return out


def make(speakers=None, confs=None):
    store = InMemoryTranscriptStore()
    for i, text in enumerate(TEXTS):
        store.add(Turn(
            id=f"t{i}", conversation_id="c", seq=i,
            speaker=(speakers or ["A", "B"] * 3)[i],
            speaker_confidence=(confs or [1.0] * 6)[i],
            text=text, start_ms=i * 1000, end_ms=i * 1000 + 900, source="synthetic",
        ))  # fmt: skip
    return store, Retriever(store, StubEmbedder())


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_search_never_returns_future_turns(mode):
    _, r = make()
    for as_of in range(len(TEXTS)):
        hits = r.search("c", "budget flights seat station Anneliese", as_of, k=10, mode=mode)
        assert all(h.turn.seq <= as_of for h in hits)
    assert r.search("c", "budget", 1, mode=mode, k=10) is not None
    assert {h.turn.id for h in r.search("c", "budget", 1, k=10, mode=mode)} <= {"t0", "t1"}


def test_exact_name_found_by_bm25_when_dense_misses():
    _, r = make()
    dense = r.search("c", "Anneliese", 4, k=1, mode="dense")
    assert dense[0].turn.id != "t1"  # stub embedder knows nothing about names
    assert r.search("c", "Anneliese", 4, k=1, mode="bm25")[0].turn.id == "t1"
    hybrid = r.search("c", "Anneliese", 4, k=5, mode="hybrid")
    assert "t1" in [h.turn.id for h in hybrid]
    assert "bm25" in hybrid[[h.turn.id for h in hybrid].index("t1")].parts


def test_speaker_filter():
    _, r = make()
    hits = r.search("c", "the", 4, k=10, speaker="B")
    assert hits and {h.turn.speaker for h in hits} == {"B"}


def test_low_confidence_speaker_is_reported_as_unknown():
    _, r = make(confs=[1.0, 0.49, 1.0, 0.5, 1.0, 1.0])
    by_id = {h.turn.id: h.turn for h in r.search("c", "sister the", 4, k=10, mode="bm25")}
    assert by_id["t1"].speaker == "unknown"
    assert by_id["t3"].speaker == "B"
    assert r.search("c", "seat", 4, k=10, speaker="B", mode="bm25")


def test_delete_removes_text_and_vectors():
    store, r = make()
    for mode in ("dense", "bm25", "hybrid"):
        assert r.search("c", "budget", 4, mode=mode)
    r.delete("c")
    assert store.turns("c", 99) == []
    assert not r._cache
    for mode in ("dense", "bm25", "hybrid"):
        assert r.search("c", "budget", 4, mode=mode) == []


def test_readding_same_id_with_new_text_does_not_reuse_stale_vector():
    store, r = make()
    assert r.search("c", "budget", 4, k=1, mode="dense")[0].turn.id == "t2"
    store.delete("c")
    for i, text in enumerate(["x", "y", "the station is far", "z", "w"]):
        store.add(Turn(id=f"t{i}", conversation_id="c", seq=i, speaker="A", text=text,
                       start_ms=i, end_ms=i + 1, source="synthetic"))  # fmt: skip
    hits = r.search("c", "station", 4, k=1, mode="dense")
    assert hits[0].turn.id == "t2" and hits[0].parts["dense"] > 0
    assert r.search("c", "budget", 4, k=1, mode="dense")[0].turn.id != "t2"


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_turn_added_after_vectors_cached_is_not_returned_past_as_of(mode):
    store, r = make()
    r.search("c", "budget", 4, mode=mode)
    store.add(Turn(id="t9", conversation_id="c", seq=9, speaker="A", text="budget budget",
                   start_ms=9, end_ms=10, source="synthetic"))  # fmt: skip
    assert "t9" not in {h.turn.id for h in r.search("c", "budget", 4, k=10, mode=mode)}
    assert "t9" in {h.turn.id for h in r.search("c", "budget", 9, k=10, mode=mode)}


def test_filtering_by_real_name_of_low_confidence_speaker_returns_nothing():
    _, r = make(speakers=["A", "B", "A", "B", "A", "B"], confs=[1.0, 0.2, 1.0, 1.0, 1.0, 1.0])
    hits = r.search("c", "sister", 4, k=10, speaker="B", mode="bm25")
    assert "t1" not in {h.turn.id for h in hits}
    assert {h.turn.id for h in r.search("c", "sister", 4, k=10, speaker="unknown", mode="bm25")} == {"t1"}


@pytest.mark.real
def test_postgres_store_roundtrip_and_delete():
    from contextlens.memory.store import PostgresTranscriptStore

    dsn = os.environ.get("TEST_DATABASE_URL", "postgresql://contextlens:contextlens@localhost:5432/contextlens")

    class Emb(StubEmbedder):
        def embed(self, texts):
            return [v + [0.0] * 762 for v in super().embed(texts)]

    store = PostgresTranscriptStore(dsn, Emb())
    store.delete("pgtest")
    for i, text in enumerate(TEXTS):
        store.add(Turn(id=f"pg{i}", conversation_id="pgtest", seq=i, speaker="A", text=text,
                       start_ms=i, end_ms=i + 1, source="synthetic"))  # fmt: skip
    try:
        assert [t.seq for t in store.turns("pgtest", 2)] == [0, 1, 2]
        r = Retriever(store, Emb())
        assert r.search("pgtest", "Anneliese", 4, k=1, mode="bm25")[0].turn.id == "pg1"
        assert all(h.turn.seq <= 2 for h in r.search("pgtest", "budget", 2, k=10))
    finally:
        store.delete("pgtest")
    assert store.turns("pgtest", 99) == []
    assert store.vectors("pgtest", [f"pg{i}" for i in range(5)]) == {}
