"""Models take the GPU when it exists and has room; a fallback to the CPU is warned about, never silent."""

import sys
import types

import pytest

from contextlens import device as dv


def _fake_torch(monkeypatch, available: bool, free_mb: int = 0):
    cuda = types.SimpleNamespace(is_available=lambda: available, mem_get_info=lambda d=None: (free_mb * 2**20, 0))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda, device=lambda s: s))


def test_the_setting_forces_the_cpu_without_looking_at_torch(monkeypatch):
    monkeypatch.setenv(dv.ENV, "cpu")
    monkeypatch.setitem(sys.modules, "torch", None)  # an import would fail
    said = []
    assert dv.resolve_device(500, said.append) == "cpu" and said == []


def test_cuda_is_the_default_when_it_is_there_and_has_room(monkeypatch):
    monkeypatch.delenv(dv.ENV)
    _fake_torch(monkeypatch, True, free_mb=4000)
    said = []
    assert dv.resolve_device(500, said.append) == "cuda" and said == []


def test_no_cuda_falls_back_to_cpu_with_a_warning(monkeypatch):
    monkeypatch.delenv(dv.ENV)
    _fake_torch(monkeypatch, False)
    said = []
    assert dv.resolve_device(500, said.append) == "cpu"
    assert len(said) == 1 and "WARNING" in said[0] and "not available" in said[0]


def test_too_little_free_memory_falls_back_to_cpu_with_a_warning(monkeypatch):
    monkeypatch.delenv(dv.ENV)
    _fake_torch(monkeypatch, True, free_mb=100)
    said = []
    assert dv.resolve_device(500, said.append) == "cpu"
    assert len(said) == 1 and "100 MB free" in said[0] and "500 MB" in said[0]


class _Scores(list):
    def tolist(self):
        return list(self)


def test_the_reranker_and_zero_shot_models_are_loaded_on_the_resolved_device(monkeypatch):
    st = pytest.importorskip("sentence_transformers")
    seen = []

    class Fake:
        device = "cpu"

        def __init__(self, name, device=None):
            seen.append(device)

        def predict(self, pairs, **kw):
            return _Scores([[0.9, 0.05, 0.05] if kw else 0.5 for _ in pairs])

    monkeypatch.setattr(st, "CrossEncoder", Fake)
    from contextlens.consistency.baselines import ZeroShotChecker
    from contextlens.retrieval.hybrid import ScoredTurn
    from contextlens.retrieval.rerank import Reranker
    from contextlens.schemas import Turn

    turn = Turn(id="t0", conversation_id="c", seq=0, speaker="a", text="hi", start_ms=0, end_ms=1, source="synthetic")
    r = Reranker()
    assert r.device == "not loaded"
    r.rerank("q", [ScoredTurn(turn=turn, score=1.0, parts={})], 1)
    z = ZeroShotChecker()
    z.check(turn, [turn])
    assert seen == ["cpu", "cpu"] and r.device == "cpu" and z.device == "cpu"
