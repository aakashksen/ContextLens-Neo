"""E1 runner on a tiny invented fixture with a stub embedder and a stub reranker model."""
import json
import re
import subprocess
from pathlib import Path

import pytest

from contextlens.data.qmsum_scenarios import build_qmsum_dev
from contextlens.eval import series
from contextlens.eval.check_run import check
from contextlens.eval.records import write_local_manifest, write_scenarios_manifest
from contextlens.experiments import e1_retrieval as e1
from contextlens.retrieval.hybrid import ScoredTurn
from contextlens.retrieval.rerank import Reranker
from contextlens.schemas import ButtonRequest, Conversation, Scenario, Turn

FIX = Path(__file__).parent / "fixtures" / "data"
TOPICS = ["budget", "garden", "server", "holiday", "recipe", "railway"]


class HashEmbedder:
    """Bag of words hashed into 64 buckets; the same text always gives the same vector."""

    model = "hash-bow-embedder"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * 64
            for w in re.findall(r"\w+", t.lower()):
                v[sum(map(ord, w)) % 64] += 1.0
            out.append(v)
        return out


class Overlap:
    def predict(self, pairs):
        return [len(set(re.findall(r"\w+", q.lower())) & set(re.findall(r"\w+", d.lower()))) for q, d in pairs]


def make_reranker() -> Reranker:
    return Reranker(model_name="test-overlap-reranker", loader=Overlap)


def _scenario(cid: str, topic: int, *, question: bool, source: str = "synthetic") -> Scenario:
    lines = [f"filler remark number {i} about nothing in particular" for i in range(7)]
    lines[2] = f"the {TOPICS[topic]} plan needs approval by friday"
    lines[6] = "closing words for the day" if question else f"remember the {TOPICS[topic]} plan"
    turns = [Turn(id=f"{cid}:t{i}", conversation_id=cid, seq=i, speaker="A" if i % 2 else "B", text=t,
                  start_ms=i * 1000, end_ms=i * 1000 + 900, source="dataset" if source == "qmsum" else "synthetic")
             for i, t in enumerate(lines)]
    return Scenario(
        id=f"{cid}-q", conversation=Conversation(id=cid, turns=turns, source=source, split="dev"),
        request=ButtonRequest(conversation_id=cid, as_of_seq=6,
                              question=f"what is the {TOPICS[topic]} plan" if question else None),
        category="recall" if question else "contradiction", gold_evidence_turn_ids=[f"{cid}:t2"],
        gold_label=None if question else "consistent", gold_answer=None,
        origin="dataset" if source == "qmsum" else "synthetic")


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=root, check=True,
                   capture_output=True)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(series, "RESAMPLES", 200)  # the run and the check use the same count
    d = tmp_path / "datasets/scenarios"
    d.mkdir(parents=True)
    ours = [_scenario(f"syn-{i}", i, question=i % 2 == 0) for i in range(6)]
    (d / "dev.jsonl").write_text("".join(s.model_dump_json() + "\n" for s in ours), encoding="utf-8")
    write_scenarios_manifest(tmp_path)
    (tmp_path / ".gitignore").write_text("datasets/raw/\nconfigs/\n", encoding="utf-8")
    local = [_scenario(f"m{i}", i, question=True, source="qmsum") for i in range(5)]
    write_local_manifest("qmsum", {"dev": local}, tmp_path)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_three_turn_windows_end_at_each_turn_and_map_back_to_turns() -> None:
    s = _scenario("c", 0, question=True)
    units, members = e1.chunks(s.conversation.turns, 3)
    assert [u.seq for u in units] == list(range(7))
    assert members[units[0].id] == ["c:t0"] and members[units[4].id] == ["c:t2", "c:t3", "c:t4"]
    one, none = e1.chunks(s.conversation.turns, 1)
    assert [u.id for u in one] == [t.id for t in s.conversation.turns] and none == {}


def test_a_scenario_without_a_question_searches_earlier_turns_for_its_last_turn() -> None:
    q, last = e1.query_for(_scenario("c", 1, question=True))
    assert q == "what is the garden plan" and last == 6
    q, last = e1.query_for(_scenario("c", 1, question=False))
    assert q == "remember the garden plan" and last == 5


def test_ranked_turns_come_from_windows_without_repeats_and_stop_at_k() -> None:
    s = _scenario("c", 0, question=True)
    units, members = e1.chunks(s.conversation.turns, 3)
    hits = [ScoredTurn(turn=units[i], score=1.0, parts={}) for i in (4, 5, 1)]
    assert e1.ranked_turns(hits, members, 5) == ["c:t2", "c:t3", "c:t4", "c:t5", "c:t0"]


def test_winner_rule_is_fixed_by_recall_then_mrr_then_order() -> None:
    def m(r5: float, mrr: float) -> dict:
        return {"recall@5": r5, "mrr": mrr}

    scores = {"a": {"dense_t1": m(0.5, 0.4), "bm25_t1": m(0.9, 0.1)},
              "b": {"dense_t1": m(0.9, 0.2), "bm25_t1": m(0.5, 0.5)}}
    assert e1.pick_winner(scores) == "dense_t1"  # recall@5 means tie at 0.7, mrr 0.3 vs 0.3, dense is listed first
    scores["b"]["bm25_t1"] = m(0.5, 0.9)
    assert e1.pick_winner(scores) == "bm25_t1"
    scores["b"]["dense_t1"] = m(0.95, 0.2)
    assert e1.pick_winner(scores) == "dense_t1"


def test_build_qmsum_dev_keeps_only_dev_conversations_with_gold(tmp_path: Path) -> None:
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "qmsum.json").write_text(json.dumps({"splits": {"dev": ["qmsum-val-0"], "test": ["other"]}}),
                                       encoding="utf-8")
    assert build_qmsum_dev(FIX, splits, tmp_path) == 1
    manifest = json.loads((tmp_path / "datasets/splits/qmsum-scenarios.json").read_text(encoding="utf-8"))
    assert manifest["source"] == "qmsum" and list(manifest["splits"]) == ["dev"]
    assert manifest["splits"]["dev"]["ids"] == ["qmsum-val-0-q1"] and manifest["splits"]["dev"]["n"] == 1


def test_the_run_writes_valid_records_and_a_config(repo: Path) -> None:
    result = e1.run_e1(root=repo, embedder=HashEmbedder(), reranker=make_reranker())
    assert result["winner"] in e1.SYSTEMS and result["complete"]
    assert check("E1", root=repo, min_n=5, split="dev") == []
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((repo / "results/runs").glob("E1-*.json"))]
    assert sorted(r["scenario_source"] for r in records) == ["qmsum", "scenarios"]
    for r in records:
        assert r["split"] == "dev" and sorted(r["predictions"]) == sorted(e1.SYSTEMS)
        assert r["models"]["reranker"] == "test-overlap-reranker" and r["models"]["embedder"] == "hash-bow-embedder"
        assert sorted(a["setting"] for a in r["attempts"]) == sorted(e1.SYSTEMS) and all(a["ok"] for a in r["attempts"])
        assert {c["metric"] for c in r["comparisons"]} == {"recall@5"} and r["intervals"][result["winner"]]
    qm = next(r for r in records if r["scenario_source"] == "qmsum")
    rows = [json.loads(x) for p in qm["predictions"].values() for x in (repo / p).read_text(encoding="utf-8").splitlines()]
    assert all(r["card"] is None and r["raw_output"] is None for r in rows)
    cfg = json.loads((repo / "configs/retrieval.json").read_text(encoding="utf-8"))
    assert cfg["winner"] == result["winner"] and cfg["split"] == "dev" and cfg["reranker_model"] == "test-overlap-reranker"
    assert set(cfg["scores"]) == {"scenarios", "qmsum"}


def test_a_reranker_that_fell_back_fails_the_run_and_saves_nothing(repo: Path) -> None:
    def broken():
        raise OSError("weights not found")

    with pytest.raises(e1.RerankerFallback, match="weights not found"):
        e1.run_e1(root=repo, embedder=HashEmbedder(),
                  reranker=Reranker(model_name="test-overlap-reranker", loader=broken))
    assert not (repo / "results/runs").exists() and not (repo / "configs/retrieval.json").exists()


def test_a_failed_system_is_listed_and_no_winner_is_chosen(repo: Path) -> None:
    class Dies:
        calls = 0

        def predict(self, pairs):  # the first call is the start-up proof; every later one fails
            Dies.calls += 1
            if Dies.calls > 1:
                raise RuntimeError("cuda out of memory")
            return [0.0] * len(pairs)

    result = e1.run_e1(root=repo, embedder=HashEmbedder(),
                       reranker=Reranker(model_name="test-overlap-reranker", loader=Dies))
    assert result["winner"] is None and not result["complete"]
    assert not (repo / "configs/retrieval.json").exists()
    rec = json.loads(next((repo / "results/runs").glob("E1-*.json")).read_text(encoding="utf-8"))
    failed = {a["setting"]: a for a in rec["attempts"] if not a["ok"]}
    assert sorted(failed) == ["hybrid_rerank_t1", "hybrid_rerank_w3"]
    assert "cuda out of memory" in failed["hybrid_rerank_t1"]["error"]
    assert "hybrid_rerank_t1" not in rec["predictions"] and rec["comparisons"] == []
    assert check("E1", root=repo, min_n=5) == []


def test_an_uncommitted_qmsum_manifest_stops_the_run(repo: Path) -> None:
    m = repo / "datasets/splits/qmsum-scenarios.json"
    m.write_text(m.read_text(encoding="utf-8").replace('"n": 5', '"n": 6'), encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest"):
        e1.run_e1(root=repo, embedder=HashEmbedder(), reranker=make_reranker())


def test_the_runner_never_reads_the_test_split() -> None:
    src = Path(e1.__file__).read_text(encoding="utf-8")
    assert '"test"' not in src and "'test'" not in src
