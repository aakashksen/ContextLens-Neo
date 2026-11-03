# ruff: noqa: F811
"""A record names the scenario source it ran on, and check_run verifies that against the source's manifest.
A source whose text cannot be committed (QMSum) is scored from a local, git-ignored file pinned by a
committed manifest of hash, ids and count."""
import json
from pathlib import Path

from contextlens.eval.check_run import check, recompute_metrics
from contextlens.eval.records import split_hash, write_local_manifest, write_run
from contextlens.replay.runner import Prediction
from contextlens.schemas import ButtonRequest, Conversation, Scenario, Turn
from tests.test_records import _edit, _git, _run, repo  # noqa: F401  (repo is a fixture)


def _local(sid: str) -> Scenario:
    turns = [Turn(id=f"{sid}:t{i}", conversation_id=sid, seq=i, speaker="A", text=f"invented line {sid} {i}",
                  start_ms=i * 1000, end_ms=i * 1000 + 900, source="dataset") for i in range(4)]
    return Scenario(id=sid, conversation=Conversation(id=sid, turns=turns, source="qmsum", split="dev"),
                    request=ButtonRequest(conversation_id=sid, as_of_seq=3, question="invented?"), category="qmsum-specific",
                    gold_evidence_turn_ids=[f"{sid}:t1"], gold_label=None, gold_answer=None, origin="dataset")


def _ranked(sid: str, ids: list[str]) -> Prediction:
    return Prediction(scenario_id=sid, ranked_turn_ids=ids, stage_latency_ms={"total": 5})


SIDS = ["q1", "q2", "q3"]


def _local_repo(repo: Path, commit: bool = True) -> Path:
    (repo / ".gitignore").write_text("datasets/raw/\n", encoding="utf-8")
    write_local_manifest("qmsum", {"dev": [_local(s) for s in SIDS]}, repo)
    if commit:
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "qmsum manifest")
    return repo


def _local_run(root: Path, *, source: str = "qmsum", dhash: str | None = None, rows=None, **extra) -> Path:
    rows = rows if rows is not None else [_ranked(s, [f"{s}:t1", f"{s}:t0"]) for s in SIDS]
    scs = [_local(s) for s in SIDS]
    return write_run("E1", "dev", {"labels_reviewed": True}, {"embedder": "real-embed"}, {"p": "v1"},
                     {"main": recompute_metrics(scs, rows)}, {}, [{"setting": "a", "ok": True}], {"main": rows},
                     dhash or split_hash("dev", root, "qmsum"), root=root, scenario_source=source, **extra)


def test_a_run_on_our_scenarios_names_its_source(repo: Path) -> None:
    path = _run(repo)
    assert json.loads(path.read_text(encoding="utf-8"))["scenario_source"] == "scenarios"
    assert check("E1", root=repo) == []


def test_a_record_without_a_source_fails(repo: Path) -> None:
    path = _run(repo)
    rec = json.loads(path.read_text(encoding="utf-8"))
    del rec["scenario_source"]
    path.write_text(json.dumps(rec), encoding="utf-8")
    assert any("missing field scenario_source" in p for p in check("E1", root=repo))


def test_an_unknown_source_fails(repo: Path) -> None:
    _edit(_run(repo), scenario_source="mystery")
    assert any("scenario source" in p for p in check("E1", root=repo))


def test_a_local_source_scored_from_its_manifest_passes(repo: Path) -> None:
    _local_run(_local_repo(repo))
    assert check("E1", root=repo) == []


def test_a_run_on_our_scenarios_cannot_be_labeled_qmsum(repo: Path) -> None:
    _local_repo(repo)
    _edit(_run(repo), scenario_source="qmsum")
    assert any("dataset_hash" in p or "manifest" in p for p in check("E1", root=repo))


def test_a_run_on_qmsum_cannot_be_labeled_as_our_scenarios(repo: Path) -> None:
    _local_repo(repo)
    _edit(_local_run(repo), scenario_source="scenarios")
    assert check("E1", root=repo)


def test_a_local_file_that_changed_fails(repo: Path) -> None:
    _local_repo(repo)
    _local_run(repo)
    f = repo / "datasets/raw/scenarios/qmsum/dev.jsonl"
    f.write_text(f.read_text(encoding="utf-8").replace("invented", "changed"), encoding="utf-8")
    assert any("no longer matches" in p for p in check("E1", root=repo))


def test_a_missing_local_file_fails(repo: Path) -> None:
    _local_repo(repo)
    _local_run(repo)
    (repo / "datasets/raw/scenarios/qmsum/dev.jsonl").unlink()
    assert any("no scenario file" in p for p in check("E1", root=repo))


def test_a_manifest_committed_after_the_run_fails(repo: Path) -> None:
    _local_repo(repo, commit=False)
    _local_run(repo, allow_dirty=True)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "late manifest")
    assert any("not committed before the run" in p for p in check("E1", root=repo))


def test_a_manifest_edited_after_the_run_fails(repo: Path) -> None:
    _local_repo(repo)
    _local_run(repo)
    m = repo / "datasets/splits/qmsum-scenarios.json"
    manifest = json.loads(m.read_text(encoding="utf-8"))
    manifest["splits"]["dev"]["ids"].reverse()
    m.write_text(json.dumps(manifest), encoding="utf-8")
    assert any("not committed before the run" in p for p in check("E1", root=repo))


def test_manifest_ids_must_match_the_local_file(repo: Path) -> None:
    _local_repo(repo, commit=False)
    m = repo / "datasets/splits/qmsum-scenarios.json"
    manifest = json.loads(m.read_text(encoding="utf-8"))
    manifest["splits"]["dev"]["ids"] = ["q1", "q2", "other"]
    m.write_text(json.dumps(manifest), encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "wrong ids")
    _local_run(repo)
    assert any("ids" in p for p in check("E1", root=repo))


def test_rows_of_a_local_source_hold_ids_only(repo: Path) -> None:
    _local_repo(repo)
    rows = [_ranked(s, [f"{s}:t1"]).model_copy(update={"raw_output": "model text"}) for s in SIDS]
    _local_run(repo, rows=rows)
    assert any("ids and metrics only" in p for p in check("E1", root=repo))


def test_retrieval_record_with_undefined_judged_metrics_needs_no_judge(repo: Path) -> None:
    path = _local_run(_local_repo(repo))
    saved = json.loads(path.read_text(encoding="utf-8"))["metrics"]["main"]
    assert saved["grounded_rate"] is None and saved["unsupported_claim_rate"] is None
    assert check("E1", root=repo) == []


def test_a_defined_judged_metric_still_needs_a_judge(repo: Path) -> None:
    path = _run(repo)
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec["metrics"]["main"]["grounded_rate"] = 0.5
    path.write_text(json.dumps(rec), encoding="utf-8")
    assert any("config.judge" in p for p in check("E1", root=repo))


def test_a_rerank_system_must_record_its_reranker_model(repo: Path) -> None:
    _local_repo(repo)
    rows = [_ranked(s, [f"{s}:t1"]) for s in SIDS]
    scs = [_local(s) for s in SIDS]
    kw = dict(models={"embedder": "real-embed"})

    def go(models: dict[str, str]) -> Path:
        return write_run("E1", "dev", {"labels_reviewed": True}, models, {"p": "v1"},
                         {"hybrid_rerank_t1": recompute_metrics(scs, rows)}, {}, [{"setting": "a", "ok": True}],
                         {"hybrid_rerank_t1": rows}, split_hash("dev", repo, "qmsum"), root=repo,
                         scenario_source="qmsum")

    go(kw["models"])
    assert any("reranker" in p for p in check("E1", root=repo))
    for f in (repo / "results/runs").glob("*"):
        f.unlink()
    go({"embedder": "real-embed", "reranker": "cross-encoder/ms-marco-MiniLM-L6-v2"})
    assert check("E1", root=repo) == []
