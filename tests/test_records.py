import json
import subprocess
from pathlib import Path

import pytest

from contextlens.eval import claims
from contextlens.eval.check_run import check, recompute_metrics
from contextlens.eval.records import is_dirty, split_hash, write_run, write_scenarios_manifest
from contextlens.replay.runner import Prediction
from contextlens.schemas import ButtonRequest, Card, Conversation, EvidenceRef, Finding, Scenario, Turn


def _scenario(sid: str, label: str | None) -> Scenario:
    turns = [
        Turn(id=f"{sid}-t{i}", conversation_id=sid, seq=i, speaker="A", text=f"w{i}",
             start_ms=i * 1000, end_ms=i * 1000 + 900, source="synthetic")
        for i in range(3)
    ]
    return Scenario(
        id=sid, conversation=Conversation(id=sid, turns=turns, source="test"),
        request=ButtonRequest(conversation_id=sid, as_of_seq=2), category="c",
        gold_evidence_turn_ids=[], gold_label=label, gold_answer=None, origin="synthetic",  # type: ignore[arg-type]
    )


def _pred(sid: str, label: str | None, ms: int = 10) -> Prediction:
    if label is None:
        card = Card(mode="recall", status="abstained", findings=[])
    else:
        f = Finding(agent="a", kind="consistency", statement="s", label=label, confidence=0.5,  # type: ignore[arg-type]
                    evidence=[EvidenceRef(turn_id=f"{sid}-t0", quote="w0", speaker="A", start_ms=0)])
        card = Card(mode="clarify", status="answered", findings=[f])
    return Prediction(scenario_id=sid, card=card, stage_latency_ms={"total": ms})


LABELS = ["contradiction", "consistent", "update", "insufficient"]
PRED_LABELS = ["contradiction", "contradiction", "update", None]


def _gold(split: str) -> list[tuple[str, str]]:
    return [(f"{split[0]}{i + 1}", g) for i, g in enumerate(LABELS)]


def _predl(split: str) -> list[tuple[str, str | None]]:
    return [(f"{split[0]}{i + 1}", g) for i, g in enumerate(PRED_LABELS)]


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=root, check=True,
                   capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    d = tmp_path / "datasets/scenarios"
    d.mkdir(parents=True)
    for split in ("dev", "test"):
        (d / f"{split}.jsonl").write_text(
            "".join(_scenario(i, g).model_dump_json() + "\n" for i, g in _gold(split)), encoding="utf-8")
    write_scenarios_manifest(tmp_path)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def _run(root: Path, split: str = "dev", *, models=None, attempts=None, preds=None, dhash=None,
         metrics=None, config=None, experiment: str = "E1", allow_dirty: bool = False,
         systems=None, **extra) -> Path:
    """One system called "main" unless `systems` (name -> predictions) is given."""
    preds = preds if preds is not None else [_pred(s, g) for s, g in _predl(split)]
    systems = systems if systems is not None else {"main": preds}
    scenarios = [_scenario(i, g) for i, g in _gold(split)]
    per_system = {n: (metrics if metrics is not None else recompute_metrics(scenarios, p))
                  for n, p in systems.items()}
    return write_run(
        experiment, split, config or {"labels_reviewed": True}, models or {"llm": "qwen-real"}, {"p": "v1"},
        per_system, extra.pop("intervals", {}),
        attempts if attempts is not None else [{"setting": "a", "ok": True}], systems,
        dhash or split_hash(split, root), root=root, allow_dirty=allow_dirty, **extra)


def _edit(path: Path, **changes) -> None:
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec.update(changes)
    path.write_text(json.dumps(rec), encoding="utf-8")


def test_good_run_passes(repo: Path) -> None:
    _run(repo)
    assert check("E1", root=repo) == []


def test_write_run_refuses_dirty_tree(repo: Path) -> None:
    (repo / "stray.txt").write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError):
        _run(repo)
    path = _run(repo, allow_dirty=True)
    assert json.loads(path.read_text(encoding="utf-8"))["dirty"] is True


def test_check_run_fails_when_dirty(repo: Path) -> None:
    (repo / "stray.txt").write_text("x", encoding="utf-8")
    _run(repo, allow_dirty=True)
    assert any("dirty" in p for p in check("E1", root=repo))


@pytest.mark.parametrize("name", ["fake-llm", "my_Stub"])
def test_check_run_fails_on_fake_model(repo: Path, name: str) -> None:
    _run(repo, models={"llm": name})
    assert any("fake or stub" in p for p in check("E1", root=repo))


def test_check_run_fails_on_dataset_hash_mismatch(repo: Path) -> None:
    _run(repo, dhash="0" * 64)
    assert any("dataset_hash" in p for p in check("E1", root=repo))


def test_check_run_fails_when_commit_not_ancestor(repo: Path) -> None:
    _edit(_run(repo), git_commit="1" * 40)
    assert any("ancestor" in p for p in check("E1", root=repo))


def test_check_run_fails_when_n_differs_from_split(repo: Path) -> None:
    _edit(_run(repo), n=3)
    assert any("differs from the split" in p for p in check("E1", root=repo))


def test_check_run_fails_below_min_n(repo: Path) -> None:
    _run(repo)
    assert any("min-n" in p for p in check("E1", root=repo, min_n=5))


def test_check_run_fails_when_attempts_empty(repo: Path) -> None:
    _run(repo, attempts=[])
    assert any("attempts" in p for p in check("E1", root=repo))


def test_check_run_fails_when_predictions_missing(repo: Path) -> None:
    path = _run(repo)
    (repo / json.loads(path.read_text(encoding="utf-8"))["predictions"]["main"]).unlink()
    assert any("predictions" in p for p in check("E1", root=repo))


def test_check_run_fails_when_metric_does_not_recompute(repo: Path) -> None:
    _run(repo, metrics={"macro_f1": 0.123})
    assert any("macro_f1" in p for p in check("E1", root=repo))


def test_check_run_fails_for_unrecomputable_metric(repo: Path) -> None:
    _run(repo, metrics={"mystery": 1.0})
    assert any("cannot be recomputed" in p for p in check("E1", root=repo))


def test_recompute_matches_saved_metrics(repo: Path) -> None:
    path = _run(repo)
    saved = json.loads(path.read_text(encoding="utf-8"))["metrics"]["main"]
    scenarios = [_scenario(i, g) for i, g in _gold("dev")]
    fresh = recompute_metrics(scenarios, [_pred(s, g) for s, g in _predl("dev")])
    assert saved == fresh
    assert fresh["false_alert_rate"] == pytest.approx(1 / 3)
    assert fresh["abstain_when_insufficient"] == 1.0


def test_second_test_record_needs_override_note(repo: Path) -> None:
    _run(repo, "test")
    assert check("E1", root=repo, split="test") == []
    _run(repo, "test")
    assert any("override_note" in p for p in check("E1", root=repo, split="test"))
    _run(repo, "test", config={"labels_reviewed": True, "override_note": "rerun after label fix"})
    assert check("E1", root=repo, split="test") != []  # the middle record still lacks a note


def _write_claims(root: Path, items: list[dict]) -> None:
    (root / "results").mkdir(exist_ok=True)
    (root / "results/claims.json").write_text(json.dumps({"claims": items}), encoding="utf-8")


def test_empty_claims_pass(repo: Path) -> None:
    _write_claims(repo, [])
    assert claims.check_claims(repo) == []


def test_claim_without_matching_run_fails(repo: Path) -> None:
    _write_claims(repo, [{"id": "c1", "run_id": "E1-nope", "metric": "macro_f1", "value": 0.5}])
    assert any("no run record" in p for p in claims.check_claims(repo))


def test_claim_value_must_equal_metric_in_run(repo: Path) -> None:
    path = _run(repo)
    rec = json.loads(path.read_text(encoding="utf-8"))
    good = {"id": "c1", "run_id": rec["run_id"], "metric": "macro_f1", "value": rec["metrics"]["main"]["macro_f1"]}
    _write_claims(repo, [good])
    assert claims.check_claims(repo) == []
    _write_claims(repo, [{**good, "value": good["value"] + 0.01}])
    assert any("value differs" in p for p in claims.check_claims(repo))


def test_check_run_fails_when_scenario_file_changed_after_manifest(repo: Path) -> None:
    _run(repo)
    f = repo / "datasets/scenarios/dev.jsonl"
    f.write_text(f.read_text(encoding="utf-8").replace("w1", "w9"), encoding="utf-8")
    rec_path = next((repo / "results/runs").glob("E1-*Z.json"))
    _edit(rec_path, dataset_hash=split_hash("dev", repo))
    assert any("manifest" in p for p in check("E1", root=repo))


def test_check_run_fails_on_duplicate_prediction_ids(repo: Path) -> None:
    labels = dict(_gold("dev"))
    preds = [_pred(i, labels[i]) for i in ["d1", "d1", "d3", "d3"]]
    scenarios = [_scenario(i, g) for i, g in _gold("dev")]
    _run(repo, preds=preds, metrics=recompute_metrics(scenarios, preds))
    assert any("prediction ids" in p for p in check("E1", root=repo))


def test_check_run_fails_on_unknown_prediction_id(repo: Path) -> None:
    preds = [_pred(s, g) for s, g in _predl("dev")]
    preds[3] = _pred("zz", None)
    _run(repo, preds=preds, metrics={"false_alert_rate": 0.0})
    assert any("prediction ids" in p for p in check("E1", root=repo))


def test_check_run_fails_when_metrics_empty(repo: Path) -> None:
    _run(repo, metrics={})
    assert any("metrics is empty" in p for p in check("E1", root=repo))


def test_check_run_fails_when_predictions_come_from_other_split(repo: Path) -> None:
    preds = [_pred(s, g) for s, g in _predl("test")]
    _run(repo, "dev", preds=preds, metrics={"false_alert_rate": 1 / 3})
    assert any("prediction ids" in p for p in check("E1", root=repo, split="dev"))


def test_check_run_fails_when_record_split_is_swapped(repo: Path) -> None:
    path = _run(repo, "dev")
    _edit(path, split="test")
    assert check("E1", root=repo, split="test") != []


def test_claims_file_without_claims_key_fails(repo: Path) -> None:
    (repo / "results").mkdir(exist_ok=True)
    (repo / "results/claims.json").write_text(json.dumps({"claimz": []}), encoding="utf-8")
    assert claims.check_claims(repo) != []
    (repo / "results/claims.json").write_text(json.dumps({"claims": {}}), encoding="utf-8")
    assert claims.check_claims(repo) != []


def test_new_run_files_do_not_make_tree_dirty_but_edits_and_junk_do(repo: Path) -> None:
    path = _run(repo)
    _run(repo)
    assert is_dirty(repo) is False
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "records")
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert is_dirty(repo) is True
    _git(repo, "checkout", "--", ".")
    (repo / "results/runs/junk.txt").write_text("x", encoding="utf-8")
    assert is_dirty(repo) is True


def test_check_splits_verifies_scenario_manifest(repo: Path) -> None:
    from contextlens.data.check_splits import check as check_splits

    splits_dir, raw = repo / "datasets/splits", repo / "datasets/raw"
    assert check_splits(splits_dir, raw) == []
    f = repo / "datasets/scenarios/dev.jsonl"
    f.write_text(f.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert any("differs from the manifest" in p for p in check_splits(splits_dir, raw))


def test_none_saved_for_defined_metric_fails_cleanly(repo: Path) -> None:
    _run(repo, metrics={"false_alert_rate": None})
    assert any("false_alert_rate" in p for p in check("E1", root=repo))


def test_claim_on_none_metric_fails(repo: Path) -> None:
    path = _run(repo)
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec["metrics"]["main"]["false_alert_rate"] = None
    path.write_text(json.dumps(rec), encoding="utf-8")
    _write_claims(repo, [{"id": "c1", "run_id": rec["run_id"], "metric": "false_alert_rate", "value": None}])
    assert any("c1" in p and "undefined" in p for p in claims.check_claims(repo))


def test_failed_atomic_write_keeps_old_file_and_leaves_no_temp(tmp_path: Path, monkeypatch) -> None:
    import os

    target = tmp_path / "x.json"
    target.write_text("old", encoding="utf-8")

    def boom(*a, **k):
        raise OSError("disk gone")

    monkeypatch.setattr(os, "replace", boom)
    from contextlens.eval.records import write_atomic

    with pytest.raises(OSError):
        write_atomic(target, "new")
    assert target.read_text(encoding="utf-8") == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


def test_interrupted_write_between_predictions_and_record_gives_no_record(repo: Path, monkeypatch) -> None:
    import contextlens.eval.records as rec

    real = rec.write_atomic

    def fail_on_record(path: Path, text: str) -> None:
        if path.suffix == ".json":
            raise OSError("crash")
        real(path, text)

    monkeypatch.setattr(rec, "write_atomic", fail_on_record)
    with pytest.raises(OSError):
        _run(repo)
    runs = repo / "results/runs"
    assert list(runs.glob("*Z.json")) == []
    assert len(list(runs.glob("*.predictions.jsonl"))) == 1
    assert [p.name for p in runs.glob(".*")] == []
    assert check("E1", root=repo) != []  # no record: the experiment has no result
    assert not is_dirty(repo)  # an orphan predictions file is an untracked run file, ignored like a record


def test_leftover_temp_is_not_a_record_and_not_dirty(repo: Path) -> None:
    _run(repo)
    runs = repo / "results/runs"
    (runs / ".E1-20250101T000000000000Z.json.123.tmp").write_text("{", encoding="utf-8")
    assert check("E1", root=repo) == []
    assert not is_dirty(repo)
