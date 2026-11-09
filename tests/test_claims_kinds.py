"""Claim kinds and label rules added in T9.3: benchmark, guard, the latency exemption and the AMI speech route."""
import json
import subprocess
from pathlib import Path

import pytest

from contextlens.eval import claims

WORDS = "on published labels"


def _rec(**extra) -> dict:
    rec = {"experiment": "E8", "scenario_source": "scenarios", "labels_reviewed": True,
           "config": {"benchmarks": {
               "qmsum": {"cases": 3, "metrics": {"mrr": 0.5}},
               "decode": {"cases": 3, "metrics": {"trained": {"f1": 0.9}, "tfidf": {"f1": 0.7}}}}}}
    rec.update(extra)
    return rec


def test_benchmark_reads_the_named_metric_exactly() -> None:
    c = {"kind": "benchmark", "benchmark": "qmsum", "metric": "mrr", "value": 0.5}
    assert claims._value_problem(c, _rec()) == (None, "mrr")
    assert claims._value_problem({**c, "value": 0.51}, _rec())[0] == "value differs from the benchmark in the run"


def test_benchmark_system_picks_a_nested_table() -> None:
    c = {"kind": "benchmark", "benchmark": "decode", "system": "tfidf", "metric": "f1", "value": 0.7}
    assert claims._value_problem(c, _rec())[0] is None
    assert claims._value_problem({**c, "system": "nope"}, _rec())[0] is not None


@pytest.mark.parametrize("patch", [{"benchmark": "missing"}, {"metric": "recall"}])
def test_benchmark_missing_pieces_fail(patch: dict) -> None:
    c = {"kind": "benchmark", "benchmark": "qmsum", "metric": "mrr", "value": 0.5, **patch}
    assert claims._value_problem(c, _rec())[0] is not None


@pytest.mark.parametrize("metric", ["latency_p50_ms", "latency_p95_ms[stage=memory_ms]", "latency_p50_ms[category=x]",
                                    "tokens_per_card"])
def test_latency_and_tokens_need_no_reviewed_labels(metric: str) -> None:
    assert claims._label_problems("c", {"metric": metric}, _rec(scenario_source="ami", labels_reviewed=False)) == []


def test_accuracy_and_judged_stay_blocked_on_unreviewed_labels() -> None:
    rec = _rec(scenario_source="ami", labels_reviewed=False)
    for metric in ("label_accuracy", "grounded_rate", "mrr"):
        assert claims._label_problems("c", {"metric": metric}, rec) != []


def test_wer_on_ami_needs_the_published_words() -> None:
    rec = _rec(experiment="E7", scenario_source="ami", labels_reviewed=False)
    assert claims._label_problems("c", {"metric": "wer", "text": f"WER 0.4 {WORDS}"}, rec) == []
    assert claims._label_problems("c", {"metric": "der", "text": f"DER {WORDS}"}, rec) == []
    assert claims._label_problems("c", {"metric": "wer", "text": "WER 0.4"}, rec) != []


def test_the_wer_route_is_only_for_e7_on_ami() -> None:
    text = f"WER {WORDS}"
    assert claims._label_problems("c", {"metric": "wer", "text": text}, _rec(experiment="E5", scenario_source="ami", labels_reviewed=False)) != []
    assert claims._label_problems("c", {"metric": "wer", "text": text}, _rec(scenario_source="scenarios", labels_reviewed=False)) != []
    rec = _rec(experiment="E7", scenario_source="ami", labels_reviewed=False)
    assert claims._label_problems("c", {"metric": "label_accuracy", "text": text}, rec) != []


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def guard_repo(tmp_path: Path) -> tuple[Path, str]:
    _git(tmp_path, "init", "-q")
    (tmp_path / "results" / "guard-heldout").mkdir(parents=True)
    (tmp_path / "results" / "runs").mkdir(parents=True)
    (tmp_path / "results" / "guard-heldout" / "abc123.json").write_text(
        json.dumps({"result": {"injections": 60, "injections_caught": 14}}), encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "guard")
    return tmp_path, _git(tmp_path, "rev-parse", "HEAD")


def _guard(root: Path, commit: str, **extra) -> list[str]:
    c = {"id": "g", "kind": "guard", "hash": "abc123", "metric": "injections_caught", "value": 14, "commit": commit, **extra}
    (root / "results" / "claims.json").write_text(json.dumps({"claims": [c]}), encoding="utf-8")
    return claims.check_claims(root)


def test_a_committed_guard_claim_passes(guard_repo: tuple[Path, str]) -> None:
    root, commit = guard_repo
    assert _guard(root, commit) == []
    assert _guard(root, commit[:12]) == []


def test_a_wrong_guard_value_or_field_fails(guard_repo: tuple[Path, str]) -> None:
    root, commit = guard_repo
    assert _guard(root, commit, value=15) != []
    assert _guard(root, commit, metric="nope") != []
    assert _guard(root, commit, hash="zzz") != []


def test_a_guard_claim_must_name_the_adding_commit(guard_repo: tuple[Path, str]) -> None:
    root, commit = guard_repo
    assert _guard(root, "0" * 40) != []
    assert _guard(root, "") != []


def test_a_changed_or_untracked_guard_file_fails(guard_repo: tuple[Path, str]) -> None:
    root, commit = guard_repo
    path = root / "results" / "guard-heldout" / "abc123.json"
    path.write_text(json.dumps({"result": {"injections_caught": 14, "x": 1}}), encoding="utf-8")
    assert _guard(root, commit) != []
    _git(root, "checkout", "--", str(path))
    (root / "results" / "guard-heldout" / "new.json").write_text(json.dumps({"result": {"injections_caught": 14}}), encoding="utf-8")
    assert _guard(root, commit, hash="new") != []
