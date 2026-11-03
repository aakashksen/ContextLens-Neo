# ruff: noqa: F811
"""A claim on a repeated test-split record fails unless the record has an override note."""
import json
from pathlib import Path

from contextlens.eval import claims
from tests.test_records import _run, _write_claims, repo  # noqa: F401


def _claim(run: dict) -> dict:
    return {"id": "c1", "run_id": run["run_id"], "metric": "macro_f1", "value": run["metrics"]["main"]["macro_f1"]}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_claim_on_a_second_test_record_fails_without_override_note(repo: Path) -> None:
    _run(repo, "test", experiment="E8")
    second = _load(_run(repo, "test", experiment="E8"))
    _write_claims(repo, [_claim(second)])
    assert any("override_note" in p for p in claims.check_claims(repo))


def test_claim_on_a_second_test_record_passes_with_override_note(repo: Path) -> None:
    _run(repo, "test", experiment="E8")
    second = _load(_run(repo, "test", experiment="E8",
                        config={"labels_reviewed": True, "override_note": "rerun after label fix"}))
    _write_claims(repo, [_claim(second)])
    assert claims.check_claims(repo) == []


def test_claim_on_the_only_test_record_passes(repo: Path) -> None:
    only = _load(_run(repo, "test", experiment="E8"))
    _write_claims(repo, [_claim(only)])
    assert claims.check_claims(repo) == []
