# ruff: noqa: F811  (the imported `repo` fixture is redefined by each test's parameter)
"""claims check runs the full record check once per cited record, not once per claim."""
import json
from pathlib import Path

import pytest

from contextlens.eval import claims
from tests.test_records import _run, _write_claims, repo  # noqa: F401, F811


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []
    real = claims.check_record

    def spy(rec, root, min_n=0):  # type: ignore[no-untyped-def]
        seen.append(rec["run_id"])
        return real(rec, root, min_n)

    monkeypatch.setattr(claims, "check_record", spy)
    return seen


def _claims_on(path: Path, n: int) -> list[dict]:
    rec = json.loads(path.read_text(encoding="utf-8"))
    value = rec["metrics"]["main"]["macro_f1"]
    return [{"id": f"c{i}", "run_id": rec["run_id"], "metric": "macro_f1", "value": value} for i in range(n)]


def test_a_record_cited_by_several_claims_is_checked_once(repo: Path, calls: list[str]) -> None:
    _write_claims(repo, _claims_on(_run(repo), 3))
    assert claims.check_claims(repo) == []
    assert len(calls) == 1


def test_a_failing_record_fails_every_claim_citing_it(repo: Path, calls: list[str]) -> None:
    _write_claims(repo, _claims_on(_run(repo, allow_dirty=True, models={"llm": "fake-model"}), 3))
    problems = claims.check_claims(repo)
    assert len(calls) == 1
    for cid in ("c0", "c1", "c2"):
        assert any(p.startswith(f"claim {cid}: run: ") for p in problems)
