# ruff: noqa: F811  (the imported `repo` fixture is redefined by each test's parameter)
"""check_run remembers passes, and only trusts one when nothing it depended on has changed."""
import json
from pathlib import Path

import pytest

from contextlens.eval import check_cache, check_run
from contextlens.eval.check_cache import PassCache, code_hash, code_modules
from contextlens.eval.check_run import check
from tests.test_records import _edit, _run, repo  # noqa: F401, F811

CODE = "code-v1"


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []
    real = check_run.check_record

    def spy(rec, root, min_n=0):  # type: ignore[no-untyped-def]
        seen.append(rec["run_id"])
        return real(rec, root, min_n)

    monkeypatch.setattr(check_run, "check_record", spy)
    return seen


@pytest.fixture(autouse=True)
def _ignore_cache(repo: Path) -> None:
    """The real repo git-ignores the cache file; do the same in the test repo so it never makes the tree dirty."""
    info = repo / ".git/info"
    info.mkdir(exist_ok=True)
    (info / "exclude").write_text(check_cache.CACHE_REL.as_posix() + "\n", encoding="utf-8")


def _cached(repo: Path, code: str = CODE, **kw) -> list[str]:  # type: ignore[no-untyped-def]
    cache = PassCache(repo, code)
    problems = check("E1", root=repo, cache=cache, **kw)
    cache.save()
    return problems


def test_unchanged_record_is_skipped(repo: Path, calls: list[str]) -> None:
    _run(repo)
    assert _cached(repo) == [] and len(calls) == 1
    assert _cached(repo) == [] and len(calls) == 1


def test_new_record_is_checked_and_old_one_is_not(repo: Path, calls: list[str]) -> None:
    _run(repo)
    _cached(repo)
    _run(repo)
    assert _cached(repo) == [] and len(calls) == 2


def test_changed_record_is_rechecked(repo: Path, calls: list[str]) -> None:
    path = _run(repo)
    _cached(repo)
    _edit(path, finished_at="2030-01-01T00:00:00+00:00")
    assert _cached(repo) == [] and len(calls) == 2


def test_changed_predictions_are_rechecked_and_fail(repo: Path, calls: list[str]) -> None:
    path = _run(repo)
    _cached(repo)
    pred = repo / json.loads(path.read_text(encoding="utf-8"))["predictions"]["main"]
    pred.write_text(pred.read_text(encoding="utf-8").replace("contradiction", "update"), encoding="utf-8")
    assert len(calls) == 1 and _cached(repo) != [] and len(calls) == 2


def test_changed_checker_rechecks_everything(repo: Path, calls: list[str]) -> None:
    _run(repo)
    _cached(repo, "code-v1")
    assert _cached(repo, "code-v2") == [] and len(calls) == 2


def test_code_hash_covers_check_run_and_what_it_imports() -> None:
    names = set(code_modules())
    assert {"contextlens.eval.check_run", "contextlens.eval.series", "contextlens.eval.metrics",
            "contextlens.eval.records", "contextlens.train_guard", "contextlens.data.scenarios"} <= names
    assert code_hash() == code_hash()


def test_code_hash_changes_with_a_module_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    before = code_hash()
    src = tmp_path / "series.py"
    src.write_text("x = 1\n", encoding="utf-8")
    modules = {**code_modules(), "contextlens.eval.series": src}
    monkeypatch.setattr(check_cache, "code_modules", lambda: modules)
    assert code_hash() != before


def test_failed_record_is_never_cached(repo: Path, calls: list[str]) -> None:
    _run(repo, allow_dirty=True, models={"llm": "fake-model"})
    assert _cached(repo) != [] and _cached(repo) != [] and len(calls) == 2
    assert PassCache(repo, CODE).passed == {}


def test_min_n_is_part_of_the_key(repo: Path, calls: list[str]) -> None:
    _run(repo)
    assert _cached(repo) == []
    assert any("--min-n" in p for p in _cached(repo, min_n=100)) and len(calls) == 2


def test_split_is_part_of_the_key(repo: Path, calls: list[str]) -> None:
    _run(repo)
    assert _cached(repo, split="dev") == [] and _cached(repo, split="test") != []
    assert len(calls) == 1  # no test record exists: the second call found no records, so nothing was checked


def test_ancestry_is_still_checked_on_a_cache_hit(repo: Path) -> None:
    path = _run(repo)
    _cached(repo)
    _edit(path, git_commit="0" * 40)  # also changes the key, so patch the cache to say it passed
    cache = PassCache(repo, CODE)
    cache.add(check_cache.record_key(path, json.loads(path.read_text(encoding="utf-8")), repo, 0, None, CODE), "x")
    assert any("ancestor" in p for p in check("E1", root=repo, cache=cache))


def test_all_ignores_remembered_passes(repo: Path, calls: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check_run, "ROOT", repo)
    real_check, real_gone = check_run.check, check_run.superseded_by  # their root defaults were bound at import
    monkeypatch.setattr(check_run, "check", lambda *a, **k: real_check(*a, root=repo, **k))
    monkeypatch.setattr(check_run, "superseded_by", lambda e, root=repo: real_gone(e, repo))
    _run(repo)
    assert check_run.main(["E1"]) == 0 and len(calls) == 1
    assert check_run.main(["E1"]) == 0 and len(calls) == 1
    assert check_run.main(["E1", "--all"]) == 0 and len(calls) == 2


def test_cache_is_written_atomically_and_ignored_by_git(repo: Path) -> None:
    _run(repo)
    _cached(repo)
    assert (repo / check_cache.CACHE_REL).is_file()
    assert not list((repo / "results").glob("*.tmp")) and not list((repo / "results").glob(".*.tmp"))
    ignore = (Path(__file__).resolve().parents[1] / ".gitignore").read_text(encoding="utf-8")
    assert check_cache.CACHE_REL.as_posix() in ignore.splitlines()


def test_unreadable_cache_is_treated_as_empty(repo: Path, calls: list[str]) -> None:
    _run(repo)
    (repo / "results").mkdir(exist_ok=True)
    (repo / check_cache.CACHE_REL).write_text("{not json", encoding="utf-8")
    assert _cached(repo) == [] and len(calls) == 1
