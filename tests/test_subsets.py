# ruff: noqa: F811
"""A record may cover a named subset of its split, fixed by a manifest committed before the run."""
import json
from pathlib import Path

import pytest

from contextlens.eval.check_run import check
from tests.test_records import _git, _pred, _predl, _run, repo  # noqa: F401  (repo is a fixture)

SUBSETS = Path("datasets/splits/subsets")


def _manifest(root: Path, name: str = "s1", ids: list[str] | None = None, split: str = "dev",
              commit: bool = True) -> None:
    path = root / SUBSETS / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"name": name, "split": split, "ids": ids or ["d1", "d2"]}), encoding="utf-8")
    if commit:
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", f"subset {name}")


def _preds(*ids: str):
    by_id = dict(_predl("dev"))
    return [_pred(i, by_id.get(i, "update")) for i in ids]


def _sub(root: Path, ids: tuple[str, ...] = ("d1", "d2"), **kw):
    return _run(root, preds=_preds(*ids), subset=kw.pop("subset", "s1"), **kw)


def test_a_named_committed_subset_passes(repo: Path) -> None:
    _manifest(repo)
    _sub(repo)
    assert check("E1", root=repo, split="dev") == []
    assert check("E1", root=repo, split="dev", min_n=2) == []
    assert any("min-n" in p for p in check("E1", root=repo, split="dev", min_n=3))


def test_two_systems_on_the_subset_pass(repo: Path) -> None:
    _manifest(repo)
    _run(repo, systems={"a": _preds("d1", "d2"), "b": _preds("d2", "d1")}, subset="s1")
    assert check("E1", root=repo, split="dev") == []


def test_a_short_record_with_no_subset_name_fails(repo: Path) -> None:
    _manifest(repo)
    _run(repo, preds=_preds("d1", "d2"))
    problems = check("E1", root=repo, split="dev")
    assert any("differs from the split" in p for p in problems)
    assert any("prediction ids" in p for p in problems)


@pytest.mark.parametrize("name", ["", "ghost", "../x", "a/b"])
def test_an_unnamed_or_unknown_subset_fails(repo: Path, name: str) -> None:
    _manifest(repo)
    _sub(repo, subset=name)
    assert any("subset" in p for p in check("E1", root=repo, split="dev"))


def test_a_manifest_committed_after_the_run_fails(repo: Path) -> None:
    _sub(repo)  # the run's commit has no manifest
    _manifest(repo)
    assert any("not committed before the run" in p for p in check("E1", root=repo, split="dev"))


def test_a_manifest_edited_after_the_run_fails(repo: Path) -> None:
    _manifest(repo)
    _sub(repo)
    _manifest(repo, ids=["d1", "d3"])
    assert any("not committed before the run" in p for p in check("E1", root=repo, split="dev"))


def test_cherry_picked_ids_fail(repo: Path) -> None:
    _manifest(repo)
    _sub(repo, ids=("d1", "d3"))
    assert any("prediction ids" in p for p in check("E1", root=repo, split="dev"))
    _sub(repo, ids=("d1", "d2", "d3"))
    assert any("n=3" in p or "prediction ids" in p for p in check("E1", root=repo, split="dev"))


def test_n_must_equal_the_subset_count(repo: Path) -> None:
    _manifest(repo)
    path = _sub(repo)
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec["n"] = 3
    path.write_text(json.dumps(rec), encoding="utf-8")
    assert any("differs from the subset" in p for p in check("E1", root=repo, split="dev"))


def test_systems_must_cover_the_same_subset(repo: Path) -> None:
    _manifest(repo)
    _run(repo, systems={"a": _preds("d1", "d2"), "b": _preds("d1", "d3")}, subset="s1")
    assert any("system b" in p and "prediction ids" in p for p in check("E1", root=repo, split="dev"))


def test_ids_must_belong_to_the_records_split(repo: Path) -> None:
    _manifest(repo, ids=["d1", "t2"])
    _sub(repo, ids=("d1", "t2"))
    assert any("not in the split" in p for p in check("E1", root=repo, split="dev"))
    _manifest(repo, name="s2", ids=["d1", "d2"], split="test")
    _sub(repo, subset="s2")
    assert any("belongs to split test" in p for p in check("E1", root=repo, split="dev"))


def test_a_manifest_with_repeated_or_no_ids_fails(repo: Path) -> None:
    _manifest(repo, ids=["d1", "d1"])
    _sub(repo, ids=("d1", "d1"))
    assert any("repeats" in p for p in check("E1", root=repo, split="dev"))
    _manifest(repo, name="s3", ids=[])
    _manifest(repo, name="s3", ids=["d1"])
    (repo / SUBSETS / "s3.json").write_text(json.dumps({"name": "s3", "split": "dev", "ids": []}), encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "empty")
    _sub(repo, subset="s3", ids=("d1",))
    assert any("no ids" in p for p in check("E1", root=repo, split="dev"))
