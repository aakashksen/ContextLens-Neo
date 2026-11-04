# ruff: noqa: F811
"""A record has one labels_reviewed flag, the one write_run sets from the label source. A copy in config cannot disagree."""
import json
from pathlib import Path

from contextlens.eval.check_run import check
from tests.test_records import _edit, _run, repo, set_h2  # noqa: F401
from tests.test_sources import _local_repo, _local_run


def _config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["config"]


def test_write_run_does_not_keep_a_labels_reviewed_copy_in_config(repo: Path) -> None:
    set_h2(repo, "pending")
    path = _local_run(_local_repo(repo))
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert rec["labels_reviewed"] is True and "labels_reviewed" not in rec["config"]


def test_check_run_fails_when_config_and_top_level_labels_reviewed_disagree(repo: Path) -> None:
    set_h2(repo, "pending")
    path = _local_run(_local_repo(repo))
    assert check("E1", root=repo) == []
    _edit(path, config={**_config(path), "labels_reviewed": False})
    assert any("labels_reviewed" in p and "disagree" in p for p in check("E1", root=repo))


def test_check_run_accepts_a_config_copy_that_agrees(repo: Path) -> None:
    set_h2(repo, "pending")
    path = _local_run(_local_repo(repo))
    _edit(path, config={**_config(path), "labels_reviewed": True})
    assert check("E1", root=repo) == []
