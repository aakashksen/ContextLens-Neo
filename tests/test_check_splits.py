import hashlib
import json
from pathlib import Path

from contextlens.data.check_splits import check

LINES = ['{"id": "a-1", "text": "alpha"}', '{"id": "a-2", "text": "beta"}']


def _setup(tmp_path: Path, source: str | None, *, write: bool = True, rel: str, hash_: str | None = None,
           n: int = 2, ids: list[str] | None = None) -> Path:
    body = "\n".join(LINES) + "\n"
    if write:
        f = tmp_path / rel / "dev.jsonl"
        f.parent.mkdir(parents=True)
        f.write_bytes(body.encode())
    entry: dict = {"hash": hash_ or hashlib.sha256(body.encode()).hexdigest(), "n": n}
    if ids is not None:
        entry["ids"] = ids
    manifest: dict = {"kind": "scenarios", "splits": {"dev": entry}}
    if source:
        manifest["source"] = source
    splits = tmp_path / "datasets" / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    (splits / "m.json").write_text(json.dumps(manifest), encoding="utf-8")
    return splits


def _check(splits: Path) -> list[str]:
    return check(splits, splits.parent / "raw")


QM = "datasets/raw/scenarios/qmsum"


def test_local_source_matches(tmp_path):
    assert _check(_setup(tmp_path, "qmsum", rel=QM, ids=["a-1", "a-2"])) == []


def test_local_source_wrong_hash(tmp_path):
    problems = _check(_setup(tmp_path, "qmsum", rel=QM, hash_="0" * 64))
    assert len(problems) == 1 and "differs from the manifest" in problems[0]


def test_local_source_absent_is_not_a_failure(tmp_path, capsys):
    assert _check(_setup(tmp_path, "qmsum", write=False, rel=QM, ids=["a-1", "a-2"])) == []
    assert "not present" in capsys.readouterr().out


def test_local_source_absent_still_checks_manifest_ids(tmp_path):
    problems = _check(_setup(tmp_path, "qmsum", write=False, rel=QM, ids=["a-1", "a-1"]))
    assert len(problems) == 1 and "duplicate" in problems[0]


def test_unknown_source_is_a_problem(tmp_path):
    problems = _check(_setup(tmp_path, "nope", rel=QM))
    assert len(problems) == 1 and "unknown scenario source 'nope'" in problems[0]


def test_committed_source_absent_fails(tmp_path):
    problems = _check(_setup(tmp_path, None, write=False, rel="datasets/scenarios"))
    assert len(problems) == 1 and "missing" in problems[0]


def test_committed_source_wrong_hash_fails(tmp_path):
    problems = _check(_setup(tmp_path, None, rel="datasets/scenarios", hash_="0" * 64))
    assert len(problems) == 1 and "differs from the manifest" in problems[0]


def test_committed_source_matches(tmp_path):
    assert _check(_setup(tmp_path, None, rel="datasets/scenarios")) == []


def test_count_and_ids_checked_against_file(tmp_path):
    problems = _check(_setup(tmp_path, "qmsum", rel=QM, n=3, ids=["a-1", "a-3"]))
    assert any("n=3" in p for p in problems) and any("ids" in p for p in problems)
