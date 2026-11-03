# ruff: noqa: F811
"""Publisher-provided labels (DECODE, QMSum) count as reviewed for claims, and a claim from such a record must say
"on published labels". The decision comes from the record's scenario_source, so older records, which were written
with labels_reviewed false and no label_source, are covered too."""
import json
from pathlib import Path

from contextlens.eval import claims
from contextlens.eval.records import labels_reviewed
from tests.test_records import _edit, _run, _write_claims, repo, set_h2  # noqa: F401
from tests.test_sources import _local_repo, _local_run

WORDS = "on published labels"


def _claim(path: Path, **extra) -> dict:
    rec = json.loads(path.read_text(encoding="utf-8"))
    return {"id": "c1", "run_id": rec["run_id"], "metric": "mrr", "value": rec["metrics"]["main"]["mrr"], **extra}


def _problems(repo: Path, claim: dict) -> list[str]:
    _write_claims(repo, [claim])
    return claims.check_claims(repo)


def _older(path: Path) -> None:
    """Make the record look like one written before this rule: flag false, no label_source."""
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec["labels_reviewed"] = False
    rec.pop("label_source", None)
    path.write_text(json.dumps(rec), encoding="utf-8")


def test_write_run_marks_publisher_labels_reviewed(repo: Path) -> None:
    set_h2(repo, "pending")
    assert labels_reviewed(repo, "qmsum") is True and labels_reviewed(repo, "decode") is True
    assert labels_reviewed(repo, "scenarios") is False
    rec = json.loads(_local_run(_local_repo(repo)).read_text(encoding="utf-8"))
    assert rec["labels_reviewed"] is True and rec["label_source"] == "publisher"


def test_a_published_label_claim_with_the_words_passes(repo: Path) -> None:
    set_h2(repo, "pending")
    path = _local_run(_local_repo(repo))
    assert _problems(repo, _claim(path, text=f"MRR 0.9 {WORDS}")) == []


def test_a_published_label_claim_without_the_words_fails(repo: Path) -> None:
    set_h2(repo, "pending")
    path = _local_run(_local_repo(repo))
    assert any(WORDS in p for p in _problems(repo, _claim(path, text="MRR is 0.9")))
    assert any(WORDS in p for p in _problems(repo, _claim(path)))


def test_an_older_record_is_decided_from_its_scenario_source(repo: Path) -> None:
    set_h2(repo, "pending")
    path = _local_run(_local_repo(repo))
    _older(path)
    assert _problems(repo, _claim(path, text=f"MRR {WORDS}")) == []


def test_our_scenarios_still_need_h2_and_need_no_wording(repo: Path) -> None:
    set_h2(repo, "pending")
    path = _run(repo)
    problems = _problems(repo, {"id": "c1", "run_id": json.loads(path.read_text(encoding="utf-8"))["run_id"],
                                "metric": "macro_f1",
                                "value": json.loads(path.read_text(encoding="utf-8"))["metrics"]["main"]["macro_f1"]})
    assert problems == ["claim c1: run has labels_reviewed false"]
    set_h2(repo, "done")
    path = _run(repo)
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert _problems(repo, {"id": "c1", "run_id": rec["run_id"], "metric": "macro_f1",
                            "value": rec["metrics"]["main"]["macro_f1"]}) == []


def test_a_forged_label_source_cannot_make_our_scenarios_publisher(repo: Path) -> None:
    set_h2(repo, "pending")
    path = _run(repo)
    _edit(path, label_source="publisher")
    problems = _problems(repo, _claim_macro(path))
    assert any("labels_reviewed false" in p for p in problems) and any("label_source" in p for p in problems)


def _claim_macro(path: Path) -> dict:
    rec = json.loads(path.read_text(encoding="utf-8"))
    return {"id": "c1", "run_id": rec["run_id"], "metric": "macro_f1", "value": rec["metrics"]["main"]["macro_f1"]}


def test_a_publisher_record_cannot_carry_our_scenarios(repo: Path) -> None:
    """A record names one scenario source and check_run ties its rows to that source's manifest, so a record that
    claims to be QMSum but holds our scenarios' rows fails, and cannot borrow the publisher rule."""
    set_h2(repo, "pending")
    _local_repo(repo)
    path = _run(repo)
    _edit(path, scenario_source="qmsum", label_source="publisher", labels_reviewed=True)
    problems = _problems(repo, _claim_macro(path))
    assert any("run:" in p for p in problems)


def test_an_undetermined_source_is_not_reviewed(repo: Path) -> None:
    set_h2(repo, "done")
    path = _run(repo)
    _edit(path, scenario_source="mystery")
    assert any("label source" in p for p in _problems(repo, _claim_macro(path)))
    rec = json.loads(path.read_text(encoding="utf-8"))
    del rec["scenario_source"]
    path.write_text(json.dumps(rec), encoding="utf-8")
    assert any("label source" in p for p in _problems(repo, _claim_macro(path)))
