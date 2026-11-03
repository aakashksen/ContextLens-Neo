import csv
from collections import Counter
from pathlib import Path

import pytest

from contextlens.experiments import e9_study as e9

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "docs" / "study"

# Invented fixture values: points out of 2, as (recall assisted, recall alone, clarify assisted, clarify alone).
POINTS = {
    "P01": (2, 1, 2, 1),
    "P02": (2, 2, 2, 1),
    "P03": (1, 1, 1, 0),
    "P04": (2, 0, 1, 1),
    "P05": (1, 1, 2, 1),
}


def rows_from(points):
    out = []
    for code, (ra, ru, ca, cu) in points.items():
        for task, cond, pts in (("recall_a", "assisted", ra), ("recall_b", "alone", ru),
                                ("clarify_a", "assisted", ca), ("clarify_b", "alone", cu)):
            out.append({"participant_code": code, "task_id": task, "condition": cond, "points": str(pts)})
    return out


@pytest.mark.parametrize("n", [5, 6, 7, 8])
def test_order_is_balanced(n):
    plans = e9.assign(n)
    assert len(plans) == n
    assert len({p.code for p in plans}) == n
    # every person does every task once, and gets the assistant on one task of each kind
    for p in plans:
        tasks = p.tasks()
        assert sorted(t for t, _ in tasks) == sorted(e9.TASKS)
        for kind in ("recall", "clarify"):
            conds = [c for t, c in tasks if t.startswith(kind)]
            assert sorted(conds) == ["alone", "assisted"]
    # each of the three counterbalanced choices is split evenly, or off by one when n is odd
    for factor in ("assisted_first", "kind_first", "assisted_version"):
        counts = Counter(getattr(p, factor) for p in plans)
        assert len(counts) == 2
        assert max(counts.values()) - min(counts.values()) == n % 2
    # no two people share the same full arrangement, so no arrangement is over-used
    assert len({(p.assisted_first, p.kind_first, p.assisted_version) for p in plans}) == n
    # the first task of each person: no condition or kind dominates it by more than one
    firsts = Counter(p.tasks()[0] for p in plans)
    assert max(firsts.values()) <= 2


def test_assign_rejects_outside_5_to_8():
    for n in (0, 4, 9):
        with pytest.raises(ValueError):
            e9.assign(n)


def test_analysis_on_fixture():
    res = e9.analyze(rows_from(POINTS))
    assert res["n"] == 5
    assert res["per_person"] == pytest.approx({"P01": 0.5, "P02": 0.25, "P03": 0.25, "P04": 0.5, "P05": 0.25})
    assert res["mean_diff"] == pytest.approx(0.35)
    lo, hi = res["t_interval"]
    assert lo == pytest.approx(0.35 - 2.776 * 0.0612372, abs=1e-3)
    assert hi == pytest.approx(0.35 + 2.776 * 0.0612372, abs=1e-3)
    blo, bhi = res["bootstrap_interval"]
    assert 0.25 <= blo <= 0.35 <= bhi <= 0.5
    assert res["per_kind"]["recall"]["n"] == 5
    assert res["per_kind"]["recall"]["mean_diff"] == pytest.approx(0.3)
    assert "p_value" not in res and "p-value" not in e9.report(res).lower().replace("no p-value", "")


def test_withdrawn_participant_is_removed_and_n_is_real():
    res = e9.analyze(rows_from(POINTS), withdrawn={"P05"})
    assert res["n"] == 4
    assert "P05" not in res["per_person"]
    assert res["mean_diff"] == pytest.approx(0.375)
    assert res["withdrawn_removed"] == 1


def test_missing_task_is_excluded_not_filled():
    rows = [r for r in rows_from(POINTS) if not (r["participant_code"] == "P04" and r["task_id"] == "clarify_b")]
    res = e9.analyze(rows)
    assert res["n"] == 5
    assert res["per_person"]["P04"] == pytest.approx(1.0)  # recall only
    assert res["per_kind"]["clarify"]["n"] == 4
    assert res["per_kind"]["recall"]["n"] == 5


def test_too_few_people_gives_no_interval():
    res = e9.analyze(rows_from({"P01": POINTS["P01"]}))
    assert res["n"] == 1 and res["t_interval"] is None
    assert "fewer than" in e9.report(res)


def test_interval_including_zero_is_not_called_an_effect():
    res = e9.analyze(rows_from({"P01": (2, 1, 2, 1), "P02": (1, 2, 1, 2), "P03": (2, 2, 1, 1),
                                "P04": (1, 1, 2, 2), "P05": (2, 1, 1, 2)}))
    lo, hi = res["t_interval"]
    assert lo < 0 < hi
    assert "includes zero" in e9.report(res)


def test_template_has_no_identity_fields():
    deny = {"name", "email", "phone", "address", "dob", "birth", "birthdate", "age", "gender", "sex", "race",
            "ethnicity", "diagnosis", "neurotype", "neurodivergent", "disability", "ucsb", "netid", "student",
            "signature", "ip", "device", "mac", "audio", "video", "photo", "notes", "comment", "comments"}
    with open(STUDY / "results-template.csv", newline="") as f:
        header = next(csv.reader(f))
    assert header == list(e9.RESULT_FIELDS)
    for field in header:
        assert not set(field.lower().split("_")) & deny, field
    assert e9.RESULT_FIELDS[0] == "participant_code"


def test_results_are_kept_out_of_git():
    ignore = (ROOT / ".gitignore").read_text().splitlines()
    assert "results/study/" in ignore


def test_study_docs_exist_and_tasks_are_covered():
    for name in ("consent.md", "script.md", "tasks.md", "answer-key.md"):
        assert (STUDY / name).exists(), name
    tasks = (STUDY / "tasks.md").read_text()
    key = (STUDY / "answer-key.md").read_text()
    for task in e9.TASKS:
        assert f"`{task}`" in tasks and f"`{task}`" in key
    consent = (STUDY / "consent.md").read_text().lower()
    assert "draft" in consent and "not been reviewed" in consent
    assert "diagnos" not in consent.replace("do not ask about diagnosis", "")


def _shingles(text, k=5):
    words = "".join(c.lower() if c.isalnum() else " " for c in text).split()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def test_staged_conversations_do_not_reuse_dev_scenario_text():
    # Only the dev split is read here; the test split is never opened by this task.
    staged = _shingles((STUDY / "tasks.md").read_text())
    assert staged
    dev = _shingles((ROOT / "datasets" / "scenarios" / "dev.jsonl").read_text(encoding="utf-8"))
    assert not staged & dev
