# ruff: noqa: F811
"""One run record holds many systems: each is checked, and so are the intervals and comparisons."""
import json
from pathlib import Path

import pytest

from contextlens.eval import claims
from contextlens.eval.check_run import check
from contextlens.eval.series import comparison, interval
from contextlens.replay.runner import Prediction
from tests.test_records import (  # noqa: F401  (repo is a fixture)
    _edit, _gold, _pred, _predl, _run, _scenario, _write_claims, repo,
)

GOLD = _gold("dev")  # contradiction, consistent, update, insufficient
A_LABELS = [g for _, g in _predl("dev")]  # right on 3 of 4 (an abstained card counts as insufficient)


def _sys_a() -> list[Prediction]:
    return [_pred(s, g) for s, g in _predl("dev")]


def _sys_b() -> list[Prediction]:
    return [_pred(s, "update") for s, _ in GOLD]  # right only on the third


def _rows(preds: list[Prediction]):
    by_id = {p.scenario_id: p for p in preds}
    return [(_scenario(i, g), by_id[i]) for i, g in GOLD]


def _two(repo: Path, **kw) -> Path:
    return _run(repo, systems={"single": _sys_a(), "team": _sys_b()}, **kw)


def _rec(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _comp(a: str = "single", b: str = "team", metric: str = "label_accuracy") -> dict:
    out = comparison(_rows(_sys_a()), _rows(_sys_b()), metric)
    assert out is not None
    return {"a": a, "b": b, **out}


def test_two_systems_in_one_record_pass(repo: Path) -> None:
    path = _two(repo)
    assert check("E1", root=repo) == []
    rec = _rec(path)
    assert set(rec["predictions"]) == set(rec["metrics"]) == {"single", "team"}
    assert rec["n"] == 4


def test_each_system_is_checked_for_its_metrics(repo: Path) -> None:
    path = _two(repo)
    rec = _rec(path)
    rec["metrics"]["team"]["macro_f1"] = 0.99
    _edit(path, metrics=rec["metrics"])
    problems = check("E1", root=repo)
    assert any("team" in p and "macro_f1" in p for p in problems)
    assert not any("single" in p for p in problems)


def test_each_system_needs_every_scenario_id_once(repo: Path) -> None:
    bad = [_pred(i, "update") for i in ["d1", "d1", "d3", "d4"]]
    _run(repo, systems={"single": _sys_a(), "team": bad})
    assert any("team" in p and "prediction ids" in p for p in check("E1", root=repo))


def test_systems_must_cover_the_same_cases(repo: Path) -> None:
    other = [_pred(s, g) for s, g in _predl("test")]
    _run(repo, systems={"single": _sys_a(), "team": other})
    assert any("team" in p and "prediction ids" in p for p in check("E1", root=repo))


def test_interval_recomputes(repo: Path) -> None:
    lo, hi = interval(_rows(_sys_a()), "label_accuracy") or (0, 0)
    assert 0 <= lo <= 0.75 <= hi <= 1
    _two(repo, intervals={"single": {"label_accuracy": (lo, hi)}})
    assert check("E1", root=repo) == []


def test_wrong_interval_fails(repo: Path) -> None:
    lo, hi = interval(_rows(_sys_a()), "label_accuracy") or (0, 0)
    _two(repo, intervals={"single": {"label_accuracy": (lo, hi + 0.01)}})
    assert any("interval" in p and "label_accuracy" in p for p in check("E1", root=repo))


def test_interval_that_cannot_be_recomputed_fails(repo: Path) -> None:
    _two(repo, intervals={"single": {"word_error_rate": (0.1, 0.9)}})
    assert any("interval" in p and "cannot be recomputed" in p for p in check("E1", root=repo))
    _two(repo, intervals={"ghost": {"label_accuracy": (0.1, 0.9)}})
    assert any("ghost" in p for p in check("E1", root=repo))


def test_comparison_recomputes_by_hand_and_by_code(repo: Path) -> None:
    c = _comp()
    assert c["diff"] == pytest.approx(0.5)  # a right on 3 of 4, b on 1 of 4, a-b per case: 1, 0, 0, 1
    assert c["mcnemar_p"] == pytest.approx(0.5)  # two cases only a gets right
    _two(repo, comparisons=[c])
    assert check("E1", root=repo) == []


@pytest.mark.parametrize("change", [{"diff": 0.6}, {"ci": [0.1, 0.9]}, {"mcnemar_p": 0.01}, {"metric": "macro_f1"},
                                    {"b": "ghost"}, {"b": "single"}])
def test_wrong_comparison_fails(repo: Path, change: dict) -> None:
    _two(repo, comparisons=[{**_comp(), **change}])
    assert any("comparison" in p for p in check("E1", root=repo))


def test_mcnemar_on_a_non_binary_series_fails(repo: Path) -> None:
    c = {**_comp(metric="tokens_per_card"), "mcnemar_p": 0.5}
    _two(repo, comparisons=[c])
    assert any("comparison" in p for p in check("E1", root=repo))


def test_one_record_for_all_systems_is_one_touch_of_test(repo: Path) -> None:
    systems = {n: [_pred(s, g) for s, g in _predl("test")] for n in ("a", "b", "c")}
    _run(repo, "test", systems=systems)
    assert check("E1", root=repo, split="test") == []


def test_renaming_the_experiment_does_not_escape_the_test_once_rule(repo: Path) -> None:
    _run(repo, "test", experiment="E8")
    _run(repo, "test", experiment="E8rerun")
    assert any("override_note" in p for p in check("E8", root=repo, split="test"))
    assert any("override_note" in p for p in check("E8rerun", root=repo, split="test"))


def test_an_earlier_bad_record_is_still_checked(repo: Path) -> None:
    _run(repo, metrics={"macro_f1": 0.99})
    _run(repo)
    assert any("macro_f1" in p for p in check("E1", root=repo, split="dev"))


def test_superseded_record_is_skipped_but_stays(repo: Path) -> None:
    old = _run(repo, metrics={"macro_f1": 0.99})
    new = _run(repo, supersedes=[{"run_id": _rec(old)["run_id"], "reason": "metric typo"}])
    assert check("E1", root=repo, split="dev") == []
    assert old.is_file() and _rec(new)["supersedes"][0]["reason"] == "metric typo"


@pytest.mark.parametrize("entry", [
    {"run_id": "RID", "reason": ""}, {"run_id": "RID", "reason": "  "}, {"run_id": "RID"},
    {"run_id": "E1-nothing", "reason": "x"},
])
def test_supersede_needs_a_reason_and_a_real_record(repo: Path, entry: dict) -> None:
    old = _run(repo, metrics={"macro_f1": 0.99})
    entry = {**entry, "run_id": _rec(old)["run_id"] if entry["run_id"] == "RID" else entry["run_id"]}
    _run(repo, supersedes=[entry])
    problems = check("E1", root=repo, split="dev")
    assert any("macro_f1" in p for p in problems) or any("supersedes" in p for p in problems)
    assert problems != []


def test_an_earlier_record_cannot_supersede_a_later_one(repo: Path) -> None:
    first = _run(repo)
    second = _run(repo, metrics={"macro_f1": 0.99})
    _edit(first, supersedes=[{"run_id": _rec(second)["run_id"], "reason": "backwards"}])
    assert check("E1", root=repo, split="dev") != []


def test_write_run_rejects_unsafe_system_names(repo: Path) -> None:
    with pytest.raises(ValueError):
        _run(repo, systems={"../x": _sys_a()})


def _claim_run(repo: Path) -> dict:
    lo, hi = interval(_rows(_sys_a()), "label_accuracy") or (0, 0)
    return _rec(_two(repo, intervals={"single": {"label_accuracy": (lo, hi)}}, comparisons=[_comp()]))


def test_claim_must_name_the_system_when_there_are_several(repo: Path) -> None:
    rec = _claim_run(repo)
    value = rec["metrics"]["single"]["macro_f1"]
    good = {"id": "c1", "run_id": rec["run_id"], "system": "single", "metric": "macro_f1", "value": value}
    _write_claims(repo, [good])
    assert claims.check_claims(repo) == []
    _write_claims(repo, [{k: v for k, v in good.items() if k != "system"}])
    assert any("system" in p for p in claims.check_claims(repo))
    _write_claims(repo, [{**good, "system": "ghost"}])
    assert any("ghost" in p for p in claims.check_claims(repo))
    _write_claims(repo, [{**good, "system": "team"}])  # the other system's value is different
    assert any("value differs" in p for p in claims.check_claims(repo))


def test_single_system_claim_needs_no_system_name(repo: Path) -> None:
    rec = _rec(_run(repo))
    _write_claims(repo, [{"id": "c1", "run_id": rec["run_id"], "metric": "macro_f1",
                          "value": rec["metrics"]["main"]["macro_f1"]}])
    assert claims.check_claims(repo) == []


def test_interval_claim_must_equal_the_record(repo: Path) -> None:
    rec = _claim_run(repo)
    good = {"id": "c1", "kind": "interval", "run_id": rec["run_id"], "system": "single",
            "metric": "label_accuracy", "value": rec["intervals"]["single"]["label_accuracy"]}
    _write_claims(repo, [good])
    assert claims.check_claims(repo) == []
    _write_claims(repo, [{**good, "value": [good["value"][0], good["value"][1] - 0.001]}])
    assert any("value differs" in p for p in claims.check_claims(repo))
    _write_claims(repo, [{**good, "metric": "mrr"}])
    assert any("not in run" in p for p in claims.check_claims(repo))


def test_comparison_claim_must_equal_the_record(repo: Path) -> None:
    rec = _claim_run(repo)
    saved = rec["comparisons"][0]
    good = {"id": "c1", "kind": "comparison", "run_id": rec["run_id"], "a": "single", "b": "team",
            "metric": "label_accuracy", "field": "diff", "value": saved["diff"]}
    _write_claims(repo, [good])
    assert claims.check_claims(repo) == []
    _write_claims(repo, [{**good, "field": "mcnemar_p", "value": saved["mcnemar_p"]}])
    assert claims.check_claims(repo) == []
    _write_claims(repo, [{**good, "value": saved["diff"] + 0.01}])
    assert any("value differs" in p for p in claims.check_claims(repo))
    _write_claims(repo, [{**good, "a": "team", "b": "single"}])
    assert any("not in run" in p for p in claims.check_claims(repo))


def test_claim_on_a_superseded_record_fails(repo: Path) -> None:
    old = _rec(_run(repo))
    _run(repo, supersedes=[{"run_id": old["run_id"], "reason": "rerun"}])
    _write_claims(repo, [{"id": "c1", "run_id": old["run_id"], "metric": "macro_f1",
                          "value": old["metrics"]["main"]["macro_f1"]}])
    assert any("superseded" in p for p in claims.check_claims(repo))


def test_write_run_keeps_one_file_per_system(repo: Path) -> None:
    rec = _rec(_two(repo))
    files = {k: repo / v for k, v in rec["predictions"].items()}
    assert all(f.is_file() for f in files.values()) and len(set(files.values())) == 2
