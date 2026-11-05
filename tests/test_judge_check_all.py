"""Judge check on every claim of named systems, over the old and the new set. FakeLLM only; nothing is called for real."""
import json
from pathlib import Path

import pytest

from contextlens.eval import series
from contextlens.eval.check_run import check
from contextlens.eval.records import RUNS_REL, write_local_manifest, write_scenarios_manifest
from contextlens.experiments import judge_check as jc
from contextlens.experiments import judge_check_all as jca
from contextlens.replay.runner import Prediction
from contextlens.schemas import Card, Scenario
from tests.test_judge_check import Cloud, _counter, _finding, _git, _local, _record, _scn

OLD, NEW = "E3T-20261104T000000000001Z", "E3T-20261104T000000000002Z"
ARM_OLD, ARM_NEW = "E3TS-20261104T000000000003Z", "E3TS-20261104T000000000004Z"
N_OLD, N_NEW = 12, 6
LAST = {OLD: ("single", "team_changed", "team"), NEW: ("single", "team_changed", "team"),
        ARM_OLD: ("team_shipping_changed",), ARM_NEW: ("team_shipping_changed",)}


def _sc(i: int, new: bool) -> Scenario:
    s = _scn(i)
    sid = f"x{i:02d}" if new else s.id
    turns = [t.model_copy(update={"id": f"{sid}:t{k}", "conversation_id": sid}) for k, t in enumerate(s.conversation.turns)]
    return s.model_copy(update={"id": sid, "conversation": s.conversation.model_copy(update={"id": sid, "turns": turns}),
                                "request": s.request.model_copy(update={"conversation_id": sid}),
                                "gold_evidence_turn_ids": [turns[0].id], "origin": "synthetic",
                                "gold_label": "insufficient" if i % 4 == 3 else None})


def _rows(system: str, new: bool, all_good: bool = False) -> list[Prediction]:
    """Insufficient cases (every fourth) abstain; the others give one claim the cloud accepts ('good') and one it rejects."""
    out = []
    for i in range(N_NEW if new else N_OLD):
        sid = _sc(i, new).id
        if i % 4 == 3:
            out.append(Prediction(scenario_id=sid, card=Card(mode="recall", status="abstained", findings=[], missing=["none"])))
            continue
        fs = [_finding(sid, f"good {system} {sid}"), _finding(sid, f"{'good' if all_good else 'bad'} {system} {sid}")]
        out.append(Prediction(scenario_id=sid, card=Card(mode="recall", status="answered", findings=fs),
                              claim_supported=[_local(i, 0, False), _local(i, 1, False) or all_good]))
    return out


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(series, "RESAMPLES", 200)
    d = tmp_path / "datasets/scenarios"
    d.mkdir(parents=True)
    (d / "dev.jsonl").write_text("".join(_sc(i, False).model_dump_json() + "\n" for i in range(N_OLD)), encoding="utf-8")
    write_scenarios_manifest(tmp_path)
    write_local_manifest("scenarios_extra", {"dev": [_sc(i, True) for i in range(N_NEW)]}, tmp_path)
    manifest = json.loads((tmp_path / "datasets/splits/scenarios-extra.json").read_text(encoding="utf-8"))
    extra = {"scenario_source": "scenarios_extra", "dataset_hash": manifest["splits"]["dev"]["hash"]}
    for rid, new in ((OLD, False), (NEW, True), (ARM_OLD, False), (ARM_NEW, True)):
        _record(tmp_path, rid, {s: _rows(s, new, all_good=new and s == "team_shipping_changed") for s in LAST[rid]},
                **(extra if new else {}))
    (tmp_path / ".gitignore").write_text("results/partial/\n", encoding="utf-8")
    for args in (("init", "-q"), ("add", "-A"), ("commit", "-q", "-m", "init")):
        _git(tmp_path, *args)
    return tmp_path


ALL = [OLD, NEW, ARM_OLD, ARM_NEW]
NAMED = ["single", "team_changed", "team_shipping_changed"]
CLAIMS = (9 * 2 + 5 * 2) * 3  # three systems, each with 9 old and 5 new answered cases of two claims


def _run(repo: Path, llm, **kw):
    msgs = kw.pop("msgs", [])
    return jca.run_all(kw.pop("ids", ALL), kw.pop("systems", NAMED), llm=llm, grader_name="test-cloud", root=repo,
                       counter_path=_counter(repo), cap=kw.pop("cap", 5000), say=msgs.append)


def test_only_the_named_systems_are_loaded_and_sent_to_the_grader(repo: Path) -> None:
    held, _, cov = jca.load_systems(ALL, NAMED, repo)
    assert {(h.system, h.source) for h in held} == {(s, src) for s in NAMED for src in ("scenarios", "scenarios_extra")}
    assert sum(len(h.claims) for h in held) == CLAIMS and cov["claims"]["single/new"] == 10
    assert cov["records"][OLD]["systems_not_graded"] == ["team"] and cov["records"][OLD]["set"] == "old"
    cloud = Cloud()
    _run(repo, cloud)
    assert len(cloud.calls) == CLAIMS
    assert "good team " not in json.dumps(cloud.calls) and "bad team " not in json.dumps(cloud.calls)


def test_it_refuses_an_unknown_system_a_system_in_two_records_and_a_source_without_cards(repo: Path) -> None:
    with pytest.raises(jc.Refused, match="no named record holds"):
        jca.load_systems(ALL, ["single", "nope"], repo)
    _record(repo, "E3T-20261104T000000000009Z", {"single": _rows("single", False)})
    with pytest.raises(jc.Refused, match="in two of the named records"):
        jca.load_systems([*ALL, "E3T-20261104T000000000009Z"], NAMED, repo)
    _record(repo, "E3T-20261104T000000000010Z", {"single": []}, scenario_source="decode")
    with pytest.raises(jc.Refused, match="only"):
        jca.load_systems(["E3T-20261104T000000000010Z"], ["single"], repo)


def test_it_prints_how_many_grader_calls_it_will_make_before_the_first_call(repo: Path) -> None:
    seen: list[tuple[str, int]] = []
    cloud = Cloud()
    jca.run_all(ALL, NAMED, llm=cloud, grader_name="test-cloud", root=repo, counter_path=_counter(repo), cap=5000,
                say=lambda t: seen.append((t, len(cloud.calls))))
    (text, calls_so_far), = seen
    assert calls_so_far == 0 and f"makes {CLAIMS} grader calls" in text and "0 already graded" in text


def test_the_records_pass_check_run_and_give_each_systems_score_on_old_new_and_both(repo: Path) -> None:
    paths = _run(repo, Cloud())
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "records")
    assert check("JC", root=repo, split="dev") == []
    recs = {json.loads(p.read_text(encoding="utf-8"))["scenario_source"]: json.loads(p.read_text(encoding="utf-8")) for p in paths}
    assert {s: r["n"] for s, r in recs.items()} == {"scenarios": N_OLD, "scenarios_extra": N_NEW}
    assert set(recs["scenarios_extra"]["predictions"]) == set(NAMED)
    table = recs["scenarios"]["config"]["scores"]
    assert table == recs["scenarios_extra"]["config"]["scores"] and set(table) == set(NAMED)
    one = table["single"]
    assert one["covers"]["sets"] == ["old", "new"] and one["covers"]["records"] == [OLD, NEW]
    # old: 9 answered cases, 18 claims; the cloud accepts the 'good' half; the local judge also accepts claim 1 of i in {0, 6, 9}
    assert one["old"]["cloud_grounded_rate"] == pytest.approx(0.5) and one["old"]["grounded_rate"] == pytest.approx(12 / 18)
    assert one["old"]["answer_when_answerable"] == 1.0 and one["old"]["abstain_when_insufficient"] == 1.0
    assert one["old"]["written_score_cloud"] == pytest.approx(0.75) and one["old"]["written_score_local"] == pytest.approx(5 / 6)
    assert one["new"]["cloud_grounded_rate"] == pytest.approx(0.5) and one["new"]["claims"] == 10
    arm = table["team_shipping_changed"]
    assert arm["old"]["written_score_cloud"] == pytest.approx(0.75) and arm["new"]["cloud_grounded_rate"] == 1.0
    assert arm["new"]["written_score_cloud"] == 1.0
    assert arm["both"]["cloud_grounded_rate"] == pytest.approx((9 + 10) / 28) and arm["both"]["claims"] == 28
    assert arm["both"]["written_score_cloud"] == pytest.approx(((9 + 10) / 28 + 1) / 2)
    m = recs["scenarios_extra"]["metrics"]["team_shipping_changed"]  # saved metrics are the ones check_run recomputed
    assert m["cloud_grounded_rate"] == 1.0 and m["answer_when_answerable"] == 1.0
    assert "Every claim" in " ".join(recs["scenarios"]["config"]["notes"]) and recs["scenarios"]["config"]["mode"] == "all_claims"
    assert recs["scenarios"]["claude_calls"] + recs["scenarios_extra"]["claude_calls"] == CLAIMS


def test_a_system_run_on_one_set_only_has_no_score_on_both(repo: Path) -> None:
    _run(repo, Cloud(), ids=[OLD, ARM_NEW], systems=["single", "team_shipping_changed"])
    (path,) = [p for p in (repo / RUNS_REL).glob("JC-*.json") if json.loads(p.read_text(encoding="utf-8"))["scenario_source"] == "scenarios"]
    table = json.loads(path.read_text(encoding="utf-8"))["config"]["scores"]
    assert table["single"]["new"] is None and table["single"]["both"] is None and table["single"]["old"]["claims"] == 18
    assert table["team_shipping_changed"]["old"] is None and table["team_shipping_changed"]["new"]["claims"] == 10


def test_a_stopped_run_resumes_by_commit_without_asking_twice_and_rows_hold_no_text(repo: Path) -> None:
    with pytest.raises(jc.Refused, match="failed in a row"):
        _run(repo, Cloud(limit=20))
    (rows_file,) = (repo / jc.PARTIAL_REL).glob("all-*.rows.jsonl")
    text = rows_file.read_text(encoding="utf-8")
    assert "friday" not in text and "good single" not in text and "Maya" not in text
    lines = [json.loads(x) for x in text.splitlines()]
    assert lines[0]["kind"] == "judge-check-all-claims-rows" and lines[0]["git_commit"]
    assert {k for ln in lines[1:] for k in ln} == {"claim_id", "status", "cloud", "error"}
    assert not list((repo / RUNS_REL).glob("JC-*"))
    msgs: list[str] = []
    again = Cloud()
    _run(repo, again, msgs=msgs)
    assert len(again.calls) == CLAIMS - 20 and "20 already graded" in msgs[0]
    assert len(list((repo / RUNS_REL).glob("JC-*.json"))) == 2


def test_a_resume_at_another_commit_is_refused_and_a_dirty_tree_stops_before_any_call(repo: Path) -> None:
    with pytest.raises(jc.Refused):
        _run(repo, Cloud(limit=3))
    _git(repo, "commit", "-q", "--allow-empty", "-m", "moved on")
    with pytest.raises(jc.Refused, match="different commit or sample"):
        _run(repo, Cloud())
    (repo / "stray.txt").write_text("x")
    cloud = Cloud()
    with pytest.raises(jc.Refused, match="uncommitted"):
        _run(repo, cloud)
    assert cloud.calls == []


def test_the_cap_is_checked_against_the_shared_counter_before_any_call(repo: Path) -> None:
    _counter(repo).write_text(json.dumps({"count": 4990, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}))
    cloud = Cloud()
    with pytest.raises(jc.Refused, match="only the person can raise the cap"):
        _run(repo, cloud)
    assert cloud.calls == []


def test_a_claim_without_a_verdict_blocks_the_record(repo: Path) -> None:
    with pytest.raises(jc.Refused, match="no record is written"):
        _run(repo, Cloud(fail_on="bad single x01"))
    assert not list((repo / RUNS_REL).glob("JC-*"))


def test_the_arguments_go_together() -> None:
    with pytest.raises(SystemExit):
        jc.main(["R", "--systems", "single"])
    with pytest.raises(SystemExit):
        jc.main(["R", "--all-claims"])
