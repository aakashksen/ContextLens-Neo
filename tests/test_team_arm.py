"""Entry 51: the run of `team_shipping_changed` alone, beside the saved rows of two E3T records. No real model."""

import json
import shutil

import pytest

from contextlens.eval import series
from contextlens.eval.check_run import check
from contextlens.experiments import e2_context as e2
from contextlens.experiments import e3_team as e3
from contextlens.experiments import team_arm as arm
from contextlens.experiments import team_attempt as ta
from contextlens.experiments.e2_context import RunBroken
from contextlens.llm.ollama_ctl import OllamaCtl
from contextlens.replay.runner import Prediction
from tests.test_e1 import HashEmbedder
from tests.test_e2 import NAMES, RETRIEVAL, Scripted, Wire, _git
from tests.test_team_attempt_run import REAL, Writer


def copy_repo(root):
    for rel in ("datasets/scenarios", "datasets/scenarios_extra", "datasets/splits"):
        shutil.copytree(REAL / rel, root / rel)
    (root / ".gitignore").write_text("results/partial/\n", encoding="utf-8")
    for args in (("init", "-q"), ("add", "-A"), ("commit", "-q", "-m", "init")):
        _git(root, *args)


def go_arm(repo, writer, prior, run_conditions="test"):
    return arm.run_arm(root=repo, writer=writer, judge_llm=Scripted("test-judge:1"), embedder=HashEmbedder(),
                       ctl=OllamaCtl(request=Wire()), names=NAMES, retrieval_config=RETRIEVAL, run_conditions=run_conditions,
                       prior=prior, say=lambda t: None)


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    """E3T run once (its four systems), then the arm: first stopped part-way, then resumed at the same commit."""
    mp = pytest.MonkeyPatch()
    mp.setattr(series, "RESAMPLES", 100)  # the runs and the checks use the same count
    repo = tmp_path_factory.mktemp("arm")
    copy_repo(repo)
    old = ta.run_attempt(root=repo, writer=Writer(), judge_llm=Scripted("test-judge:1"), embedder=HashEmbedder(),
                         ctl=OllamaCtl(request=Wire()), names=NAMES, retrieval_config=RETRIEVAL, run_conditions="test",
                         say=lambda t: None)
    prior = {s: p.stem for s, p in old.items()}
    e3t_partial = (repo / ta.PARTIAL_REL).read_text(encoding="utf-8")
    with pytest.raises(RunBroken):
        go_arm(repo, Writer(die_after=600), prior)
    stopped = (repo / arm.PARTIAL_REL).read_text(encoding="utf-8").splitlines()
    first_records = sorted((repo / "results/runs").glob(f"{arm.EXPERIMENT}-*.json"))
    writer = Writer()
    paths = go_arm(repo, writer, prior)
    yield {"repo": repo, "prior": prior, "paths": paths, "writer": writer, "stopped": stopped, "first_records": first_records,
           "e3t_partial": e3t_partial}
    mp.undo()


def rec_of(ran, source):
    return json.loads(ran["paths"][source].read_text(encoding="utf-8"))


def test_an_interrupted_run_keeps_rows_keyed_by_commit_and_system_and_the_resume_runs_only_the_rest(ran):
    repo = ran["repo"]
    saved = [json.loads(x) for x in ran["stopped"]]
    assert saved and not ran["first_records"]
    assert {(e["winner"], e["model"]) for e in saved} == {(arm.ARM, NAMES.writer)}
    assert len({e["commit"] for e in saved}) == 1
    done = {e["row"]["scenario_id"] for e in saved if e2.row_state(Prediction.model_validate(e["row"])) != "broken"}
    assert done and len(done) < 256
    seen = json.dumps(ran["writer"].calls)
    assert not any(f'"{sid}:' in seen or f"{sid}:t" in seen for sid in done)  # a saved row was not run again
    assert len(ran["paths"]) == 2 and (repo / arm.PARTIAL_REL).exists()
    assert (repo / ta.PARTIAL_REL).read_text(encoding="utf-8") == ran["e3t_partial"]  # E3T's file is not touched
    assert sum(rec_of(ran, s)["config"]["rows_reused_from_partial"] for s in ta.SOURCES) == len(done)


def test_the_new_records_pass_the_check_and_so_do_entry_48s(ran):
    repo = ran["repo"]
    assert check(arm.EXPERIMENT, root=repo, min_n=100, split="dev") == []
    assert check(ta.EXPERIMENT, root=repo, min_n=100, split="dev") == []
    for source, n in (("scenarios", 148), ("scenarios_extra", 108)):
        rec = rec_of(ran, source)
        assert rec["experiment"] == "E3TS" and rec["scenario_source"] == source and rec["n"] == n and rec["split"] == "dev"
        assert set(rec["predictions"]) == {arm.ARM} and set(rec["metrics"]) == {arm.ARM} and rec["comparisons"] == []


def test_the_config_states_what_entry_51_requires(ran):
    cfg = rec_of(ran, "scenarios_extra")["config"]
    assert cfg["decision_entry"] == 51 and cfg["arm"] == "team_shipping_changed" and cfg["no_further_team_variants"] is True
    assert "entry 51" in cfg["written_before_results"] and "exception to entry 48's one-attempt rule" in cfg["written_before_results"]
    assert "no further team variants" in cfg["written_before_results"]
    assert "unreviewed by a person" in cfg["new_set_labels"]
    assert "team_shipping_changed" in cfg["entry_51_rule_looks_at"] and "both" in cfg["entry_51_rule_looks_at"]
    assert cfg["cross_record"]["source_records"] == ran["prior"] and cfg["judge"]["model"] == NAMES.judge
    assert cfg["consistency_checker"]["class"] == "PromptedChecker" and cfg["deadline_s"] == 8.0
    assert {"row_report", "health", "drops", "devices", "memory_build", "memory_empty_by_case"} <= set(cfg)
    assert "failed_reasons" in cfg["memory_build"]


def test_scores_beside_the_other_arms_come_from_the_saved_rows_of_the_named_records(ran):
    repo = ran["repo"]
    cfg = rec_of(ran, "scenarios_extra")["config"]
    cross = cfg["cross_record"]
    assert cross["systems"] == list(ta.RUN_SYSTEMS) + [arm.ARM]
    assert set(cross["scores"]) == {*ta.SETS, *ta.SLICES} and all(set(v) == set(cross["systems"]) for v in cross["scores"].values())
    for source, key in (("scenarios", "old"), ("scenarios_extra", "new")):
        saved = json.loads((repo / "results/runs" / f"{ran['prior'][source]}.json").read_text(encoding="utf-8"))
        for name in ta.RUN_SYSTEMS:
            assert cross["scores"][key][name] == e3.score(saved["metrics"][name])  # the E3T record's own numbers, not retyped
        assert cross["scores"][key][arm.ARM] == e3.score(rec_of(ran, source)["metrics"][arm.ARM])
    assert {c["metric"] for c in cross["sets"]["new"]["comparisons"] if (c["a"], c["b"]) == (arm.ARM, "single")} >= {"grounded_rate"}
    assert cross["twins"]["all"]["by_system"][arm.ARM]["pairs"] == 54


def test_entry_51_is_stated_for_the_arm_and_for_team_changed_and_names_no_winner(ran):
    cross = rec_of(ran, "scenarios")["config"]["cross_record"]
    local = cross["entry_51_local"]
    assert local["judge"] == "local" and set(local["arms"]) == {arm.ARM, "team_changed"} and "not chosen" in local["winner"]
    new = cross["scores"]["new"]
    for name, got in local["arms"].items():
        assert got["score_new"] == new[name] and got["single_score_new"] == new["single"]
        assert got["beats_single_on_new_by_written_score"] == (new[name] > new["single"])


def test_the_change_fires_on_the_insufficient_twins_and_the_arm_is_the_checked_team(ran):
    rec = rec_of(ran, "scenarios_extra")
    assert rec["config"]["memory_empty_by_case"] == {"insufficient_cases": 54, "answerable_cases": 0}
    rows = [Prediction.model_validate_json(x) for x in (ran["repo"] / rec["predictions"][arm.ARM]).read_text(encoding="utf-8").splitlines()]
    assert all(p.card.status == "abstained" for p in rows if p.scenario_id.endswith("-ins"))
    assert not any(p.card.status == "abstained" for p in rows if p.scenario_id.endswith("-ans"))
    assert rec["config"]["drops"][arm.ARM]["answerable_cases_lost_to_a_check"] == 0


def test_the_cli_takes_only_the_named_arm_and_needs_a_run_condition():
    with pytest.raises(SystemExit):
        ta.main(["--arm", "team", "--run-conditions", "x"])
    with pytest.raises(SystemExit):
        ta.main(["--arm", arm.ARM, "--run-conditions", " "])
    with pytest.raises(SystemExit):
        ta.main(["--arm", arm.ARM])


def test_a_dirty_tree_a_missing_condition_or_a_missing_record_stops_before_any_model_call(tmp_path):
    copy_repo(tmp_path)
    writer = Writer()
    prior = {"scenarios": "E3T-x", "scenarios_extra": "E3T-y"}
    with pytest.raises(RunBroken, match="run_conditions"):
        go_arm(tmp_path, writer, prior, run_conditions=" ")
    with pytest.raises(RunBroken, match="not under"):
        go_arm(tmp_path, writer, prior)
    (tmp_path / "stray.txt").write_text("x", encoding="utf-8")
    with pytest.raises(RunBroken, match="uncommitted"):
        go_arm(tmp_path, writer, prior)
    assert writer.calls == [] and not (tmp_path / "results/runs").exists() and not (tmp_path / arm.PARTIAL_REL).exists()
