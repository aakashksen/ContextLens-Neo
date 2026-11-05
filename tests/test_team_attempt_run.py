"""The runner of entry 48: three systems on both dev sources, one valid record per source, resumable. No real model."""

import json
import shutil
import threading
from pathlib import Path

import pytest

from contextlens.agents.consistency_check import PromptedChecker
from contextlens.agents.graph import TeamAssistant
from contextlens.agents.single import SingleAgent
from contextlens.data.extra_cases import SHAPES
from contextlens.data.scenarios import load_scenarios
from contextlens.eval import series
from contextlens.eval.check_run import check
from contextlens.eval.records import head_commit
from contextlens.experiments import e2_context as e2
from contextlens.experiments import team_attempt as ta
from contextlens.experiments.e2_context import RunBroken
from contextlens.llm.client import LLMError
from contextlens.llm.ollama_ctl import OllamaCtl
from contextlens.memory.extract import MemoryStore
from contextlens.replay.runner import Prediction
from contextlens.schemas import Card, EvidenceRef, Finding
from tests.test_e1 import HashEmbedder
from tests.test_e2 import NAMES, RETRIEVAL, Scripted, Wire, _git
from tests.test_e3 import AgentScripted

REAL = Path(__file__).resolve().parents[1]
EXTRA = load_scenarios("dev", REAL / "datasets/scenarios_extra")


def test_the_new_cases_split_into_the_24_weaker_pairs_and_the_other_30():
    keys = {ta.pair_key(s.id)[0] for s in EXTRA}
    weak = {k for k in keys if ta.pair_shape(k) in ta.WEAK_SHAPES}
    assert len(keys) == 54 and len(weak) == 24 and len(keys - weak) == 30
    assert {ta.pair_shape(k) for k in keys} == set(SHAPES)
    assert ta.pair_key("syn-xtra-promise-3-ins") == ("syn-xtra-promise-3", True)
    assert ta.pair_key("syn-con-00") is None


def row(sid, status):
    found = [Finding(agent="a", kind="recall", statement="s", confidence=0.5,
                     evidence=[EvidenceRef(turn_id="t", quote="q", speaker="x", start_ms=0)])] * (status == "answered")
    card = Card(mode="recall", status=status, findings=found, missing=[])
    return Prediction(scenario_id=sid, card=card)


def test_twin_outcomes_are_the_outcomes_the_selection_metrics_count():
    key = ta.pair_key(EXTRA[0].id)[0]
    scs = [s for s in EXTRA if s.id.startswith(key + "-")]
    assert len(scs) == 2
    for ins_status, ans_status in (("abstained", "answered"), ("answered", "answered"), ("abstained", "abstained")):
        rows = [row(s.id, ins_status if s.id.endswith("-ins") else ans_status) for s in scs]
        got = ta.twin_outcomes(scs, rows)[key]
        pairs = list(zip(scs, rows, strict=True))
        metric = dict(series.units(pairs, "abstain_when_insufficient") or []) | dict(series.units(pairs, "answer_when_answerable") or [])
        assert got == (bool(metric[key + "-ins"]), bool(metric[key + "-ans"]))


def test_the_pair_table_names_each_way_a_pair_can_fall():
    out = {"a": (True, True), "b": (False, True), "c": (True, False), "d": (False, False)}
    assert ta.twin_table(out) == {"pairs": 4, "both_right": 1, "answers_both": 1, "abstains_both": 1, "reversed": 1,
                                  "pair_right_rate": 0.25}


def test_better_or_worse_needs_the_interval_and_mcnemar_to_agree():
    assert ta.pair_verdict(0.2, 0.1, 0.3, 0.01) == "better" and ta.pair_verdict(-0.2, -0.3, -0.1, 0.01) == "worse"
    assert ta.pair_verdict(0.2, 0.1, 0.3, 0.2) == "no clear difference"  # the interval alone is not enough
    assert ta.pair_verdict(0.2, -0.1, 0.3, 0.01) == "no clear difference"
    assert ta.pair_verdict(0.2, 0.1, 0.3, None) == "no clear difference"
    assert ta.pair_verdict(-5, -9, -1, None, binary=False, higher_is_better=False) == "better"


def sets(old, new, both, worse=()):
    def one(score):
        return {"scores": {"single": 0.7, "team": score[0], "team_changed": score[1]}, "comparisons": []}

    out = {"old": one(old), "new": one(new), "both": one(both)}
    out["both"]["comparisons"] = [
        {"a": "team_changed", "b": "single", "metric": m, "diff": -0.3, "ci": [-0.4, -0.2], "mcnemar_p": 0.001} for m in worse]
    return out


def test_single_stays_selected_unless_a_challenger_wins_on_both_sets():
    assert ta.select(sets((0.6, 0.9), (0.6, 0.6), (0.6, 0.8)))["selected"] == "single"  # wins on old only
    assert ta.select(sets((0.6, 0.9), (0.6, 0.9), (0.6, 0.9)))["selected"] == "team_changed"
    assert ta.select(sets((0.6, 0.7), (0.6, 0.9), (0.6, 0.9)))["selected"] == "single"  # a tie is not a win
    got = ta.select(sets((0.6, 0.9), (0.6, 0.9), (0.6, 0.9), worse=("answer_when_answerable",)))
    assert got["selected"] == "single" and got["challengers"]["team_changed"]["clearly_worse_on"] == ["answer_when_answerable"]
    assert ta.select(sets((0.6, None), (0.6, 0.9), (0.6, 0.9)))["challengers"]["team_changed"]["qualifies"] is False


def test_the_changed_team_is_the_team_with_the_one_switch_on_and_the_other_two_are_not():
    args = (EXTRA[0], Scripted(), HashEmbedder(), RETRIEVAL, MemoryStore())
    single, team, changed = (ta.make_assistant(n, *args) for n in ta.SYSTEMS)
    assert isinstance(single, SingleAgent) and isinstance(team, TeamAssistant) and isinstance(changed, TeamAssistant)
    assert (team.memory_gap_abstains, changed.memory_gap_abstains) == (False, True)
    assert changed.deadline_s == team.deadline_s == 8.0 and ta.DEADLINE_S == 8.0


def test_only_the_shipping_arm_has_a_consistency_checker_and_the_others_are_as_entry_48_ran_them():
    from contextlens.api import production

    args = (EXTRA[0], Scripted(), HashEmbedder(), RETRIEVAL, MemoryStore())
    built = {n: ta.make_assistant(n, *args) for n in ta.RUN_SYSTEMS}
    assert ta.SYSTEMS == ("single", "team", "team_changed") and ta.RUN_SYSTEMS == (*ta.SYSTEMS, "team_shipping")
    assert all(built[n].tools.checker is None for n in ta.SYSTEMS)
    ship = built["team_shipping"]
    assert isinstance(ship, TeamAssistant) and isinstance(ship.tools.checker, PromptedChecker)
    assert ship.memory_gap_abstains is False and ship.deadline_s == built["team"].deadline_s == 8.0
    assert ship._specs["consistency"][1:] != built["team"]._specs["consistency"][1:]  # the checker changes prompt and tools
    assert production.MODEL == e2.WRITER_MODEL  # production's llms["consistency"] is this model


# ---- a whole run on the real dev files, with scripted models ----


class Writer(AgentScripted):
    """The memory agent finds nothing on the insufficient twins; the consistency agent always writes a finding.
    Memory extraction fails for `fail_memory_for` (a case id)."""

    fail_memory_for: str | None = None

    def __init__(self, **kw):
        super().__init__(**kw)
        self.local = threading.local()

    def _raw(self, messages, schema, max_tokens, temperature, timeout_s):
        self.local.system = messages[0]["content"]
        return super()._raw(messages, schema, max_tokens, temperature, timeout_s)

    def reply(self, name, user):
        if name == "_Out" and self.fail_memory_for and f"{self.fail_memory_for}:" in user:
            raise LLMError("request to the model server failed")
        if name == "_Step" and "memory agent" in self.local.system and "-ins:" in user:
            return {"tool": "final", "findings": []}
        return super().reply(name, user)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(series, "RESAMPLES", 100)  # the run and the check use the same count
    for rel in ("datasets/scenarios", "datasets/scenarios_extra", "datasets/splits"):
        shutil.copytree(REAL / rel, tmp_path / rel)
    (tmp_path / ".gitignore").write_text("results/partial/\n", encoding="utf-8")
    for args in (("init", "-q"), ("add", "-A"), ("commit", "-q", "-m", "init")):
        _git(tmp_path, *args)
    return tmp_path


def go(repo, writer=None, run_conditions="test"):
    return ta.run_attempt(root=repo, writer=writer or Writer(), judge_llm=Scripted("test-judge:1"), embedder=HashEmbedder(),
                          ctl=OllamaCtl(request=Wire()), names=NAMES, retrieval_config=RETRIEVAL,
                          run_conditions=run_conditions, say=lambda t: None)


def test_a_run_writes_one_valid_record_per_source_with_the_three_systems(repo):
    paths = go(repo)
    assert check(ta.EXPERIMENT, root=repo, min_n=100, split="dev") == []
    recs = {s: json.loads(p.read_text(encoding="utf-8")) for s, p in paths.items()}
    assert {s: r["n"] for s, r in recs.items()} == {"scenarios": 148, "scenarios_extra": 108}
    for source, rec in recs.items():
        assert rec["scenario_source"] == source and rec["split"] == "dev" and rec["supersedes"] == []
        assert set(rec["predictions"]) == set(ta.RUN_SYSTEMS)
        cfg = rec["config"]
        assert (cfg["attempt"], cfg["decision_entry"], cfg["deadline_s"]) == (ta.ATTEMPT, 40, 8.0) and cfg["hypothesis"]
        assert set(cfg["row_report"]) == set(ta.RUN_SYSTEMS) and "findings_dropped_by_reason" in cfg["row_report"]["team_changed"]
        assert "cards_cut_by_the_deadline" in cfg["row_report"]["single"] and "cards_with_llm_timeout" in cfg["row_report"]["team"]
        assert set(cfg["sets"]) == {"old", "new", "both"} and set(cfg["slices"]) == {"new_weak_24", "new_other_30"}
        assert (cfg["slices"]["new_weak_24"]["cases"], cfg["slices"]["new_other_30"]["cases"]) == (48, 60)
        assert cfg["sets"]["both"]["cases"] == 256 and cfg["selection"]["selected"] in ("single", "team", "team_changed")
        pairs = {(c["a"], c["b"], c["metric"]) for c in rec["comparisons"]}
        assert ("team_changed", "single", "abstain_when_insufficient") in pairs
        assert ("team_changed", "team", "answer_when_answerable") in pairs
        assert "answer rate" in " ".join(cfg["notes"])
    twins = recs["scenarios_extra"]["config"]["twins"]
    assert twins["all"]["by_system"]["team"]["pairs"] == 54 and twins["weak_24"]["by_system"]["team"]["pairs"] == 24
    assert twins["other_30"]["by_system"]["team"]["pairs"] == 30 and set(twins["by_shape"]) == set(SHAPES)
    # the change shows in the rows only for the changed team
    rec = recs["scenarios_extra"]
    rows = {n: [Prediction.model_validate_json(x) for x in (repo / p).read_text(encoding="utf-8").splitlines()]
            for n, p in rec["predictions"].items()}
    ins = {n: [p for p in r if p.scenario_id.endswith("-ins")] for n, r in rows.items()}
    ans = {n: [p for p in r if p.scenario_id.endswith("-ans")] for n, r in rows.items()}
    assert all(p.card.status == "abstained" for p in ins["team_changed"])
    assert all(any(d.reason == "memory_empty" for d in p.drops) for p in ins["team_changed"])
    assert not any(d.reason == "memory_empty" for r in (rows["team"] + rows["single"]) for d in r.drops)
    assert not any(p.card.status == "abstained" for p in ans["team_changed"])
    assert all(p.card.status != "abstained" for p in ins["team"])  # the unchanged team is outvoted by the consistency finding
    assert rec["config"]["row_report"]["team_changed"]["findings_dropped_by_reason"].get("memory_empty") == 54
    assert not any(d.reason == "memory_empty" for d in sum((p.drops for p in rows["team_shipping"]), []))


def test_the_shipping_arm_is_reported_apart_and_leaves_entry_48s_numbers_as_they_were(repo):
    paths = go(repo)
    recs = {s: json.loads(p.read_text(encoding="utf-8")) for s, p in paths.items()}
    for rec in recs.values():
        cfg = rec["config"]
        assert cfg["entry_48_systems"] == list(ta.SYSTEMS) and cfg["systems"] == list(ta.RUN_SYSTEMS)
        assert all(set(cfg["sets"][k]["scores"]) == set(ta.SYSTEMS) for k in ta.SETS)  # entry 48's block has no shipping arm
        assert set(cfg["selection"]["challengers"]) == set(ta.CHALLENGERS)
        ship = cfg["team_shipping"]
        assert ship["decision_entry"] == 49 and ship["consistency_checker"]["class"] == "PromptedChecker"
        assert ship["consistency_checker"]["model"] == NAMES.writer
        assert set(ship["sets"]) == set(ta.SETS) and set(ship["slices"]) == {"new_weak_24", "new_other_30"}
        assert set(ship["scores"]) == {*ta.SETS, *ta.SLICES} and all(set(v) == set(ta.RUN_SYSTEMS) for v in ship["scores"].values())
        assert (ship["slices"]["new_weak_24"]["cases"], ship["slices"]["new_other_30"]["cases"]) == (48, 60)
        sig = ship["stop_signal"]
        assert isinstance(sig["team_shipping_beats_single_on_dev"], bool)
        assert sig["adding_it_would_change_the_selection"] == (sig["selected_with_team_shipping"] != sig["selected_by_entry_48"])
        assert sig["selected_by_entry_48"] == cfg["selection"]["selected"]
        pairs = {(c["a"], c["b"]) for c in rec["comparisons"]}
        assert ("team_shipping", "single") in pairs and ("team_shipping", "team") in pairs
    twins = recs["scenarios_extra"]["config"]["team_shipping"]["twins"]
    assert twins["weak_24"]["by_system"]["team_shipping"]["pairs"] == 24 and twins["other_30"]["by_system"]["team_shipping"]["pairs"] == 30
    assert any((c["a"], c["b"]) == ("team_shipping", "single") for c in twins["all"]["comparisons"])
    # entry 48's blocks equal what the three systems alone give
    by_source = {}
    for source, rec in recs.items():
        scs = load_scenarios("dev", REAL / "datasets" / source)
        by_source[source] = (scs, {n: [Prediction.model_validate_json(x) for x in (repo / p).read_text(encoding="utf-8").splitlines()]
                                   for n, p in rec["predictions"].items()})
    again = ta.all_reports(by_source, {s: "" for s in by_source})
    cfg = recs["scenarios_extra"]["config"]
    assert again["selection"]["selected"] == cfg["selection"]["selected"]
    assert {k: v["scores"] for k, v in again["sets"].items()} == {k: v["scores"] for k, v in cfg["sets"].items()}


def test_a_failed_memory_build_is_listed_with_its_reason_and_no_text(repo):
    class Failing(Writer):
        fail_memory_for = EXTRA[0].id

    cfg = json.loads(go(repo, Failing())["scenarios_extra"].read_text(encoding="utf-8"))["config"]
    build = cfg["memory_build"]
    assert build["failed_ids"] == [EXTRA[0].id] and [r["id"] for r in build["failed_reasons"]] == [EXTRA[0].id]
    assert build["failed_reasons"][0]["reason"].startswith("LLMError: request to")


def test_a_dirty_tree_or_a_missing_run_condition_stops_before_any_model_call(repo):
    writer = Writer()
    with pytest.raises(RunBroken, match="run_conditions"):
        go(repo, writer, run_conditions=" ")
    (repo / "stray.txt").write_text("x", encoding="utf-8")
    with pytest.raises(RunBroken, match="uncommitted"):
        go(repo, writer)
    assert writer.calls == [] and not (repo / "results/runs").exists()
    with pytest.raises(SystemExit):
        ta.main([])


def test_an_interrupted_run_resumes_at_the_same_commit_without_redoing_saved_rows(repo):
    # 1400 calls: past single, team and team_changed on the old cases, partway through team_shipping
    with pytest.raises(RunBroken):
        go(repo, Writer(die_after=1400))
    saved = [json.loads(x) for x in (repo / ta.PARTIAL_REL).read_text(encoding="utf-8").splitlines()]
    assert saved and not (repo / "results/runs").exists()
    per_case: dict[str, int] = {}  # rows a resume may keep: a broken one is saved, but it is run again
    broken = set()
    for e in saved:
        sid = e["row"]["scenario_id"]
        if e2.row_state(Prediction.model_validate(e["row"])) == "broken":
            broken.add(sid)
            continue
        per_case[sid] = per_case.get(sid, 0) + 1
    finished = [sid for sid, n in per_case.items() if n == len(ta.RUN_SYSTEMS)]
    assert finished and broken  # the dead server left broken rows behind
    writer = Writer()
    paths = go(repo, writer)
    seen = json.dumps(writer.calls)
    assert len(paths) == 2 and (repo / ta.PARTIAL_REL).exists()  # entry 49: the partial rows are kept
    assert not any(f'"{sid}:' in seen or f"{sid}:t" in seen for sid in finished)  # a finished case was not run again
    assert all(f'"{sid}:' in seen or f"{sid}:t" in seen for sid in broken)  # a broken row was run again
    assert check(ta.EXPERIMENT, root=repo, min_n=100, split="dev") == []


def test_the_arm_of_entry_51_has_the_checker_and_the_switch_and_the_other_four_are_built_as_before():
    args = (EXTRA[0], Scripted(), HashEmbedder(), RETRIEVAL, MemoryStore())
    built = {n: ta.make_assistant(n, *args) for n in (*ta.RUN_SYSTEMS, ta.SHIPPING_CHANGED)}
    assert ta.SHIPPING_CHANGED == "team_shipping_changed" and ta.RUN_SYSTEMS == (*ta.SYSTEMS, "team_shipping")
    arm = built[ta.SHIPPING_CHANGED]
    assert isinstance(arm, TeamAssistant) and isinstance(arm.tools.checker, PromptedChecker) and arm.memory_gap_abstains is True
    assert arm.deadline_s == built["team"].deadline_s == 8.0
    assert arm._specs["consistency"][1:] == built["team_shipping"]._specs["consistency"][1:]  # same prompt and tools as team_shipping
    assert isinstance(built["single"], SingleAgent) and built["single"].tools.checker is None
    assert (built["team"].memory_gap_abstains, built["team"].tools.checker) == (False, None)
    assert (built["team_changed"].memory_gap_abstains, built["team_changed"].tools.checker) == (True, None)
    ship = built["team_shipping"]
    assert (ship.memory_gap_abstains, isinstance(ship.tools.checker, PromptedChecker)) == (False, True)


def test_finishing_a_run_keeps_the_partial_rows_and_a_later_commit_does_not_reuse_them(repo):
    go(repo)
    kept = [json.loads(x) for x in (repo / ta.PARTIAL_REL).read_text(encoding="utf-8").splitlines()]
    commit = head_commit(repo)
    assert len(kept) == (148 + 108) * len(ta.RUN_SYSTEMS) and {e["commit"] for e in kept} == {commit}
    again = Writer()  # same commit: every row is reused, so no case is run again and no memory is built
    assert len(go(repo, again)) == 2 and again.calls == [] and check(ta.EXPERIMENT, root=repo, min_n=100, split="dev") == []
    assert len((repo / ta.PARTIAL_REL).read_text(encoding="utf-8").splitlines()) == len(kept)  # nothing was added or removed
    (repo / "note.txt").write_text("a new commit", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "next")
    later = Writer()
    assert len(go(repo, later)) == 2 and later.calls  # the kept rows belong to the old commit
    seen = json.dumps(later.calls)
    assert all(f"{s.id}:" in seen for s in EXTRA[:3])
    assert {e["commit"] for e in (json.loads(x) for x in (repo / ta.PARTIAL_REL).read_text(encoding="utf-8").splitlines())} \
        == {commit, head_commit(repo)}
