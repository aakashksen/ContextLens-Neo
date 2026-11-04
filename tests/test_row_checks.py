"""check_run does not trust the predictions file: rows must be uniform, complete, and cite only the past."""
import json
import subprocess
from pathlib import Path

import pytest

from contextlens.eval import claims
from contextlens.eval.check_run import check, recompute_metrics
from contextlens.eval.records import split_hash, write_run, write_scenarios_manifest
from contextlens.replay.runner import Prediction
from contextlens.schemas import (
    ButtonRequest,
    Card,
    Conversation,
    EvidenceRef,
    Finding,
    Scenario,
    Turn,
)
from tests.test_records import set_h2

IDS = ["d1", "d2", "d3", "d4"]


def _scenario(sid: str) -> Scenario:
    turns = [
        Turn(id=f"{sid}-t{i}", conversation_id=sid, seq=i, speaker="A", text=f"w{i}",
             start_ms=i * 1000, end_ms=i * 1000 + 900, source="synthetic")
        for i in range(4)
    ]
    return Scenario(
        id=sid, conversation=Conversation(id=sid, turns=turns, source="test"),
        request=ButtonRequest(conversation_id=sid, as_of_seq=2), category="c",  # t3 is in the future
        gold_evidence_turn_ids=[f"{sid}-t1"], gold_label="update", gold_answer=None, origin="synthetic",
    )


SCENARIOS = [_scenario(i) for i in IDS]


def _card(sid: str, turn: int = 1) -> Card:
    ev = EvidenceRef(turn_id=f"{sid}-t{turn}", quote="q", speaker="A", start_ms=0)
    f = Finding(agent="a", kind="consistency", statement="s", label="update", confidence=0.5, evidence=[ev])
    return Card(mode="clarify", status="answered", findings=[f])


def _row(sid: str, **kw) -> Prediction:
    base: dict = {"card": _card(sid), "claim_supported": [True], "stage_latency_ms": {"total": 5}}
    return Prediction(scenario_id=sid, **{**base, **kw})


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=root, check=True,
                   capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    d = tmp_path / "datasets/scenarios"
    d.mkdir(parents=True)
    (d / "dev.jsonl").write_text("".join(s.model_dump_json() + "\n" for s in SCENARIOS), encoding="utf-8")
    write_scenarios_manifest(tmp_path)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def _save(root: Path, rows: list[Prediction], metrics: dict | None = None, config: dict | None = None) -> None:
    metrics = metrics if metrics is not None else recompute_metrics(SCENARIOS, rows)
    write_run("E2", "dev", config or {"labels_reviewed": True}, {"llm": "qwen-real"}, {"p": "v1"},
              {"main": metrics}, {}, [{"setting": "a"}], {"main": rows}, split_hash("dev", root), root=root)


def _problems(root: Path) -> list[str]:
    return check("E2", root=root)


def test_complete_card_rows_pass(repo: Path) -> None:
    rows = [_row(i, raw_output=_card(i).model_dump_json(), ranked_turn_ids=[f"{i}-t1"]) for i in IDS]
    _save(repo, rows, {"schema_valid_rate": 1.0, "mrr": 1.0, "recall@1": 1.0, "latency_p50_ms": 5.0})
    assert _problems(repo) == []


@pytest.mark.parametrize("ranked", [["d1-t3", "d1-t1"], ["d1-t1", "bogus"], ["d1-t0", "d1-t1", "d1-t3"]])
def test_ranked_ids_past_as_of_or_unknown_fail(repo: Path, ranked: list[str]) -> None:
    rows = [Prediction(scenario_id=i, ranked_turn_ids=[f"{i}-t1"]) for i in IDS]
    rows[0] = Prediction(scenario_id="d1", ranked_turn_ids=ranked)
    _save(repo, rows, {"mrr": 1.0})
    assert any("d1" in p and "as_of_seq" in p for p in _problems(repo))


def test_ranked_ids_from_another_conversation_fail(repo: Path) -> None:
    rows = [Prediction(scenario_id=i, ranked_turn_ids=[f"{i}-t1"]) for i in IDS]
    rows[0] = Prediction(scenario_id="d1", ranked_turn_ids=["d2-t1"])
    _save(repo, rows, {"mrr": 0.75})
    assert any("as_of_seq" in p for p in _problems(repo))


def test_card_evidence_past_as_of_fails(repo: Path) -> None:
    rows = [_row(i) for i in IDS]
    rows[1] = _row("d2", card=_card("d2", turn=3))
    _save(repo, rows, {"speaker_accuracy": 1.0})
    assert any("d2" in p and "as_of_seq" in p for p in _problems(repo))


def test_half_the_rows_without_cards_fails(repo: Path) -> None:
    rows = [_row("d1"), _row("d2"), Prediction(scenario_id="d3", ranked_turn_ids=[]),
            Prediction(scenario_id="d4", ranked_turn_ids=[])]
    _save(repo, rows, {"macro_f1": 0.0})
    assert any("rows are mixed" in p for p in _problems(repo))


def test_an_error_card_counts_as_a_card(repo: Path) -> None:
    err = Card(mode="recall", status="abstained", findings=[], missing=["error: Boom"])
    rows = [_row("d1"), _row("d2"), _row("d3"), Prediction(scenario_id="d4", card=err, errored=True)]
    _save(repo, rows)
    assert not any("mixed" in p for p in _problems(repo))


def test_schema_valid_rate_needs_raw_output_on_every_row(repo: Path) -> None:
    rows = [_row(i) for i in IDS]
    rows[0] = _row("d1", raw_output=_card("d1").model_dump_json())
    _save(repo, rows, {"schema_valid_rate": 1.0})
    assert any("schema_valid_rate" in p and "cannot be recomputed" in p for p in _problems(repo))


def test_ranked_metrics_need_ranked_ids_on_every_row(repo: Path) -> None:
    rows = [_row(i) for i in IDS]
    rows[0] = _row("d1", ranked_turn_ids=["d1-t1"])
    _save(repo, rows, {"mrr": 1.0, "recall@1": 1.0})
    problems = _problems(repo)
    assert any("mrr" in p and "cannot be recomputed" in p for p in problems)
    assert any("recall@1" in p and "cannot be recomputed" in p for p in problems)


def test_stage_latency_needs_the_stage_on_every_row(repo: Path) -> None:
    rows = [_row(i) for i in IDS]
    rows[0] = _row("d1", stage_latency_ms={"total": 5, "retrieve": 2})
    _save(repo, rows, {"latency_p50_ms[stage=retrieve]": 2.0})
    assert any("retrieve" in p and "cannot be recomputed" in p for p in _problems(repo))


def test_total_latency_on_some_rows_only_is_not_recomputable(repo: Path) -> None:
    rows = [_row(i) for i in IDS]
    rows[0] = _row("d1", stage_latency_ms={})
    _save(repo, rows, {"latency_p50_ms": 5.0})
    assert any("latency_p50_ms" in p and "cannot be recomputed" in p for p in _problems(repo))


def test_grounded_rate_needs_a_judgement_on_every_card(repo: Path) -> None:
    rows = [_row(i) for i in IDS]
    rows[0] = _row("d1", claim_supported=None)
    _save(repo, rows, {"grounded_rate": 1.0})
    assert any("grounded_rate" in p and "cannot be recomputed" in p for p in _problems(repo))


JUDGE = {"model": "qwen3-judge", "rubric_version": "r1", "same_model_and_prompt_as_writer": False}


def _judged(repo: Path, config: dict) -> list[str]:
    rows = [_row(i) for i in IDS]
    _save(repo, rows, config={"labels_reviewed": True, **config})
    return _problems(repo)


def test_judged_metrics_need_a_named_judge(repo: Path) -> None:
    assert any("judge" in p for p in _judged(repo, {}))


@pytest.mark.parametrize("judge", [
    {**JUDGE, "model": "fake-judge"}, {**JUDGE, "model": "my_Stub"}, {**JUDGE, "model": ""},
    {**JUDGE, "rubric_version": ""}, {"model": "m", "rubric_version": "r1"},
    {**JUDGE, "same_model_and_prompt_as_writer": "no"},
])
def test_bad_judge_config_fails(repo: Path, judge: dict) -> None:
    assert any("judge" in p for p in _judged(repo, {"judge": judge}))


def test_named_judge_passes(repo: Path) -> None:
    assert _judged(repo, {"judge": JUDGE}) == []
    assert _judged(repo, {"judge": {**JUDGE, "same_model_and_prompt_as_writer": True}}) == []


def _claim(repo: Path, config: dict, metric: str = "grounded_rate") -> list[str]:
    set_h2(repo, "done")  # labels_reviewed comes from H2's signed state, not from the config
    _save(repo, [_row(i) for i in IDS], config={"judge": JUDGE, **config})
    rec = json.loads(next((repo / "results/runs").glob("E2-*Z.json")).read_text(encoding="utf-8"))
    (repo / "results").mkdir(exist_ok=True)
    (repo / "results/claims.json").write_text(json.dumps({"claims": [
        {"id": "c1", "run_id": rec["run_id"], "metric": metric, "value": rec["metrics"]["main"][metric]}]}),
        encoding="utf-8")
    return claims.check_claims(repo)


REVIEW = {"agreement_rate": 0.9, "sample_size": 20, "sample": "random"}


def test_claim_on_grounded_rate_needs_a_person_review(repo: Path) -> None:
    assert any("judge_review" in p for p in _claim(repo, {}))


def test_claim_with_a_recorded_review_passes(repo: Path) -> None:
    assert _claim(repo, {"judge_review": REVIEW}) == []


@pytest.mark.parametrize("review", [
    {**REVIEW, "agreement_rate": 1.5}, {**REVIEW, "agreement_rate": "high"}, {**REVIEW, "sample_size": 0},
    {**REVIEW, "sample": "hand-picked"}, {"agreement_rate": 0.9},
])
def test_claim_with_an_invalid_review_fails(repo: Path, review: dict) -> None:
    assert any("judge_review" in p for p in _claim(repo, {"judge_review": review}))


def test_claim_on_unjudged_metric_needs_no_review(repo: Path) -> None:
    assert _claim(repo, {}, metric="macro_f1") == []
