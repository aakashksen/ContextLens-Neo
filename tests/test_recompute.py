"""Every metric an experiment saves must recompute from the saved prediction rows."""
import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from contextlens.eval.check_run import check, recompute_metrics
from contextlens.eval.records import split_hash, write_run, write_scenarios_manifest
from contextlens.replay.runner import Prediction
from contextlens.schemas import ButtonRequest, Card, Conversation, EvidenceRef, Finding, Scenario, Turn

IDS = ["d1", "d2", "d3", "d4"]


def _scenario(sid: str, origin: str = "synthetic", category: str = "c") -> Scenario:
    speakers = ["A", "B", "A"]
    turns = [
        Turn(id=f"{sid}-t{i}", conversation_id=sid, seq=i, speaker=speakers[i], text=f"w{i}",
             start_ms=i * 1000, end_ms=i * 1000 + 900, source="synthetic")
        for i in range(3)
    ]
    return Scenario(
        id=sid, conversation=Conversation(id=sid, turns=turns, source="test"),
        request=ButtonRequest(conversation_id=sid, as_of_seq=2), category=category,
        gold_evidence_turn_ids=[f"{sid}-t1"], gold_label="update", gold_answer=None,
        origin=origin,  # type: ignore[arg-type]
    )


def _scenarios() -> list[Scenario]:
    return [
        _scenario("d1", "synthetic", "x"), _scenario("d2", "synthetic", "y"),
        _scenario("d3", "dataset", "x"), _scenario("d4", "dataset", "y"),
    ]


def _finding(sid: str, turn: int, speaker: str) -> Finding:
    ev = EvidenceRef(turn_id=f"{sid}-t{turn}", quote="q", speaker=speaker, start_ms=0)
    return Finding(agent="a", kind="recall", statement="s", evidence=[ev], confidence=0.5)


def _card_pred(sid: str, speaker: str, supported: bool | None, tokens: int, cost: float,
               stages: dict[str, int] | None = None) -> Prediction:
    card = Card(mode="recall", status="answered", findings=[_finding(sid, 1, speaker)],
                tokens=tokens, cost_usd=cost)
    return Prediction(scenario_id=sid, card=card, claim_supported=None if supported is None else [supported],
                      stage_latency_ms=stages or {"total": 10})


def _preds() -> list[Prediction]:
    # t1 is spoken by B, so B is the right attribution
    return [
        _card_pred("d1", "B", True, 100, 0.01),
        _card_pred("d2", "A", True, 200, 0.02),
        _card_pred("d3", "B", False, 300, 0.03),
        _card_pred("d4", "B", True, 400, 0.04),
    ]


def test_card_metrics_recompute_by_hand() -> None:
    fresh = recompute_metrics(_scenarios(), _preds())
    assert fresh["speaker_accuracy"] == pytest.approx(3 / 4)
    assert fresh["tokens_per_card"] == pytest.approx(250)
    assert fresh["cost_usd_per_card"] == pytest.approx(0.025)
    assert fresh["grounded_rate"] == pytest.approx(3 / 4)
    assert fresh["unsupported_claim_rate"] == pytest.approx(1 / 4)


def test_judgement_changes_grounded_rate() -> None:
    preds = _preds()
    preds[0] = _card_pred("d1", "B", False, 100, 0.01)
    assert recompute_metrics(_scenarios(), preds)["grounded_rate"] == pytest.approx(2 / 4)


def test_grounded_metrics_absent_without_judgements() -> None:
    preds = _preds()
    preds[1] = _card_pred("d2", "A", None, 200, 0.02)
    fresh = recompute_metrics(_scenarios(), preds)
    assert "grounded_rate" not in fresh and "unsupported_claim_rate" not in fresh


def test_judgement_count_must_match_findings() -> None:
    preds = _preds()
    preds[1] = preds[1].model_copy(update={"claim_supported": [True, True]})
    assert "grounded_rate" not in recompute_metrics(_scenarios(), preds)


def test_retrieval_rows_without_a_card() -> None:
    ranked = {"d1": ["d1-t0", "d1-t1"], "d2": ["d2-t1"], "d3": ["d3-t0", "d3-t2"],
              "d4": ["d4-t2", "d4-t0", "d4-t1"]}
    preds = [Prediction(scenario_id=i, ranked_turn_ids=r) for i, r in ranked.items()]
    fresh = recompute_metrics(_scenarios(), preds, ks=(1, 2))
    assert fresh["recall@1"] == pytest.approx(1 / 4)
    assert fresh["recall@2"] == pytest.approx(2 / 4)
    assert fresh["mrr"] == pytest.approx((1 / 2 + 1 + 0 + 1 / 3) / 4)
    assert fresh["speaker_accuracy"] is None and fresh["grounded_rate"] is None
    assert fresh["macro_f1"] is None


def test_row_needs_a_card_or_ranked_ids() -> None:
    with pytest.raises(ValidationError):
        Prediction(scenario_id="d1")


def test_stage_and_group_breakdowns() -> None:
    preds = [_card_pred(i, "B", True, 100, 0.0, stages={"total": 10 * n, "retrieve": n})
             for n, i in enumerate(IDS, start=1)]
    fresh = recompute_metrics(_scenarios(), preds)
    assert fresh["latency_p50_ms"] == pytest.approx(25)
    assert fresh["latency_p50_ms[stage=retrieve]"] == pytest.approx(2.5)
    assert fresh["tokens_per_card[origin=dataset]"] == pytest.approx(100)
    assert fresh["speaker_accuracy[category=x]"] == 1.0


def test_schema_valid_rate_validates_raw_output() -> None:
    good = _preds()[0].card.model_dump_json()  # type: ignore[union-attr]
    preds = _preds()
    preds[0] = preds[0].model_copy(update={"raw_output": good})
    preds[1] = preds[1].model_copy(update={"raw_output": '{"mode": "nope"}'})
    assert recompute_metrics(_scenarios(), preds)["schema_valid_rate"] == pytest.approx(1 / 2)
    assert recompute_metrics(_scenarios(), _preds())["schema_valid_rate"] is None


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=root, check=True,
                   capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    d = tmp_path / "datasets/scenarios"
    d.mkdir(parents=True)
    (d / "dev.jsonl").write_text("".join(s.model_dump_json() + "\n" for s in _scenarios()), encoding="utf-8")
    write_scenarios_manifest(tmp_path)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def _run(root: Path, preds: list[Prediction], metrics: dict) -> None:
    write_run("E2", "dev", {"labels_reviewed": True}, {"llm": "qwen-real"}, {"p": "v1"}, metrics, {},
              [{"setting": "a"}], preds, split_hash("dev", root), root=root)


def test_check_run_accepts_recomputable_card_metrics(repo: Path) -> None:
    preds = _preds()
    _run(repo, preds, recompute_metrics(_scenarios(), preds))
    assert check("E2", root=repo) == []


def test_check_run_rejects_wrong_card_metric(repo: Path) -> None:
    _run(repo, _preds(), {"grounded_rate": 1.0})
    assert any("grounded_rate" in p for p in check("E2", root=repo))


def test_check_run_rejects_grounded_rate_without_judgements(repo: Path) -> None:
    preds = [_card_pred(i, "B", None, 1, 0.0) for i in IDS]
    _run(repo, preds, {"grounded_rate": 1.0})
    assert any("cannot be recomputed" in p for p in check("E2", root=repo))


def test_check_run_accepts_retrieval_record_and_none_metrics(repo: Path) -> None:
    preds = [Prediction(scenario_id=i, ranked_turn_ids=[f"{i}-t1"]) for i in IDS]
    _run(repo, preds, {"recall@1": 1.0, "recall@7": 1.0, "mrr": 1.0, "speaker_accuracy": None})
    assert check("E2", root=repo) == []


def test_check_run_rejects_zero_for_undefined_metric(repo: Path) -> None:
    preds = [Prediction(scenario_id=i, ranked_turn_ids=[f"{i}-t1"]) for i in IDS]
    _run(repo, preds, {"speaker_accuracy": 0.0})
    assert any("speaker_accuracy" in p for p in check("E2", root=repo))


def test_retrieval_record_still_needs_each_id_once(repo: Path) -> None:
    preds = [Prediction(scenario_id=i, ranked_turn_ids=[f"{i}-t1"]) for i in ["d1", "d1", "d3", "d4"]]
    _run(repo, preds, {"mrr": 1.0})
    assert any("prediction ids" in p for p in check("E2", root=repo))


def test_check_run_rejects_metric_with_no_recompute(repo: Path) -> None:
    _run(repo, _preds(), {"word_error_rate": 0.1})
    assert any("cannot be recomputed" in p for p in check("E2", root=repo))


def test_old_rows_without_new_fields_still_load() -> None:
    row = json.loads(_preds()[0].model_dump_json())
    for k in ("claim_supported", "ranked_turn_ids", "raw_output", "errored"):
        row.pop(k)
    assert Prediction.model_validate(row).card is not None
