"""E4: trained consistency model. Tiny models, fixtures and invented text only; no GPU, no network."""
import pytest

from contextlens.eval import metrics as M
from contextlens.eval import series
from contextlens.eval.check_run import recompute_metrics
from contextlens.replay.runner import Prediction
from contextlens.schemas import ButtonRequest, Conversation, Scenario, Turn


def _sc(sid: str, gold: str | None, category: str = "contradiction") -> Scenario:
    turns = [Turn(id=f"{sid}:t{i}", conversation_id=sid, seq=i, speaker="A", text=f"remark {i} in {sid}",
                  start_ms=i * 1000, end_ms=i * 1000 + 900, source="synthetic") for i in range(3)]
    return Scenario(id=sid, conversation=Conversation(id=sid, turns=turns, source="synthetic", split="dev"),
                    request=ButtonRequest(conversation_id=sid, as_of_seq=2), category=category,
                    gold_evidence_turn_ids=[], gold_label=gold, gold_answer=None, origin="synthetic")  # type: ignore[arg-type]


# ---- calibration ----

def test_ece_is_the_weighted_gap_between_confidence_and_accuracy() -> None:
    assert M.ece([True, True, False, False], [0.9] * 4) == pytest.approx(0.4)
    assert M.ece([True] * 5, [1.0] * 5) == 0.0  # a probability of exactly 1 lands in the last bin
    # two bins: 0.15 -> acc 0 (gap .15, weight .5); 0.85 -> acc 1 (gap .15, weight .5)
    assert M.ece([False, True], [0.15, 0.85]) == pytest.approx(0.15)
    assert M.ece([], []) is None
    with pytest.raises(ValueError):
        M.ece([True], [0.5, 0.5])


def test_label_rows_carry_the_prediction_without_a_card() -> None:
    row = Prediction(scenario_id="x", label="contradiction", probability=0.7)
    assert row.card is None and row.raw_output is None
    with pytest.raises(ValueError):
        Prediction(scenario_id="x")
    with pytest.raises(ValueError):
        Prediction(scenario_id="x", label="update", probability=1.5)


def test_ece_and_label_metrics_recompute_from_label_rows() -> None:
    scs = [_sc("a", "contradiction"), _sc("b", "consistent"), _sc("c", "update"), _sc("d", None)]
    preds = [Prediction(scenario_id="a", label="contradiction", probability=0.9),
             Prediction(scenario_id="b", label="contradiction", probability=0.8),
             Prediction(scenario_id="c", label="update", probability=0.6),
             Prediction(scenario_id="d", label="consistent", probability=0.99)]
    m = recompute_metrics(scs, preds)
    # the unlabelled case d is not scored: correct = a, c; confidences .9 .8 .6
    assert m["ece"] == pytest.approx(M.ece([True, False, True], [0.9, 0.8, 0.6]))
    assert m["false_alert_rate"] == pytest.approx(0.5)  # of b and c, only b is called a contradiction
    assert "ece[category=contradiction]" in m


def test_ece_is_left_out_when_a_scored_row_has_no_probability() -> None:
    scs = [_sc("a", "contradiction"), _sc("b", "consistent")]
    preds = [Prediction(scenario_id="a", label="contradiction", probability=0.9),
             Prediction(scenario_id="b", label="consistent")]
    assert "ece" not in recompute_metrics(scs, preds)
    assert "ece" not in recompute_metrics(scs, [Prediction(scenario_id=p.scenario_id, label=p.label) for p in preds])


def test_ece_has_an_interval_from_resampled_conversations() -> None:
    scs = [_sc(f"s{i}", "contradiction" if i % 2 else "consistent") for i in range(12)]
    preds = [Prediction(scenario_id=s.id, label="contradiction", probability=0.5 + (i % 5) / 10) for i, s in enumerate(scs)]
    series.RESAMPLES = 100
    try:
        lo, hi = series.interval(list(zip(scs, preds)), "ece")  # type: ignore[misc]
    finally:
        series.RESAMPLES = 10_000
    assert 0 <= lo <= hi <= 1


# ---- the train-only scenario set ----

from pathlib import Path  # noqa: E402

from contextlens.data import train_scenarios as ts  # noqa: E402
from contextlens.data.scenarios import load_scenarios  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_training_set_covers_every_label_and_card_mode_and_is_synthetic() -> None:
    scs = ts.load_train_scenarios(ROOT / "datasets/scenarios_train")
    c = ts.counts(scs)
    assert set(c["by_label"]) >= {"consistent", "update", "contradiction", "insufficient"}
    assert set(c["by_mode"]) == {"recall", "clarify", "respond"}
    assert c["abstention_cases"]["insufficient"] >= 5 and c["injected_turn_cases"]["with_injection"] >= 5
    assert all(s.origin == "synthetic" and s.conversation.split == "train" for s in scs)
    assert c["cases"]["total"] == c["cases"]["conversations"] == len({s.id for s in scs})
    assert scs == ts.build_train_scenarios()  # the committed file is what the code builds


def test_training_set_shares_nothing_with_dev_or_test_scenarios() -> None:
    train = ts.load_train_scenarios(ROOT / "datasets/scenarios_train")
    held = [s for sp in ("dev", "test") for s in load_scenarios(sp, ROOT / "datasets/scenarios")]
    assert ts.overlap_problems(train, held) == []
    assert not {s.conversation.id for s in train} & {s.conversation.id for s in held}


def test_overlap_check_catches_an_id_a_turn_and_an_eight_word_run() -> None:
    train = ts.build_train_scenarios()
    held = [train[0].model_copy(update={"id": "x"})]
    problems = ts.overlap_problems(train[:1], held)
    assert any("repeats a held-out turn" in p for p in problems) and any("8-word runs" in p for p in problems)
    other = train[1]
    reworded = other.model_copy(update={"id": "y", "conversation": other.conversation.model_copy(update={"id": "y"})})
    assert any("also in" in p for p in ts.overlap_problems([other], [other])) and ts.overlap_problems([other], [reworded])


def test_training_cases_are_not_near_duplicates_of_each_other() -> None:
    scs = ts.build_train_scenarios()
    seen: dict[str, str] = {}
    for s in scs:
        for run in ts.word_runs([t.text for t in s.conversation.turns], 6):
            assert seen.setdefault(run, s.id) == s.id, f"{run!r} appears in {seen[run]} and {s.id}"


def test_deadline_update_pair_is_in_eval_set() -> None:
    from contextlens.consistency.data import build_pairs

    pairs = build_pairs("dev", ROOT / "tests/fixtures/data", ROOT / "datasets/scenarios")
    hit = [p for p in pairs if p.source == "scenarios" and p.label == "update"
           and p.statement == "We moved the deadline to Wednesday"]
    assert len(hit) == 1 and hit[0].earlier == ["The deadline is Monday"]


# ---- model, checkpoint, checker, training on a tiny random model ----

import json  # noqa: E402

import torch  # noqa: E402

from contextlens.consistency import model as cm  # noqa: E402
from contextlens.consistency import train as ct  # noqa: E402

WORDS = "the a is was deadline monday wednesday moved we never always it not budget ten zero ok fine".split()


def _tiny(tmp_path: Path):
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(dict.fromkeys(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "a", "b", "c", "d", *WORDS])), encoding="utf-8")
    tok = BertTokenizer(str(vocab))
    cfg = BertConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=1, num_attention_heads=2, initializer_range=0.2,
                     intermediate_size=64, max_position_embeddings=128, hidden_dropout_prob=0.0,
                     attention_probs_dropout_prob=0.0, num_labels=len(cm.LABELS),
                     id2label=dict(enumerate(cm.LABELS)), label2id={n: i for i, n in enumerate(cm.LABELS)})
    torch.manual_seed(0)
    return BertForSequenceClassification(cfg), tok


def _ex(label: str, statement: str, earlier: str = "the deadline is monday", group: str = "g") -> cm.Example:
    return cm.Example(statement=statement, speaker="a", earlier=[("b", earlier)], label=label, group=group, source="x")


EXAMPLES = [_ex("update", "we moved the deadline to wednesday"), _ex("contradiction", "the deadline is never"),
            _ex("consistent", "ok fine"), _ex("insufficient", "it is a budget")] * 2


def test_train_one_step_on_tiny_model_reduces_loss(tmp_path: Path) -> None:
    model, tok = _tiny(tmp_path)
    batch = ct.collate(tok, EXAMPLES)
    weights = torch.tensor(ct.class_weights([e.label for e in EXAMPLES]))
    model.eval()
    before = ct.batch_loss(model, batch, weights).item()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    reported = ct.train_step(model, batch, opt, weights)
    model.eval()
    after = ct.batch_loss(model, batch, weights).item()
    assert reported == pytest.approx(before, rel=1e-4) and after < before


def test_class_weights_are_balanced_and_zero_for_absent_labels() -> None:
    w = ct.class_weights(["consistent"] * 6 + ["contradiction"] * 2)
    assert w[0] == pytest.approx(8 / (2 * 6)) and w[2] == pytest.approx(8 / (2 * 2)) and w[1] == w[3] == 0.0


def test_checkpoint_saves_label_map(tmp_path: Path) -> None:
    model, tok = _tiny(tmp_path)
    out = tmp_path / "ckpt"
    cm.save_checkpoint(model, tok, out, {"lr": 2e-5})
    saved = json.loads((out / "label_map.json").read_text(encoding="utf-8"))
    assert saved["labels"] == list(cm.LABELS) and saved["meta"] == {"lr": 2e-5}
    loaded, _, labels = cm.load_checkpoint(out)
    assert labels == cm.LABELS and loaded.config.id2label == dict(enumerate(cm.LABELS))
    (out / "label_map.json").write_text(json.dumps({"labels": ["x", "y"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="label map"):
        cm.load_checkpoint(out)


def test_trained_checker_abstains_below_threshold(tmp_path: Path) -> None:
    model, tok = _tiny(tmp_path)
    turns = [Turn(id="t0", conversation_id="c", seq=0, speaker="b", text="the deadline is monday", start_ms=0, end_ms=1, source="synthetic")]
    stmt = Turn(id="t1", conversation_id="c", seq=1, speaker="a", text="we moved the deadline to wednesday", start_ms=2, end_ms=3, source="synthetic")
    sure = cm.TrainedChecker(model, tok, cm.LABELS, threshold=0.0)
    label, p = sure.check(stmt, turns)
    assert label in cm.LABELS and 0.25 <= p <= 1.0
    probs = cm.predict_proba(model, tok, [cm.example_from_turns(stmt, turns)])[0]
    assert label == cm.LABELS[max(range(4), key=probs.__getitem__)] and p == pytest.approx(max(probs))
    unsure = cm.TrainedChecker(model, tok, cm.LABELS, threshold=0.99)  # a random model is never that sure
    label, p = unsure.check(stmt, turns)
    assert label == "insufficient" and p == pytest.approx(probs[cm.LABELS.index("insufficient")])


def test_fit_stops_early_when_dev_macro_f1_does_not_improve(tmp_path: Path) -> None:
    model, tok = _tiny(tmp_path)
    logged: list[tuple[dict, int]] = []
    res = ct.fit(model, tok, EXAMPLES, EXAMPLES, lr=0.0, epochs=5, batch_size=4, patience=1,
                 out_dir=tmp_path / "best", log=lambda m, step: logged.append((m, step)))
    assert [h["epoch"] for h in res.history] == [1, 2] and res.best_epoch == 1
    assert (tmp_path / "best" / "label_map.json").is_file() and [s for _, s in logged] == [1, 2]
    assert {"train_loss", "dev_macro_f1"} <= set(logged[0][0])


def test_fit_learns_a_tiny_task(tmp_path: Path) -> None:
    model, tok = _tiny(tmp_path)
    res = ct.fit(model, tok, EXAMPLES * 4, EXAMPLES, lr=5e-3, epochs=12, batch_size=4, patience=12)
    assert res.best_macro_f1 > 0.9


def test_mlflow_logging_goes_to_a_local_store(tmp_path: Path) -> None:
    import mlflow

    with ct.mlflow_run(tmp_path / "mlruns", "e4-test", {"lr": 1e-3}) as log:
        log({"dev_macro_f1": 0.5}, 1)
    found = mlflow.MlflowClient(tracking_uri=(tmp_path / "mlruns").as_uri()).search_runs(
        [mlflow.MlflowClient(tracking_uri=(tmp_path / "mlruns").as_uri()).get_experiment_by_name("e4-test").experiment_id])
    assert found[0].data.params["lr"] == "0.001" and found[0].data.metrics["dev_macro_f1"] == 0.5
