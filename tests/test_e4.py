"""E4: trained consistency model. Tiny models, fixtures and invented text only; no GPU, no network."""
import time

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


# ---- the runner on a tiny model and a tiny invented fixture ----

import shutil  # noqa: E402
import subprocess  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from contextlens.agents.consistency_check import PromptedChecker  # noqa: E402
from contextlens.consistency.baselines import ZeroShotChecker  # noqa: E402
from contextlens.eval.check_run import check  # noqa: E402
from contextlens.eval.records import write_scenarios_manifest  # noqa: E402
from contextlens.experiments import e4_consistency as e4  # noqa: E402

FIXTURE_RAW = ROOT / "tests/fixtures/data"
DEV_CASES = [
    ("contradiction", "contradiction", ["Quincy: The kettle has never once been broken.", "Rue: Handy.", "Quincy: I replaced the kettle after it cracked in half."]),
    ("contradiction", "contradiction", ["Pim: My stall opens at six every day.", "Ode: Early.", "Pim: On Mondays the stall stays closed till noon."]),
    ("legitimate_update", "update", ["Tal: The quiz starts at seven.", "Una: Fine.", "Tal: The host asked for eight instead, so the quiz is at eight."]),
    ("legitimate_update", "update", ["Vic: Our table is number nine.", "Wyn: Good.", "Vic: They moved us, so we are at table twelve now."]),
    ("wrong_speaker_trap", "consistent", ["Xan: I will drive on Sunday.", "Yoli: Okay.", "Yoli: I will take the train on Sunday."]),
    ("hypothetical_or_joking", "consistent", ["Zed: The picnic is on the ninth.", "Abe: Great.", "Zed: If it snowed we would eat indoors, ha."]),
    ("insufficient_evidence", "insufficient", ["Bix: We booked the boat.", "Cy: Lovely."]),
    ("insufficient_evidence", "insufficient", ["Dov: A parcel arrived for us.", "Eli: Good."]),
    ("recall", None, ["Fen: The key is under the mat.", "Gus: Thanks."]),
]


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=root, check=True, capture_output=True)


def _stub_llm(label: str = "consistent"):
    class Llm:
        model = "stub-llm-for-tests"

        def complete(self, messages, *, schema=None, max_tokens=512, temperature=0.0, timeout_s=10.0):
            return SimpleNamespace(parsed=schema(label=label, confidence=0.7))

    return Llm()


def _scorer(pairs):
    return [[0.9, 0.05, 0.05] if "never" in b else [0.1, 0.8, 0.1] for _, b in pairs]


class Repo:
    def __init__(self, root: Path, base: Path) -> None:
        self.root, self.base = root, base

    def run(self, **kw):
        args = dict(root=self.root, raw_root=FIXTURE_RAW, base_model=str(self.base), device="cpu",
                    grid=((1e-3, 2), (3e-3, 2)), make_zeroshot=lambda: ZeroShotChecker(scorer=_scorer),
                    make_prompted=lambda: PromptedChecker(_stub_llm()))
        args.update(kw)
        return e4.run_e4(**args)


def _dev_scenario(i: int, cat: str, label: str | None, lines: list[str]) -> Scenario:
    s = ts._scenario("x", i, cat, lines, [], label=label)  # type: ignore[arg-type]
    cid = f"dev-fx-{i}"
    turns = [t.model_copy(update={"conversation_id": cid, "id": f"{cid}:t{t.seq}"}) for t in s.conversation.turns]
    conv = s.conversation.model_copy(update={"id": cid, "split": "dev", "turns": turns})
    return s.model_copy(update={"id": cid, "conversation": conv, "request": s.request.model_copy(update={"conversation_id": cid})})


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Repo:
    from contextlens.eval import series as ser

    monkeypatch.setattr(ser, "RESAMPLES", 100)
    root = tmp_path / "repo"
    d = root / "datasets/scenarios"
    d.mkdir(parents=True)
    fixed = [_dev_scenario(i, *case) for i, case in enumerate(DEV_CASES)]
    (d / "dev.jsonl").write_text("".join(s.model_dump_json() + "\n" for s in fixed), encoding="utf-8")
    write_scenarios_manifest(root)
    (root / "datasets/scenarios_train").mkdir(parents=True)
    shutil.copy(ROOT / "datasets/scenarios_train/train.jsonl", root / "datasets/scenarios_train/train.jsonl")
    (root / ".gitignore").write_text("datasets/raw/\nmodels/\nmlruns/\n", encoding="utf-8")
    e4.prepare_decode_dev(FIXTURE_RAW, root)
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    base, tok = _tiny(tmp_path)
    base.save_pretrained(tmp_path / "base")
    tok.save_pretrained(tmp_path / "base")
    return Repo(root, tmp_path / "base")


def _records(root: Path) -> dict[str, dict]:
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / "results/runs").glob("E4-*.json"))]
    return {r["scenario_source"]: r for r in recs}


def test_the_run_writes_valid_records_for_both_sources(repo: Repo) -> None:
    result = repo.run()
    assert result["complete"] and result["winner"] in {"lr0.001_ep2", "lr0.003_ep2"}
    assert check("E4", root=repo.root, min_n=1, split="dev") == []
    recs = _records(repo.root)
    assert set(recs) == {"scenarios", "decode"}
    for r in recs.values():
        assert r["split"] == "dev" and sorted(r["predictions"]) == sorted(e4.SYSTEMS)
        training = [a for a in r["attempts"] if a["kind"] == "training"]
        assert [a["setting"] for a in training] == ["lr0.001_ep2", "lr0.003_ep2"] and all(a["ok"] for a in training)
        assert {a["setting"] for a in r["attempts"] if a["kind"] == "system"} == {"tfidf", "zeroshot", "prompted"}
        assert r["config"]["winner_rule"] == e4.WINNER_RULE and "own writing style" in r["config"]["style_note"]
        assert {c["b"] for c in r["comparisons"]} <= {"tfidf", "zeroshot", "prompted"}
        assert r["config"]["train_set_review_note"] == e4.TRAIN_SET_REVIEW_NOTE
    sc =recs["scenarios"]
    assert {"macro_f1", "false_alert_rate", "label_accuracy", "ece"} <= set(sc["metrics"]["trained"])
    assert sc["n"] == len(DEV_CASES) and sc["config"]["scored_cases"] == len(DEV_CASES) - 1
    assert (repo.root / "results/error-analysis-e4.md").is_file()
    assert any((repo.root / "models/e4").glob("*/label_map.json"))


def test_decode_rows_hold_ids_and_labels_only_and_check_run_enforces_it(repo: Repo) -> None:
    repo.run()
    rec = _records(repo.root)["decode"]
    for rel in rec["predictions"].values():
        for line in (repo.root / rel).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            assert row["card"] is None and row["raw_output"] is None and row["label"] in ("contradiction", "consistent")
    tracked = subprocess.run(["git", "ls-files"], cwd=repo.root, capture_output=True, text=True).stdout
    assert "scenarios/decode" not in tracked and (repo.root / "datasets/raw/scenarios/decode/dev.jsonl").is_file()
    path = repo.root / rec["predictions"]["trained"]
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["card"] = {"mode": "clarify", "status": "abstained", "findings": []}
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert any("ids and metrics only" in p for p in check("E4", root=repo.root, split="dev"))


def test_a_tampered_ece_is_rejected_and_an_honest_one_is_not(repo: Repo) -> None:
    repo.run()
    run_json = next(p for p in (repo.root / "results/runs").glob("E4-*.json")
                    if json.loads(p.read_text(encoding="utf-8"))["scenario_source"] == "scenarios")
    rec = json.loads(run_json.read_text(encoding="utf-8"))
    assert rec["metrics"]["trained"]["ece"] is not None
    rec["metrics"]["trained"]["ece"] += 0.05
    run_json.write_text(json.dumps(rec), encoding="utf-8")
    assert any("ece does not recompute" in p for p in check("E4", root=repo.root, split="dev"))


def test_a_search_run_that_fails_is_listed_and_the_search_goes_on(repo: Repo, monkeypatch: pytest.MonkeyPatch) -> None:
    real = ct.fit

    def flaky(*a, **kw):
        if kw["lr"] == 1e-3:
            raise RuntimeError("CUDA out of memory")
        return real(*a, **kw)

    monkeypatch.setattr(ct, "fit", flaky)
    result = repo.run()
    assert result["winner"] == "lr0.003_ep2" and result["complete"]
    attempts = {a["setting"]: a for a in _records(repo.root)["scenarios"]["attempts"] if a["kind"] == "training"}
    assert not attempts["lr0.001_ep2"]["ok"] and "out of memory" in attempts["lr0.001_ep2"]["error"]
    assert check("E4", root=repo.root, min_n=1, split="dev") == []


def test_when_every_run_fails_nothing_is_written(repo: Repo, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*a, **kw):
        raise RuntimeError("no GPU memory")

    monkeypatch.setattr(ct, "fit", broken)
    with pytest.raises(RuntimeError, match="no training run finished"):
        repo.run()
    assert not (repo.root / "results").exists()


def test_a_baseline_that_fails_is_listed_and_left_out(repo: Repo) -> None:
    class Down:
        def complete(self, *a, **kw):
            raise ConnectionError("ollama is not running")

    result = repo.run(make_prompted=lambda: PromptedChecker(Down()))
    assert not result["complete"] and "prompted" in result["failed"]
    recs = _records(repo.root)
    assert "prompted" not in recs["scenarios"]["predictions"]
    failed = [a for a in recs["decode"]["attempts"] if a["kind"] == "system" and not a["ok"]]
    assert [a["setting"] for a in failed] == ["prompted"] and "ollama is not running" in failed[0]["error"]
    assert check("E4", root=repo.root, min_n=1, split="dev") == []


def test_the_winner_rule_is_fixed_by_macro_f1_then_fewer_epochs_then_lower_lr() -> None:
    def a(lr, ep, f1, ok=True):
        return {"setting": f"{lr}-{ep}", "lr": lr, "epochs": ep, "ok": ok, "dev_macro_f1": f1}

    assert e4.pick_winner([a(1e-5, 2, 0.5), a(2e-5, 3, 0.6), a(3e-5, 2, 0.4)])["setting"] == "2e-05-3"
    assert e4.pick_winner([a(2e-5, 3, 0.6), a(2e-5, 2, 0.6), a(1e-5, 3, 0.6)])["setting"] == "2e-05-2"
    assert e4.pick_winner([a(1e-5, 3, 0.6), a(2e-5, 3, 0.6)])["setting"] == "1e-05-3"
    assert e4.pick_winner([a(1e-5, 2, None, ok=False)]) is None
    assert len(e4.GRID) <= 6


def test_dev_scenarios_cannot_reach_training(repo: Repo) -> None:
    mine = ts.load_train_scenarios(repo.root / "datasets/scenarios_train")
    with pytest.raises(RuntimeError, match="share text with dev"):
        e4.training_data(FIXTURE_RAW, repo.root, mine[:3])
    dev = load_scenarios("dev", repo.root / "datasets/scenarios")
    data = e4.training_data(FIXTURE_RAW, repo.root, dev)
    assert data.counts["decode_train_pairs"] == 4 and {e.label for e in data.examples} >= {"update", "insufficient"}
    assert not {e.group for e in data.examples} & {s.id for s in dev}


def test_the_runner_never_reads_the_test_split() -> None:
    src = Path(e4.__file__).read_text(encoding="utf-8")
    assert '"test"' not in src and "'test'" not in src


def test_error_analysis_lists_fifty_by_category_and_never_shows_decode_text() -> None:
    def sc(i: int, cat: str, text: str) -> Scenario:
        s = _sc(f"{cat}-{i}", "contradiction", cat)
        turns = [t.model_copy(update={"text": text}) for t in s.conversation.turns]
        return s.model_copy(update={"conversation": s.conversation.model_copy(update={"turns": turns})})

    mine = [sc(i, c, "invented words here") for c in ("update_case", "trap_case") for i in range(40)]
    theirs = [sc(i, "decode", "SECRET DECODE SENTENCE") for i in range(40)]

    def wrong(ss):
        return [Prediction(scenario_id=s.id, label="consistent", probability=0.6) for s in ss]

    md = e4.error_analysis({"scenarios": mine, "decode": theirs}, {"scenarios": wrong(mine), "decode": wrong(theirs)})
    assert md.count("\n### ") == 50 and "SECRET DECODE SENTENCE" not in md and "invented words here" in md
    counts = {c: md.count(f"### {c}-") for c in ("update_case", "trap_case", "decode")}
    assert counts == {"update_case": 16, "trap_case": 17, "decode": 17}  # alphabetical, so decode and trap_case get the extra one
    assert "- decode: 40" in md and "- trap_case: 40" in md


# ---- workers, timing and memory, and out-of-memory retries ----


def test_fit_gives_the_same_result_with_loader_workers(tmp_path: Path) -> None:
    def run(workers: int):
        model, tok = _tiny(tmp_path)
        return ct.fit(model, tok, EXAMPLES * 2, EXAMPLES, lr=5e-3, epochs=2, batch_size=4, patience=2,
                      num_workers=workers)

    plain, workers = run(0), run(2)
    assert [h["train_loss"] for h in plain.history] == [h["train_loss"] for h in workers.history]
    assert plain.best_macro_f1 == workers.best_macro_f1


def test_fit_reports_seconds_per_epoch_and_no_memory_figure_without_a_gpu(tmp_path: Path) -> None:
    model, tok = _tiny(tmp_path)
    logged: list[dict] = []
    res = ct.fit(model, tok, EXAMPLES, EXAMPLES, lr=1e-3, epochs=2, batch_size=4, patience=2,
                 log=lambda m, step: logged.append(m))
    assert res.seconds_per_epoch is not None and res.seconds_per_epoch > 0
    assert res.peak_gpu_memory_mb is None  # not a false 0
    assert all("epoch_seconds" in m for m in logged) and not any("peak_gpu_memory_mb" in m for m in logged)


def test_the_first_finished_search_run_prints_its_speed_and_memory(repo: Repo) -> None:
    said: list[str] = []
    repo.run(say=said.append)
    first = [m for m in said if m.startswith("FIRST SEARCH RUN DONE")]
    assert len(first) == 1 and "s/epoch" in first[0] and "not measured: no GPU" in first[0] and "lr0.001_ep2" in first[0]
    assert said.index(first[0]) < said.index("training lr0.003_ep2")
    attempts = [a for a in _records(repo.root)["scenarios"]["attempts"] if a["kind"] == "training"]
    assert all(a["seconds_per_epoch"] > 0 and a["peak_gpu_memory_mb"] is None and a["peak_gpu_memory_note"] for a in attempts)
    assert all(a["batch_size"] == e4.BATCH_SIZE and not a["lowered_batch_size"] for a in attempts)


def test_out_of_memory_retries_at_half_the_batch_size_and_flags_it(repo: Repo, monkeypatch: pytest.MonkeyPatch) -> None:
    real = ct.fit
    sizes: list[int] = []

    def tight(*a, **kw):
        sizes.append(kw["batch_size"])
        if kw["lr"] == 1e-3 and kw["batch_size"] > 2:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return real(*a, **kw)

    monkeypatch.setattr(ct, "fit", tight)
    result = repo.run()
    assert sizes == [8, 4, 2, 8] and result["complete"]
    for rec in _records(repo.root).values():
        att = {a["setting"]: a for a in rec["attempts"] if a["kind"] == "training"}
        low, normal = att["lr0.001_ep2"], att["lr0.003_ep2"]
        assert low["ok"] and low["error"] is None and low["batch_size"] == 2 and low["lowered_batch_size"]
        assert [r["batch_size"] for r in low["oom_retries"]] == [8, 4]
        assert normal["batch_size"] == 8 and not normal["lowered_batch_size"] and normal["oom_retries"] == []
        assert rec["config"]["settings_at_lower_batch_size"] == ["lr0.001_ep2"] and "batch size" in rec["config"]["batch_size_note"]
    assert check("E4", root=repo.root, min_n=1, split="dev") == []


def test_out_of_memory_below_the_floor_is_a_failed_setting(repo: Repo, monkeypatch: pytest.MonkeyPatch) -> None:
    real = ct.fit
    sizes: list[int] = []

    def never_fits(*a, **kw):
        if kw["lr"] == 1e-3:
            sizes.append(kw["batch_size"])
            raise RuntimeError("CUDA out of memory")
        return real(*a, **kw)

    monkeypatch.setattr(ct, "fit", never_fits)
    repo.run()
    att = {a["setting"]: a for a in _records(repo.root)["scenarios"]["attempts"] if a["kind"] == "training"}
    assert sizes == [8, 4, 2] and not att["lr0.001_ep2"]["ok"] and "floor" in att["lr0.001_ep2"]["error"]
    assert att["lr0.001_ep2"]["batch_size"] == 2


def test_another_error_is_not_retried(repo: Repo, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def broken(*a, **kw):
        calls.append(kw["batch_size"])
        raise ValueError("bad data")

    monkeypatch.setattr(ct, "fit", broken)
    with pytest.raises(RuntimeError, match="no training run finished"):
        repo.run()
    assert calls == [8, 8]


# ---- the NaN loss found by the first real run: fp16 weights, a non-finite loss that was recorded as ok ----


def test_pretrained_weights_load_as_float32_even_from_a_float16_checkpoint(tmp_path: Path) -> None:
    # The real DeBERTa checkpoint is stored in fp16 and transformers 5 loads the stored dtype; AdamW's eps (1e-8)
    # underflows to 0 in fp16, so the first optimizer step turned the weights into NaN.
    model, tok = _tiny(tmp_path)
    model.half().save_pretrained(tmp_path / "half")
    tok.save_pretrained(tmp_path / "half")
    loaded, _ = cm.load_pretrained(str(tmp_path / "half"))
    assert {p.dtype for p in loaded.parameters()} == {torch.float32}


def _nan_loss(monkeypatch: pytest.MonkeyPatch, only_lr: float | None = None) -> None:
    real = ct.batch_loss
    real_fit = ct.fit
    state: dict[str, float | None] = {"lr": None}

    def fit(*a, **kw):
        state["lr"] = kw["lr"]
        return real_fit(*a, **kw)

    def loss(model, batch, weights):
        out = real(model, batch, weights)
        return out * float("nan") if only_lr is None or state["lr"] == only_lr else out

    monkeypatch.setattr(ct, "fit", fit)
    monkeypatch.setattr(ct, "batch_loss", loss)


def test_a_non_finite_loss_stops_training_at_once_and_leaves_the_weights_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlens.train_guard import NonFiniteLoss

    model, tok = _tiny(tmp_path)
    before = [p.clone() for p in model.parameters()]
    _nan_loss(monkeypatch)
    with pytest.raises(NonFiniteLoss, match="non-finite"):
        ct.fit(model, tok, EXAMPLES, EXAMPLES, lr=1e-3, epochs=2, batch_size=4)
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_the_guard_helper_names_the_value_and_where() -> None:
    from contextlens.train_guard import NonFiniteLoss, non_finite_epochs, require_finite

    assert require_finite(0.5, "x") == 0.5
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(NonFiniteLoss, match="step 3"):
            require_finite(bad, "step 3")
    assert non_finite_epochs([{"epoch": 1, "train_loss": 0.4}, {"epoch": 2, "train_loss": float("nan")},
                              {"epoch": 3, "train_loss": None}, {"epoch": 4}]) == [2, 3, 4]


def test_a_setting_with_a_nan_loss_is_listed_as_failed_never_ok(repo: Repo, monkeypatch: pytest.MonkeyPatch) -> None:
    _nan_loss(monkeypatch, only_lr=1e-3)
    result = repo.run()
    assert result["winner"] == "lr0.003_ep2"
    for rec in _records(repo.root).values():
        att = {a["setting"]: a for a in rec["attempts"] if a["kind"] == "training"}
        assert att["lr0.001_ep2"]["ok"] is False and "non-finite" in att["lr0.001_ep2"]["error"]
        assert att["lr0.001_ep2"]["history"] == [] and att["lr0.003_ep2"]["ok"]
    assert check("E4", root=repo.root, min_n=1, split="dev") == []


def test_when_every_setting_gives_a_nan_loss_nothing_is_written(repo: Repo, monkeypatch: pytest.MonkeyPatch) -> None:
    _nan_loss(monkeypatch)
    with pytest.raises(RuntimeError, match="no training run finished.*non-finite"):
        repo.run()
    assert not (repo.root / "results").exists()


def test_a_rerun_supersedes_the_first_records_with_a_reason_and_says_what_changed(repo: Repo) -> None:
    from contextlens.eval.check_run import superseded_by

    repo.run()
    olds = sorted((repo.root / "results/runs").glob("E4-*.json"))
    old = {json.loads(p.read_text(encoding="utf-8"))["run_id"] for p in olds}
    _git(repo.root, "add", "-A")
    _git(repo.root, "commit", "-q", "-m", "first run")  # the first records are committed before a rerun, as in practice
    time.sleep(0.01)
    entries = [{"run_id": i, "reason": e4.RERUN_REASON} for i in sorted(old)]
    repo.run(supersedes=entries)
    new = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((repo.root / "results/runs").glob("E4-*.json"))
           if p not in olds]
    assert len(new) == 2 and all(len(r["supersedes"]) == 1 and r["supersedes"][0] in entries for r in new)
    assert {r["supersedes"][0]["run_id"] for r in new} == old
    assert all(p.is_file() for p in olds)  # the first records stay on disk
    assert set(superseded_by("E4", repo.root)) == old
    assert all("float32" in r["config"]["rerun_note"] for r in new)
    assert check("E4", root=repo.root, min_n=1, split="dev") == []


def test_the_smoke_run_checks_a_finite_falling_loss_and_writes_nothing(repo: Repo) -> None:
    out = e4.smoke(root=repo.root, raw_root=FIXTURE_RAW, base_model=str(repo.base), device="cpu", lr=3e-3, steps=24)
    assert len(out["losses"]) == 24 and all(x == x for x in out["losses"]) and out["last"] < out["first"]
    assert out["peak_gpu_memory_mb"] is None  # no GPU here: not measured, not zero
    assert not (repo.root / "results").exists() and not (repo.root / "models").exists()


def test_the_smoke_run_fails_on_a_loss_that_does_not_fall(repo: Repo) -> None:
    with pytest.raises(RuntimeError, match="did not fall"):
        e4.smoke(root=repo.root, raw_root=FIXTURE_RAW, base_model=str(repo.base), device="cpu", lr=0.0, steps=8)


def test_the_smoke_run_stops_on_a_nan_loss(repo: Repo, monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlens.train_guard import NonFiniteLoss

    _nan_loss(monkeypatch)
    with pytest.raises(NonFiniteLoss):
        e4.smoke(root=repo.root, raw_root=FIXTURE_RAW, base_model=str(repo.base), device="cpu", lr=1e-3, steps=4)


@pytest.mark.real
def test_real_deberta_trains_finitely_on_cpu_at_the_real_settings() -> None:
    """Slow (loads DeBERTa-v3-base, a few CPU steps), so it runs only when asked:
    `pytest -m real tests/test_e4.py -k deberta`. It is the check that would have caught the first real run's NaN
    without a GPU."""
    model, tok = cm.load_pretrained(cm.BASE_MODEL)
    assert {p.dtype for p in model.parameters()} == {torch.float32}
    for p in model.get_input_embeddings().parameters():
        p.requires_grad = False
    labels = ["consistent"] * 4129 + ["contradiction"] * 4079 + ["insufficient"] * 80 + ["update"] * 96
    weights = torch.tensor(ct.class_weights(labels))
    batch = ct.collate(tok, [_ex(lab, "we moved the deadline to wednesday") for lab in cm.LABELS] * 2)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5, weight_decay=0.01)
    model.train()
    for _ in range(3):
        loss = ct.train_step(model, dict(batch), opt, weights)
        assert loss == loss and abs(loss) != float("inf")
        assert all(torch.isfinite(p).all() for p in model.parameters())


# ---- per-class F1, the training balance and the DECODE caveat ----


def test_per_class_f1_is_saved_by_source_recomputes_and_a_tampered_one_fails(repo: Repo) -> None:
    repo.run()
    recs = _records(repo.root)
    for name in e4.SYSTEMS:
        sc, dc = recs["scenarios"]["metrics"][name], recs["decode"]["metrics"][name]
        assert {f"f1[label={x}]" for x in M.LABELS} <= set(sc)
        assert {k for k in dc if k.startswith("f1[")} == {"f1[label=contradiction]", "f1[label=consistent]"}
    # the fixture has no gold case for neither label of a missing kind: every label here has gold, so all are numbers
    assert all(isinstance(recs["scenarios"]["metrics"]["trained"][f"f1[label={x}]"], float) for x in M.LABELS)
    assert check("E4", root=repo.root, min_n=1, split="dev") == []
    run_json = next(p for p in (repo.root / "results/runs").glob("E4-*.json")
                    if json.loads(p.read_text(encoding="utf-8"))["scenario_source"] == "scenarios")
    rec = json.loads(run_json.read_text(encoding="utf-8"))
    rec["metrics"]["trained"]["f1[label=update]"] = 0.123
    run_json.write_text(json.dumps(rec), encoding="utf-8")
    assert any("f1[label=update] does not recompute" in p for p in check("E4", root=repo.root, split="dev"))


def test_the_config_states_the_train_only_counts_weights_ratio_and_the_few_examples_caveat(repo: Repo) -> None:
    repo.run()
    mine = ts.build_train_scenarios()
    distinct = {x: sum(s.gold_label == x for s in mine) for x in M.LABELS}
    for r in _records(repo.root).values():
        tb = r["config"]["train_only_balance"]
        assert tb["distinct_cases_per_label"] == distinct and tb["repeats"] == e4.SCENARIO_REPEAT
        counts = r["config"]["training"]
        weights = tb["class_weights"]
        assert list(weights.values()) == pytest.approx(ct.class_weights(
            [x for x in M.LABELS for _ in range(counts[f"label_{x}"])]))
        assert tb["weight_ratio"] == pytest.approx(max(weights.values()) / min(w for w in weights.values() if w > 0))
        assert "no new examples" in tb["note"] and "very few distinct" in tb["note"]


def test_the_decode_record_says_its_score_is_binary_and_the_error_analysis_has_no_decode_accuracy(repo: Repo) -> None:
    repo.run()
    note = _records(repo.root)["decode"]["config"]["decode_note"]
    assert "contradiction vs not only" in note and "not four-label performance" in note
    md = (repo.root / "results/error-analysis-e4.md").read_text(encoding="utf-8")
    assert "contradiction vs not only" in md and "accuracy" not in md.lower()


# ---- phase 5 review fixes: portable checkpoint pointer, Ollama unload, shared leakage guard, caveats ----

import threading  # noqa: E402
from http.server import BaseHTTPRequestHandler, HTTPServer  # noqa: E402

from contextlens.consistency.data import assert_trainable, build_pairs  # noqa: E402
from contextlens.eval.checkpoints import chosen_checkpoint  # noqa: E402
from contextlens.llm import client as llm_client  # noqa: E402


def test_new_records_store_the_checkpoint_relative_and_the_helper_verifies_its_hash(repo: Repo) -> None:
    repo.run()
    rec = _records(repo.root)["scenarios"]
    training = [a for a in rec["attempts"] if a["kind"] == "training"]
    assert all(a["checkpoint"] == f"models/e4/{a['setting']}" for a in training)
    path = chosen_checkpoint(rec, repo.root)
    assert path == repo.root / "models/e4" / rec["config"]["winner"]
    (path / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="sha256"):
        chosen_checkpoint(rec, repo.root)


def test_the_helper_finds_a_checkpoint_from_an_old_absolute_path_and_fails_loudly_when_missing(tmp_path: Path) -> None:
    ckpt = tmp_path / "models/e4/lr1e-05_ep2"
    ckpt.mkdir(parents=True)
    (ckpt / "model.safetensors").write_bytes(b"weights")
    import hashlib
    rec = {"config": {"winner": "lr1e-05_ep2", "checkpoint_sha256": hashlib.sha256(b"weights").hexdigest()},
           "attempts": [{"kind": "training", "setting": "lr1e-05_ep2",
                         "checkpoint": "C:/Users/someone/Work/_worktrees/Gone/T5.2/models/e4/lr1e-05_ep2"}]}
    assert chosen_checkpoint(rec, tmp_path) == ckpt
    with pytest.raises(FileNotFoundError):
        chosen_checkpoint(rec, tmp_path / "elsewhere")


class _OllamaStub(BaseHTTPRequestHandler):
    seen: list[tuple[str, dict]] = []

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _OllamaStub.seen.append((self.path, body))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *a) -> None:
        pass


def test_unload_ollama_posts_keep_alive_zero() -> None:
    _OllamaStub.seen = []
    server = HTTPServer(("127.0.0.1", 0), _OllamaStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert llm_client.unload_ollama("m:1", f"http://127.0.0.1:{server.server_port}")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert _OllamaStub.seen == [("/api/generate", {"model": "m:1", "keep_alive": 0})]
    assert llm_client.unload_ollama("m:1", "http://127.0.0.1:9") is False  # nothing listening: reported, not raised


def test_the_runner_unloads_the_ollama_model_even_when_the_run_fails(repo: Repo, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(e4, "unload_ollama", lambda model, *a, **k: calls.append(model))
    with pytest.raises(RuntimeError, match="no training run finished"):
        e4.run_and_unload(root=repo.root, raw_root=FIXTURE_RAW, base_model=str(repo.base / "missing"), device="cpu",
                          grid=((1e-3, 2),), make_zeroshot=lambda: ZeroShotChecker(scorer=_scorer),
                          make_prompted=lambda: PromptedChecker(_stub_llm()))
    assert calls == [e4.PROMPTED_MODEL]


def test_a_dev_or_test_conversation_cannot_reach_training() -> None:
    mine = ROOT / "datasets/scenarios_train"
    for split in ("dev", "test"):
        held = build_pairs(split, FIXTURE_RAW, mine)
        assert held
        with pytest.raises(RuntimeError, match=f"{split}"):
            assert_trainable(held[:1] + build_pairs("train", FIXTURE_RAW, mine), FIXTURE_RAW, mine)
    assert_trainable(build_pairs("train", FIXTURE_RAW, mine), FIXTURE_RAW, mine)


def test_training_stops_when_a_train_scenario_repeats_a_test_scenario(repo: Repo) -> None:
    mine = ts.load_train_scenarios(repo.root / "datasets/scenarios_train")
    (repo.root / "datasets/scenarios/test.jsonl").write_text(mine[0].model_dump_json() + "\n", encoding="utf-8")
    dev = load_scenarios("dev", repo.root / "datasets/scenarios")
    with pytest.raises(RuntimeError, match="held-out"):
        e4.training_data(FIXTURE_RAW, repo.root, dev)


def test_the_error_analysis_states_the_style_and_selection_caveats() -> None:
    text = e4.error_analysis({"scenarios": []}, {"scenarios": []})
    assert e4.STYLE_NOTE in text and e4.SELECTION_NOTE in text
