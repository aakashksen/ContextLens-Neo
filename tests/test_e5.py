"""E5: fine-tuned against prompted. Fakes and invented text only; no GPU, no network, no model loaded, no torch."""
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from contextlens.cards import dataset as ds
from contextlens.eval import series
from contextlens.eval.check_run import check
from contextlens.eval.records import write_scenarios_manifest
from contextlens.experiments import e5_cards as e5
from contextlens.llm.client import FakeLLM
from contextlens.llm.hf import HFClient
from contextlens.replay.runner import Prediction, replay
from contextlens.schemas import ButtonRequest, Card, Conversation, EvidenceRef, Finding, Scenario, Turn
from contextlens.train_guard import NonFiniteLoss


def _turns(sid: str, texts: list[str]) -> list[Turn]:
    return [Turn(id=f"{sid}:t{i}", conversation_id=sid, seq=i, speaker="Ana" if i % 2 else "Ben", text=t,
                 start_ms=i * 1000, end_ms=i * 1000 + 900, source="synthetic") for i, t in enumerate(texts)]


def _sc(sid: str, label: str | None, question: str | None = None, gold: list[int] | None = None,
        answer: str | None = None) -> Scenario:
    turns = _turns(sid, [f"The {sid} shipment leaves on day {i}." for i in range(5)])
    return Scenario(id=sid, conversation=Conversation(id=sid, turns=turns, source="synthetic", split="dev"),
                    request=ButtonRequest(conversation_id=sid, as_of_seq=4, question=question), category="c",
                    gold_evidence_turn_ids=[turns[i].id for i in (gold or [])], gold_label=label,  # type: ignore[arg-type]
                    gold_answer=answer, origin="synthetic")


# ---- HFClient ----

def test_hfclient_loads_nothing_until_used_and_returns_an_llm_result(monkeypatch: pytest.MonkeyPatch) -> None:
    client = HFClient("some/base", adapter_path="models/x")
    assert client._model is None and client.model == "some/base+models/x"
    monkeypatch.setattr(HFClient, "_generate", lambda self, m, mt, t, to: ('{"mode":"recall","status":"abstained","findings":[]}', 12, 7))
    res = client.complete([{"role": "user", "content": "hi"}], schema=Card)
    assert isinstance(res.parsed, Card) and (res.prompt_tokens, res.completion_tokens) == (12, 7) and res.latency_ms >= 0


@pytest.mark.parametrize("module", ["contextlens.llm.hf", "contextlens.experiments.e5_cards"])
def test_importing_e5_modules_does_not_load_torch(module: str) -> None:
    code = f"import sys, {module}\nprint(','.join(m for m in ('torch','transformers','peft') if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-400:]
    assert out.stdout.strip() == ""


# ---- training data and the NaN guard ----

def test_consistent_scenario_becomes_a_clarify_card_citing_both_turns() -> None:
    sc = _sc("cons", "consistent", gold=[0, 3])
    stats: Counter = Counter()
    assert ds._example(sc, "train", "scenarios_train", stats) is None and stats["dropped_no_card_for_label"] == 1
    ex = ds._example(sc, "train", "scenarios_train", stats, convert_consistent=True)
    assert ex is not None and stats["converted_consistent"] == 1
    card = Card.model_validate_json(ex.completion)
    assert (card.mode, card.status) == ("clarify", "answered")
    (finding,) = card.findings
    assert finding.label == "consistent" and [e.turn_id for e in finding.evidence] == ["cons:t0", "cons:t3"]
    assert "agree" in finding.statement
    assert ds._example(_sc("one", "consistent", gold=[0]), "train", "scenarios_train", Counter(), True) is None


def test_training_summary_reports_mix_and_flags_a_small_set() -> None:
    st = Counter({"train_final_size": 120, "train_class_answered": 90, "train_class_abstained": 30, "train_source_qmsum": 100,
                  "train_source_scenarios_train": 20, "train_synthetic_share": 20 / 120, "converted_consistent": 5})
    s = e5.training_summary(st)
    assert s["by_source"] == {"qmsum": 100, "scenarios_train": 20} and s["under_minimum"] is True
    assert s["abstained_share"] == pytest.approx(0.25) and s["max_gold"] == 4 and s["abstain_cap"] == 0.25
    assert e5.training_summary(Counter({"train_final_size": 300}))["under_minimum"] is False


def test_non_finite_loss_stops_training_before_the_update(monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlens.cards import train_lora as tl

    calls = {"backward": 0, "step": 0}

    class Tensor:
        def to(self, _device):
            return self

    class Loss:
        def __init__(self, v): self.v = v
        def item(self): return self.v
        def backward(self): calls["backward"] += 1

    seen = iter([0.5, float("nan")])

    class Model:
        device = "cpu"

        def train(self): pass
        def parameters(self): return [SimpleNamespace(requires_grad=True)]
        def __call__(self, **kw): return SimpleNamespace(loss=Loss(next(seen)))

    model = Model()

    class Opt:
        def __init__(self, *a, **k): pass
        def step(self): calls["step"] += 1
        def zero_grad(self): pass

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(optim=SimpleNamespace(AdamW=Opt)))
    monkeypatch.setattr(tl, "tokenize_example", lambda tok, ex, n: {"input_ids": [1], "labels": [1]})
    monkeypatch.setattr(tl, "collate", lambda rows, pad: {"input_ids": Tensor()})
    tok = SimpleNamespace(pad_token_id=0)
    with pytest.raises(NonFiniteLoss):
        tl.run_steps(model, tok, [SimpleNamespace()], 10, steps=3)
    assert calls == {"backward": 1, "step": 1}  # the first step ran; the NaN step never reached backward


# ---- the two writers ----

def test_adapter_side_is_shown_the_training_format_and_prompted_side_the_prompt() -> None:
    sc = _sc("w", None, question="When does it leave?", gold=[1])
    card = Card(mode="recall", status="abstained", findings=[], missing=["x"])
    sel = {sc.id: [(sc.conversation.turns[1].id, 0.9)]}
    seen = {}
    for name, cls in (("adapter", e5.AdapterAssistant), ("prompted", e5.PromptedAssistant)):
        llm = FakeLLM([card])
        replay([sc], lambda llm=llm, cls=cls: cls(llm, e5.replay_retriever(sc, sel), "v1", deadline_s=30.0))
        seen[name] = llm.calls[0]
    assert seen["adapter"][0]["content"] == ds.SYSTEM
    assert seen["adapter"][1]["content"].startswith("Recent exchange:") and "Question: When does it leave?" in seen["adapter"][1]["content"]
    assert seen["prompted"][0]["content"] != ds.SYSTEM and "Retrieved evidence" not in seen["prompted"][1]["content"]


# ---- measures ----

def test_gold_mode_and_false_alarm_rate() -> None:
    assert e5.gold_mode(_sc("a", "contradiction", gold=[0, 1])) == "clarify"
    assert e5.gold_mode(_sc("b", None, question="q?")) == "recall"
    assert e5.gold_mode(_sc("c", None)) == "respond"
    ev = [EvidenceRef(turn_id="x:t0", quote="q", speaker="Ben", start_ms=0)]

    def row(label: str) -> Prediction:
        f = Finding(agent="a", kind="consistency", statement="s", label=label, confidence=1.0, evidence=ev)  # type: ignore[arg-type]
        return Prediction(scenario_id="x", card=Card(mode="clarify", status="answered", findings=[f]))

    rows = [(_sc("a", "consistent", gold=[0, 1]), row("consistent")), (_sc("b", "consistent", gold=[0, 1]), row("update")),
            (_sc("c", "contradiction", gold=[0, 1]), row("update"))]
    assert e5.false_alarm_rate(rows) == {"n": 2, "false_alarms": 1, "rate": 0.5}


def test_winner_rule_needs_a_clear_grounded_gain_and_nothing_worse() -> None:
    good = [{"metric": "grounded_rate", "ci": [0.02, 0.2]}]
    m = {"adapter": {"unsupported_claim_rate": 0.1, "schema_valid_rate": 1.0, "abstain_when_insufficient": 0.8,
                     "answer_when_answerable": 0.9},
         "prompted": {"unsupported_claim_rate": 0.2, "schema_valid_rate": 0.9, "abstain_when_insufficient": 0.8,
                      "answer_when_answerable": 0.9}}
    assert e5.choose_winner(good, m)[0] == "adapter"
    assert e5.choose_winner([{"metric": "grounded_rate", "ci": [-0.01, 0.2]}], m)[0] == "prompted"
    worse = {**m, "adapter": {**m["adapter"], "unsupported_claim_rate": 0.3}}
    assert e5.choose_winner(good, worse)[0] == "prompted"


# ---- the record ----

def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    tmp_path = tmp_path / "repo"
    d = tmp_path / "datasets/scenarios"
    d.mkdir(parents=True)
    scs = [_sc("s1", "contradiction", gold=[0, 1]), _sc("s2", "consistent", gold=[0, 1]), _sc("s3", "insufficient", question="q?"),
           _sc("s4", None, question="When?", gold=[2], answer="day 2"), _sc("s5", "update", gold=[0, 3]), _sc("s6", None)]
    (d / "dev.jsonl").write_text("".join(s.model_dump_json() + "\n" for s in scs), encoding="utf-8")
    (d / "test.jsonl").write_text("", encoding="utf-8")
    write_scenarios_manifest(tmp_path)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def _card_for(sc: Scenario, grounded_quote: str) -> Prediction:
    t = sc.conversation.turns[0]
    f = Finding(agent="a", kind="consistency", statement="Earlier it left.", label="consistent", confidence=1.0,
                evidence=[EvidenceRef(turn_id=t.id, quote=grounded_quote, speaker=t.speaker, start_ms=t.start_ms)])
    card = Card(mode="clarify", status="answered", findings=[f])
    return Prediction(scenario_id=sc.id, card=card, claim_supported=[True], raw_output=card.model_dump_json(),
                      stage_latency_ms={"total": 1000})


def _training(adapter: Path, loss: float = 0.5) -> dict:
    return {"training_set": {"size": 10}, "adapter_hash": e5.adapter_hash(adapter), "epochs": 2, "batch_size": 2,
            "train_meta": {"base_model": "Qwen/Qwen3.5-4B", "fallback_used": False, "fallback_reason": None,
                           "peak_vram_gb": 5.0, "losses": [1.0, loss]}}


def _adapter(path: Path, weights: bytes = b"weights") -> Path:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(weights)
    return path


def _write(repo: Path, adapter: Path, loss: float = 0.5) -> Path:
    from contextlens.data.scenarios import load_scenarios
    from contextlens.eval.records import split_hash

    scs = load_scenarios("dev", repo / "datasets/scenarios")
    preds = {"prompted": [_card_for(s, "x") for s in scs], "adapter": [_card_for(s, "y") for s in scs]}
    series.RESAMPLES = 50
    try:
        return e5.build_record(root=repo, scenarios=scs, dataset_hash=split_hash("dev", repo), preds=preds, extra_config={},
                               models={"writer": "Qwen/Qwen3.5-4B", "judge": "claude-haiku-5-5"}, prompt_version="v3",
                               training=_training(adapter, loss), qmsum={"n": 0, "judged": False})
    finally:
        series.RESAMPLES = 10_000


def test_tiny_fixture_writes_a_valid_record(repo: Path, tmp_path: Path) -> None:
    path = _write(repo, _adapter(tmp_path / "ad"))
    rec = json.loads(path.read_text(encoding="utf-8"))
    cfg = rec["config"]
    assert set(rec["predictions"]) == {"prompted", "adapter"} and rec["prompt_versions"]["prompted"] == "v3"
    for key in ("synthetic_statement", "overlap_guard_statement", "independent_read_statement", "per_mode", "per_source",
                "false_alarm_on_consistent", "unsupported_claims", "winner_rule"):
        assert key in cfg
    assert "paraphrase" in cfg["overlap_guard_statement"] and "0.13 to 0.18" in cfg["independent_read_statement"]
    assert "Haiku failed" in cfg["judge"]["statement"] and "reading and not proof" in cfg["judge"]["statement"]
    assert cfg["false_alarm_on_consistent"]["adapter"]["n"] == 1
    series.RESAMPLES = 50
    try:
        assert check("E5", root=repo) == []
    finally:
        series.RESAMPLES = 10_000


def test_adapter_hash_recorded(repo: Path, tmp_path: Path) -> None:
    adapter = _adapter(tmp_path / "ad")
    rec = json.loads(_write(repo, adapter).read_text(encoding="utf-8"))
    assert rec["config"]["adapter_hash"] == e5.adapter_hash(adapter) and len(rec["config"]["adapter_hash"]) == 64
    other = _adapter(tmp_path / "ad2", b"other weights")
    assert e5.adapter_hash(other) != e5.adapter_hash(adapter)
    with pytest.raises(FileNotFoundError):
        e5.adapter_hash(tmp_path)


def test_a_non_finite_training_loss_in_the_record_is_rejected(repo: Path, tmp_path: Path) -> None:
    path = _write(repo, _adapter(tmp_path / "ad"))
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec["config"]["winner"] = "adapter"
    rec["attempts"][0]["history"][1]["train_loss"] = float("nan")
    path.write_text(json.dumps(rec), encoding="utf-8")
    series.RESAMPLES = 50
    try:
        assert any("non-finite training loss" in p for p in check("E5", root=repo))
    finally:
        series.RESAMPLES = 10_000
