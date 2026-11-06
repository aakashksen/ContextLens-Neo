"""E5: fine-tuned against prompted. Fakes and invented text only; no GPU, no network, no model loaded, no torch."""
import json
import re
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


def test_max_gold_is_12_and_a_card_cites_every_gold_turn_or_the_example_is_dropped() -> None:
    assert ds.MAX_GOLD == 12

    def wide(sid: str, n_gold: int) -> Scenario:
        turns = _turns(sid, [f"Remark number {i} about the {sid} plan." for i in range(n_gold + 4)])
        return Scenario(id=sid, conversation=Conversation(id=sid, turns=turns, source="synthetic", split="train"),
                        request=ButtonRequest(conversation_id=sid, as_of_seq=len(turns) - 1), category="c",
                        gold_evidence_turn_ids=[t.id for t in turns[:n_gold]], gold_label="contradiction", gold_answer=None,
                        origin="synthetic")

    stats: Counter = Counter()
    ex = ds._example(wide("ok", 12), "train", "scenarios_train", stats)
    assert ex is not None
    cited = [e.turn_id for e in Card.model_validate_json(ex.completion).findings[0].evidence]
    assert cited == [f"ok:t{i}" for i in range(12)]  # every gold turn, none left out
    assert ds._example(wide("too-wide", 13), "train", "scenarios_train", stats) is None  # dropped, not cut to 12
    assert stats["dropped_answer_not_citable"] == 1


def test_the_abstain_cap_is_applied_after_the_length_filter() -> None:
    def ex(i: str, status: str) -> ds.Example:
        card = (Card(mode="recall", status="abstained", findings=[], missing=["q"]) if status == "abstained" else
                Card(mode="recall", status="answered", findings=[Finding(
                    agent="a", kind="recall", statement="s", confidence=1.0,
                    evidence=[EvidenceRef(turn_id="c:t0", quote="q", speaker="A", start_ms=0)])]))
        return ds.Example(id=i, source="qmsum", split="train", conversation_id=i, messages=[], completion=card.model_dump_json())

    answered = [ex(f"a{i}", "answered") for i in range(8)] + [ex(f"long{i}", "answered") for i in range(4)]
    abstained = [ex(f"x{i}", "abstained") for i in range(4)]
    by_file = {"qmsum": {"train": [*answered, *abstained], "val": []}}
    stats: Counter = Counter()
    ds._cap_train(by_file, 0.25, 13, stats, fits=lambda e: not e.id.startswith("long"))
    kept = by_file["qmsum"]["train"]
    assert not any(e.id.startswith("long") for e in kept) and stats["train_dropped_too_long"] == 4
    n_abs = sum(e.id.startswith("x") for e in kept)
    assert len(kept) == 8 + n_abs and n_abs / len(kept) <= 0.25 and n_abs == 2  # 8 answered fit, so 2 of 10 may abstain
    assert stats["train_final_size"] == len(kept)


def test_training_summary_reports_mix_and_flags_a_small_set() -> None:
    st = Counter({"train_final_size": 120, "train_class_answered": 90, "train_class_abstained": 30, "train_source_qmsum": 100,
                  "train_source_scenarios_train": 20, "train_synthetic_share": 20 / 120, "converted_consistent": 5})
    s = e5.training_summary(st)
    assert s["by_source"] == {"qmsum": 100, "scenarios_train": 20} and s["under_minimum"] is True
    assert s["abstained_share"] == pytest.approx(0.25) and s["max_gold"] == 12 and s["abstain_cap"] == 0.25
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
    (tmp_path / ".gitignore").write_text("results/partial/\n", encoding="utf-8")
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
            "train_meta": {"base_model": e5.BASE_MODEL, "fallback_used": False, "fallback_reason": None,
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


# ---- write, then judge, end to end with fakes ----

class FakeHF(e5.RecordingHF):
    """No model: replies with a one-finding card quoting the first turn id found in the prompt."""

    log: list[tuple[str, str | None]] = []
    turns: dict[str, Turn] = {}
    generated = 0

    def load(self) -> None:
        FakeHF.log.append(("load", self.adapter_path))

    def unload(self) -> None:
        FakeHF.log.append(("unload", self.adapter_path))

    def _generate(self, messages, max_tokens, temperature, timeout_s):
        FakeHF.generated += 1
        t = FakeHF.turns[re.search(r"[\w-]+:t\d+", messages[-1]["content"]).group(0)]
        f = Finding(agent="a", kind="recall", statement=f"{t.speaker} said it.", confidence=1.0,
                    evidence=[EvidenceRef(turn_id=t.id, quote=t.text, speaker=t.speaker, start_ms=t.start_ms)])
        return Card(mode="recall", status="answered", findings=[f]).model_dump_json(), 10, 5


class FakeHaiku(FakeLLM):
    def __init__(self, n: int) -> None:
        super().__init__([{"supported": True}] * n)
        self.model, self.calls_made = "claude-haiku-5-5", 0


def test_write_then_judge_runs_offline_and_judges_qmsum_too(repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import contextlens.experiments.judge_check as jc
    import contextlens.llm.client as client_mod
    from contextlens.data.scenarios import load_scenarios

    shared = tmp_path / "shared"
    adapter = _adapter(shared / "models/e5-adapter")
    (adapter / "e5_training.json").write_text(json.dumps(_training(adapter)), encoding="utf-8")
    own = load_scenarios("dev", repo / "datasets/scenarios")
    qm = [_sc("q1", None, question="When?", gold=[1]), _sc("q2", "contradiction", gold=[0, 2])]
    FakeHF.log, FakeHF.generated = [], 0
    FakeHF.turns = {t.id: t for s in [*own, *qm] for t in s.conversation.turns}
    unloaded: list[str] = []
    monkeypatch.setattr(e5, "data_root", lambda: shared)
    monkeypatch.setattr(e5, "eval_scenarios", lambda root: {"scenarios": own, "qmsum": qm})
    monkeypatch.setattr(e5, "RecordingHF", FakeHF)
    monkeypatch.setattr(e5, "select_evidence",
                        lambda scs, emb, cfg: {s.id: [(s.conversation.turns[1].id, 0.5), (s.conversation.turns[2].id, 0.4)] for s in scs})
    monkeypatch.setattr(client_mod, "unload_ollama", lambda model, *a, **k: unloaded.append(model))
    monkeypatch.setattr(jc, "preflight", lambda *a, **k: 0)

    folder = e5.run_write(repo, "v1", say=lambda *_: None)
    assert unloaded == [e5.EMBED_MODEL]  # the embedder is released before any model loads
    assert FakeHF.log == [("load", None), ("unload", None), ("load", str(adapter)), ("unload", str(adapter))]
    assert FakeHF.generated == 2 * (len(own) + len(qm))
    for side in ("prompted", "adapter"):
        for source in ("scenarios", "qmsum"):
            assert (folder / f"{side}.{source}.rows.jsonl").is_file()
    e5.run_write(repo, "v1", say=lambda *_: None)
    assert FakeHF.generated == 2 * (len(own) + len(qm))  # a rerun finds every card saved and writes none

    haiku = FakeHaiku(200)
    series.RESAMPLES = 50
    try:
        path = e5.run_judge(repo, "v1", say=lambda *_: None, llm=haiku, guard=SimpleNamespace(start=lambda: 0.0, check=lambda: None),
                            concurrency=1)
        assert check("E5", root=repo) == []
    finally:
        series.RESAMPLES = 10_000
    rec = json.loads(path.read_text(encoding="utf-8"))
    q = rec["config"]["per_source"]["qmsum"]
    assert q["judged"] is True and q["adapter"]["grounded_rate"] == 1.0 and "unsupported_claims" in q["prompted"]
    assert rec["models"]["judge"] == "claude-haiku-5-5" and rec["config"]["adapter_hash"] == e5.adapter_hash(adapter)
    saved = (folder / "judge.rows.jsonl").read_text(encoding="utf-8")
    assert "shipment" not in saved and "claim_id" in saved  # ids and verdicts only
    assert len(haiku.calls) == 2 * (len(own) + len(qm))  # one call per claim, one claim per card here


def test_judge_refuses_when_cards_are_missing(repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlens.data.scenarios import load_scenarios

    shared = tmp_path / "shared"
    adapter = _adapter(shared / "models/e5-adapter")
    (adapter / "e5_training.json").write_text(json.dumps(_training(adapter)), encoding="utf-8")
    monkeypatch.setattr(e5, "data_root", lambda: shared)
    monkeypatch.setattr(e5, "eval_scenarios", lambda root: {"scenarios": load_scenarios("dev", repo / "datasets/scenarios"), "qmsum": []})
    with pytest.raises(RuntimeError, match="run --write first"):
        e5.run_judge(repo, "v1", say=lambda *_: None, llm=FakeHaiku(1), guard=SimpleNamespace(start=lambda: 0.0, check=lambda: None))


# ---- the fallback subset, the speed check, and the generation settings ----

def _many(prefix: str, n: int) -> list[Scenario]:
    return [_sc(f"{prefix}{i}", ["contradiction", "consistent", None][i % 3], question="q?" if i % 3 == 2 else None,
                gold=[0, 1] if i % 3 != 2 else None) for i in range(n)]


def test_fallback_manifest_is_seeded_ids_only_and_inside_the_dev_sets(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    own, q100 = _many("d", 148), _many("q", 100)
    monkeypatch.setattr(e5, "eval_scenarios", lambda root: {"scenarios": own, "qmsum": q100})
    path = e5.make_fallback_subset(repo)
    m = json.loads(path.read_text(encoding="utf-8"))
    assert len(m["scenarios"]) == 60 and len(m["qmsum"]) == 50
    assert set(m["scenarios"]) <= {s.id for s in own} and set(m["qmsum"]) <= {s.id for s in q100}
    assert m["seed"] and "70 s per card" in m["reason"] and "9.7 h" in m["reason"] and "coordinator" in m["reason"]
    assert "text" not in json.dumps(m).replace("context", "")  # ids and the reason only
    first = path.read_text(encoding="utf-8")
    assert e5.make_fallback_subset(repo).read_text(encoding="utf-8") == first  # same seed, same ids


def test_fallback_write_covers_only_the_manifest_and_the_record_says_why(repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import contextlens.experiments.judge_check as jc
    import contextlens.llm.client as client_mod
    from contextlens.data.scenarios import load_scenarios

    shared = tmp_path / "shared"
    adapter = _adapter(shared / "models/e5-adapter")
    (adapter / "e5_training.json").write_text(json.dumps(_training(adapter)), encoding="utf-8")
    own = load_scenarios("dev", repo / "datasets/scenarios")
    qm = [_sc("q1", None, question="When?", gold=[1]), _sc("q2", "contradiction", gold=[0, 2])]
    man = repo / "datasets/splits/subsets" / "e5-fallback.json"
    man.parent.mkdir(parents=True, exist_ok=True)
    keep_own = [s.id for s in own][:3]
    man.write_text(json.dumps({"name": "e5-fallback", "seed": "x", "scenarios": keep_own, "qmsum": ["q1"],
                               "reason": "70 s per card"}), encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "subset")
    FakeHF.log, FakeHF.generated = [], 0
    FakeHF.turns = {t.id: t for s in [*own, *qm] for t in s.conversation.turns}
    monkeypatch.setattr(e5, "data_root", lambda: shared)
    monkeypatch.setattr(e5, "eval_scenarios", lambda root: {"scenarios": own, "qmsum": qm})
    monkeypatch.setattr(e5, "RecordingHF", FakeHF)
    monkeypatch.setattr(e5, "select_evidence",
                        lambda scs, emb, cfg: {s.id: [(s.conversation.turns[1].id, 0.5), (s.conversation.turns[2].id, 0.4)] for s in scs})
    monkeypatch.setattr(client_mod, "unload_ollama", lambda *a, **k: None)
    monkeypatch.setattr(jc, "preflight", lambda *a, **k: 0)
    folder = e5.run_write(repo, "v1", say=lambda *_: None, fallback=True)
    assert folder.name.endswith("-fallback") and FakeHF.generated == 2 * (3 + 1)
    series.RESAMPLES = 50
    try:
        path = e5.run_judge(repo, "v1", say=lambda *_: None, llm=FakeHaiku(50), fallback=True, concurrency=1,
                            guard=SimpleNamespace(start=lambda: 0.0, check=lambda: None))
    finally:
        series.RESAMPLES = 10_000
    cfg = json.loads(path.read_text(encoding="utf-8"))["config"]
    assert cfg["subset"]["name"] == "e5-fallback" and cfg["subset"]["n"] == {"scenarios": 3, "qmsum": 1}
    assert "70 s per card" in cfg["subset"]["reason"] and cfg["per_source"]["scenarios"]["n"] == 3


def test_speed_check_writes_three_base_cards_and_reports_rates(repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlens.data.scenarios import load_scenarios

    own = load_scenarios("dev", repo / "datasets/scenarios")
    FakeHF.log, FakeHF.generated = [], 0
    FakeHF.turns = {t.id: t for s in own for t in s.conversation.turns}
    monkeypatch.setattr(e5, "data_root", lambda: tmp_path / "shared")
    monkeypatch.setattr(e5, "eval_scenarios", lambda root: {"scenarios": own, "qmsum": []})
    monkeypatch.setattr(e5, "RecordingHF", FakeHF)
    monkeypatch.setattr(e5, "select_evidence",
                        lambda scs, emb, cfg: {s.id: [(s.conversation.turns[1].id, 0.5)] for s in scs})
    import contextlens.llm.client as client_mod
    monkeypatch.setattr(client_mod, "unload_ollama", lambda *a, **k: None)
    lines: list[str] = []
    out = e5.run_speed_check(repo, "v1", say=lines.append)
    assert FakeHF.generated == 3 and FakeHF.log == [("load", None), ("unload", None)]  # base side only
    assert out["cards"] == 3 and out["tokens"] == 15 and out["s_per_card"] >= 0 and out["tokens_per_s"] > 0
    assert any("s per card" in x and "tokens per second" in x for x in lines)


def test_generation_uses_the_cache_in_eval_mode_without_gradients() -> None:
    torch = pytest.importorskip("torch")
    seen = {}

    class Tok:
        pad_token_id = 0
        def apply_chat_template(self, m, **kw): return "p"
        def __call__(self, text, **kw):
            return SimpleNamespace(to=lambda dev: {"input_ids": torch.tensor([[1, 2]])})
        def decode(self, ids, **kw): return "x"

    class Model:
        device = "cpu"
        training = True
        is_gradient_checkpointing = True
        def eval(self): self.training = False
        def gradient_checkpointing_disable(self): self.is_gradient_checkpointing = False
        def generate(self, **kw):
            seen.update(kw, training=self.training, grad=torch.is_grad_enabled(), gc=self.is_gradient_checkpointing)
            return torch.tensor([[1, 2, 3]])

    client = HFClient("m")
    client._model, client._tok = Model(), Tok()
    client._generate([{"role": "user", "content": "x"}], 123, 0.0, 5.0)
    assert seen["use_cache"] is True and seen["training"] is False and seen["grad"] is False and seen["gc"] is False
    assert seen["max_new_tokens"] == 123 and seen["do_sample"] is False  # decoding settings are the caller's


def test_card_reply_limit_fits_the_longest_training_card() -> None:
    assert e5.REPLY_TOKENS == 800  # the longest of the 212 training cards is 788 tokens; a cut card is invalid and costs a retry


def test_speed_check_ignores_cards_saved_by_an_earlier_run(repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlens.data.scenarios import load_scenarios
    import contextlens.llm.client as client_mod

    own = load_scenarios("dev", repo / "datasets/scenarios")
    FakeHF.log, FakeHF.generated = [], 0
    FakeHF.turns = {t.id: t for s in own for t in s.conversation.turns}
    for name, val in (("data_root", lambda: tmp_path / "shared"), ("eval_scenarios", lambda root: {"scenarios": own, "qmsum": []}),
                      ("RecordingHF", FakeHF), ("select_evidence", lambda scs, emb, cfg: {s.id: [(s.conversation.turns[1].id, 0.5)] for s in scs})):
        monkeypatch.setattr(e5, name, val)
    monkeypatch.setattr(client_mod, "unload_ollama", lambda *a, **k: None)
    first = e5.run_speed_check(repo, "v1", say=lambda *_: None)
    second = e5.run_speed_check(repo, "v1", say=lambda *_: None)  # the saved rows from the first run must not be reused
    assert FakeHF.generated == 6 and second["tokens"] == 15 == first["tokens"]


def test_speed_check_fails_when_no_token_was_counted(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import contextlens.llm.client as client_mod
    from contextlens.data.scenarios import load_scenarios

    own = load_scenarios("dev", repo / "datasets/scenarios")
    monkeypatch.setattr(e5, "eval_scenarios", lambda root: {"scenarios": own, "qmsum": []})
    monkeypatch.setattr(e5, "RecordingHF", FakeHF)
    monkeypatch.setattr(e5, "select_evidence", lambda scs, emb, cfg: {})
    monkeypatch.setattr(e5, "write_side", lambda *a, **k: [])
    monkeypatch.setattr(client_mod, "unload_ollama", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="no generated tokens"):
        e5.run_speed_check(repo, "v1", say=lambda *_: None)


def test_record_states_the_base_model_change() -> None:
    c = e5.BASE_MODEL_CHANGE
    assert (c["from"], c["to"]) == ("Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-2B") == ("Qwen/Qwen3.5-4B", e5.BASE_MODEL)
    assert "5.5 GB" in c["why"] and "82 examples" in c["why"] and "2026-11-05" in c["why"]
    assert e5.training_summary(Counter({"train_final_size": 212}))["base_model_change"] == c


def test_a_different_loaded_model_stops_write_and_judge(repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shared = tmp_path / "shared"
    adapter = _adapter(shared / "models/e5-adapter")
    info = _training(adapter)
    info["train_meta"].update(base_model="Qwen/Qwen3-4B", fallback_used=True)
    (adapter / "e5_training.json").write_text(json.dumps(info), encoding="utf-8")
    monkeypatch.setattr(e5, "data_root", lambda: shared)
    for call in (lambda: e5.run_write(repo, "v1", say=lambda *_: None), lambda: e5.run_judge(repo, "v1", say=lambda *_: None)):
        with pytest.raises(RuntimeError, match="Qwen/Qwen3.5-2B.*trained on Qwen/Qwen3-4B"):
            call()


def test_e5_training_does_not_fall_back_to_another_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlens.cards import train_lora as tl

    loaded = []
    def boom(model_id):
        loaded.append(model_id)
        raise ValueError("unsupported architecture")
    monkeypatch.setattr(tl, "_load_4bit", boom)
    with pytest.raises(ValueError, match="unsupported"):
        tl.train(tmp_path / "o", steps=1, base_model=e5.BASE_MODEL, allow_fallback=False)
    assert loaded == [e5.BASE_MODEL]  # no second load of another model


# ---- embedding the cases before any GPU model loads ----

class LimitedEmbedder:
    """Like Ollama's: a request of more than 100 texts is refused (the real server refused 256 with a 400)."""

    model = "fake-embed"

    def __init__(self, fail_on: str | None = None) -> None:
        self.requests: list[list[str]] = []
        self.fail_on = fail_on

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.requests.append(list(texts))
        if len(texts) > 100 or (self.fail_on and self.fail_on in " ".join(texts)):
            raise RuntimeError("request failed: HTTP Error 400: Bad Request")
        return [[float(len(t) % 7), 1.0, float(len(t) % 3)] for t in texts]


def _long_sc(sid: str, n: int) -> Scenario:
    sc = _sc(sid, None, question="When?", gold=[1])
    turns = _turns(sid, [f"turn number {i} about topic {i % 5}" for i in range(n)])
    return sc.model_copy(update={"conversation": sc.conversation.model_copy(update={"turns": turns}),
                                 "request": sc.request.model_copy(update={"as_of_seq": n - 1})})


def test_a_long_conversation_is_embedded_in_batches_the_server_accepts() -> None:
    emb = LimitedEmbedder()
    out = e5.select_evidence([_long_sc("big", 300)], emb, {"k": 3, "mode": "hybrid"})
    assert len(out["big"]) == 3 and max(len(r) for r in emb.requests) <= 32  # the E1 batch size


def test_preflight_names_the_case_that_cannot_be_embedded_and_goes_biggest_first() -> None:
    emb = LimitedEmbedder(fail_on="turn number 7 ")
    with pytest.raises(RuntimeError, match="case big-2: embedding failed"):
        e5.select_evidence([_long_sc("small", 5), _long_sc("big-2", 40), _long_sc("big-1", 30)], emb, {"k": 3, "mode": "hybrid"})
    assert emb.requests[0][0].startswith("turn number 0")  # the biggest conversation went first
    assert len(emb.requests) <= 2  # it stopped at the first failure


def test_selection_reuses_what_the_preflight_embedded() -> None:
    emb = LimitedEmbedder()
    e5.select_evidence([_long_sc("a", 20)], emb, {"k": 3, "mode": "hybrid"})
    embedded = [t for r in emb.requests for t in r]
    assert len(embedded) == len(set(embedded))  # no text was sent twice
