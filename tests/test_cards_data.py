import json
import re
from collections import Counter
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from contextlens.cards.dataset import MAX_GOLD, Example, build_examples, load_examples
from contextlens.cards.train_lora import attach_lora, check_vram, is_architecture_error, run_steps, tokenize_example
from contextlens.llm.untrusted import unescape
from contextlens.data.scenarios import load_scenarios
from contextlens.schemas import ButtonRequest, Card, Conversation, EvidenceRef, Finding, Scenario, Turn

N_TRAIN, N_DEV, N_TEST = 12, 3, 3


def _meeting(k: int) -> dict:
    turns = [{"speaker": f"S{i % 3}", "content": f"meeting {k} turn {i} <b> & word{k}x{i} about topic{k} plan {i}"}
             for i in range(10)]
    queries = [
        {"query": f"What was said about topic{k}?", "answer": f"They discussed topic{k}.",
         "relevant_text_span": [[str(4 + k % 3), str(5 + k % 3)]]},
        {"query": f"Second question {k}?", "answer": f"Answer {k}.", "relevant_text_span": [["7", "7"]]},
    ]
    return {"meeting_transcripts": turns, "general_query_list": [{"query": "Summarize."}],
            "specific_query_list": queries}


@pytest.fixture
def data(tmp_path: Path) -> tuple[Path, Path]:
    """Invented meetings: the first 12 are our train, then 3 dev, then 3 test."""
    raw, splits = tmp_path / "raw", tmp_path / "splits"
    (raw / "qmsum").mkdir(parents=True)
    splits.mkdir()
    ids: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    lines = []
    for k in range(N_TRAIN + N_DEV + N_TEST):
        name = "train" if k < N_TRAIN else "dev" if k < N_TRAIN + N_DEV else "test"
        lines.append(json.dumps(_meeting(k)))
        ids[name].append(f"qmsum-all-{k}")
    (raw / "qmsum" / "all.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (splits / "qmsum.json").write_text(json.dumps({"splits": ids}), encoding="utf-8")
    return raw, splits


def _build(data, tmp_path: Path) -> tuple[dict[str, list[Example]], dict[str, list[str]]]:
    raw, splits = data
    out = tmp_path / "cards"
    counts = build_examples(raw, splits, out)
    exs = {s: [Example.model_validate_json(line)
               for line in (out / f"qmsum-{s}.jsonl").read_text(encoding="utf-8").splitlines()] for s in counts}
    return exs, json.loads((splits / "qmsum.json").read_text(encoding="utf-8"))["splits"]


def test_every_example_parses_as_card(data, tmp_path):
    exs, _ = _build(data, tmp_path)
    all_ex = [e for v in exs.values() for e in v]
    assert len(all_ex) > 10
    cards = [Card.model_validate_json(e.completion) for e in all_ex]
    assert {c.status for c in cards} == {"answered", "abstained"}
    assert all(e.messages[-1]["role"] == "user" for e in all_ex)


def test_every_citation_exists_in_its_input(data, tmp_path):
    exs, _ = _build(data, tmp_path)
    for e in (e for v in exs.values() for e in v):
        user = e.messages[-1]["content"]
        shown = dict(re.findall(r"^\[([^\]]+)\] (.*)$", user.split("Retrieved evidence:")[1], re.M))
        for f in Card.model_validate_json(e.completion).findings:
            assert f.evidence
            for ref in f.evidence:
                speaker, text = unescape(shown[ref.turn_id]).split(": ", 1)
                assert speaker == ref.speaker
                assert ref.quote and ref.quote in text
    assert any("&amp;" in e.messages[-1]["content"] for v in exs.values() for e in v)  # escaping is in effect


def test_no_conversation_in_two_splits(data, tmp_path):
    exs, manifest = _build(data, tmp_path)
    seen = {s: {e.conversation_id for e in v} for s, v in exs.items()}
    assert seen["train"] and seen["val"] and not seen["train"] & seen["val"]
    used = seen["train"] | seen["val"]
    assert used <= set(manifest["train"])
    assert not used & (set(manifest["dev"]) | set(manifest["test"]))


def test_manifest_overlap_is_refused(data):
    raw, splits = data
    m = json.loads((splits / "qmsum.json").read_text(encoding="utf-8"))
    m["splits"]["test"].append("qmsum-all-0")
    (splits / "qmsum.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ValueError):
        build_examples(raw, splits)


def _tiny_tokenizer(texts: list[str]) -> PreTrainedTokenizerFast:
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=400, special_tokens=["<pad>", "<eos>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>")
    tok.chat_template = ("{% for m in messages %}<{{ m.role }}>{{ m.content }}\n{% endfor %}"
                         "{% if add_generation_prompt %}<assistant>{% endif %}")
    return tok


def _small(data, tmp_path):
    exs, _ = _build(data, tmp_path)
    ex = exs["train"][:4]
    return ex, _tiny_tokenizer([m["content"] for e in ex for m in e.messages] + [e.completion for e in ex])


def test_tokenized_labels_mask_the_prompt(data, tmp_path):
    ex, tok = _small(data, tmp_path)
    row = tokenize_example(tok, ex[0], 100_000)
    assert len(row["input_ids"]) == len(row["labels"])
    n_prompt = sum(1 for x in row["labels"] if x == -100)
    assert 0 < n_prompt < len(row["labels"]) and set(row["labels"][:n_prompt]) == {-100}
    assert row["labels"][n_prompt:] == row["input_ids"][n_prompt:]
    assert tok.decode(row["labels"][n_prompt:]) == ex[0].completion + "<eos>"
    assert tokenize_example(tok, ex[0], 10) is None


def test_dry_run_one_step_on_tiny_model(data, tmp_path):
    ex, tok = _small(data, tmp_path)
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)
    model = attach_lora(Qwen2ForCausalLM(cfg))
    base_before = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
    lora_b = {n: p.clone() for n, p in model.named_parameters() if "lora_B" in n}
    assert lora_b and all(not t.any() for t in lora_b.values())
    (loss,) = run_steps(model, tok, ex, max_len=4096, steps=1)
    assert torch.isfinite(torch.tensor(loss))
    assert any(p.any() for n, p in model.named_parameters() if "lora_B" in n)  # LoRA moved
    assert all(torch.equal(p, base_before[n]) for n, p in model.named_parameters() if n in base_before)  # base did not


def _long_meeting(k: int, span: tuple[int, int]) -> dict:
    turns = [{"speaker": f"S{i % 3}",
              "content": (f"GOLD{k}x{i} decision text" if span[0] <= i <= span[1] else f"filler{k}x{i} chatter")}
             for i in range(60)]
    q = {"query": f"What was decided {k}?", "answer": f"A long answer {k}.",
         "relevant_text_span": [[str(span[0]), str(span[1])]]}
    return {"meeting_transcripts": turns, "general_query_list": [], "specific_query_list": [q]}


@pytest.fixture
def long_data(tmp_path: Path) -> tuple[Path, Path]:
    """Forty train meetings: each has one question whose gold span is 20 turns (more than MAX_GOLD),
    plus a few with a short span that can be fully cited."""
    raw, splits = tmp_path / "lraw", tmp_path / "lsplits"
    (raw / "qmsum").mkdir(parents=True)
    splits.mkdir()
    metas = [_long_meeting(k, (10, 29)) for k in range(40)] + [_long_meeting(40 + k, (10, 11)) for k in range(10)]
    (raw / "qmsum" / "all.jsonl").write_text(
        "\n".join(json.dumps(m) for m in metas) + "\n", encoding="utf-8")
    ids = [f"qmsum-all-{k}" for k in range(len(metas))]
    (splits / "qmsum.json").write_text(json.dumps({"splits": {"train": ids, "dev": [], "test": []}}), encoding="utf-8")
    return raw, splits


def test_abstained_input_holds_no_gold_turn_from_a_long_span(long_data, tmp_path):
    stats: Counter = Counter()
    build_examples(*long_data, tmp_path / "cards", stats=stats)
    exs = [e for s in ("train", "val") for e in _load(tmp_path / "cards", s)]
    abstained = [e for e in exs if Card.model_validate_json(e.completion).status == "abstained"]
    assert len(abstained) >= 3
    for e in abstained:
        assert "GOLD" not in e.messages[-1]["content"]  # not in the recent exchange, the evidence, or the question


def test_answered_cites_the_whole_span_or_is_dropped(long_data, tmp_path):
    stats: Counter = Counter()
    build_examples(*long_data, tmp_path / "cards", stats=stats)
    exs = [e for s in ("train", "val") for e in _load(tmp_path / "cards", s)]
    answered = [e for e in exs if Card.model_validate_json(e.completion).status == "answered"]
    assert answered
    for e in answered:
        (f,) = Card.model_validate_json(e.completion).findings
        assert len(f.evidence) <= MAX_GOLD
        assert e.messages[-1]["content"].count("GOLD") == len(f.evidence)  # every gold turn shown is cited
    assert stats["dropped_answer_not_citable"] > 0


def _load(out: Path, split: str) -> list[Example]:
    return [Example.model_validate_json(line) for line in (out / f"qmsum-{split}.jsonl").read_text(encoding="utf-8").splitlines()]


def test_only_architecture_errors_trigger_the_fallback():
    import errno

    assert all(is_architecture_error(e) for e in (ValueError("x"), KeyError("qwen3_5"), TypeError("x"),
                                                  AttributeError("x"), OSError("no config.json found")))
    assert not any(is_architecture_error(e) for e in (
        RuntimeError("CUDA out of memory"), ImportError("bitsandbytes"), ConnectionError("down"),
        TimeoutError(), PermissionError("denied"), OSError(errno.ENOSPC, "No space left on device")))


def test_vram_limit_is_checked_during_training(data, tmp_path, monkeypatch):
    ex, tok = _small(data, tmp_path)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)
    model = attach_lora(Qwen2ForCausalLM(cfg))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 9 * 1024**3)
    with pytest.raises(RuntimeError, match="blocked"):
        run_steps(model, tok, ex, max_len=4096, steps=5, vram_limit_gb=7.0)
    assert check_vram(10.0) == 9.0


@pytest.mark.real
def test_real_model_loads_under_peft_and_vram_limit():
    from contextlens.cards.train_lora import train

    meta = train(Path("models/cards-check"), steps=1)
    assert meta["peak_vram_gb"] <= 7.0


def _scn(cid: str, lines: list[str], *, split="train", origin="synthetic", label=None, question=None, answer=None,
         gold=(0,)) -> Scenario:
    turns = [Turn(id=f"{cid}:t{i}", conversation_id=cid, seq=i, speaker=ln.split(": ")[0], text=ln.split(": ", 1)[1],
                  start_ms=i * 1000, end_ms=i * 1000 + 900, source="synthetic") for i, ln in enumerate(lines)]
    return Scenario(id=cid, conversation=Conversation(id=cid, turns=turns, source="synthetic", split=split),
                    request=ButtonRequest(conversation_id=cid, as_of_seq=len(turns) - 1, question=question),
                    category="x", gold_evidence_turn_ids=[turns[g].id for g in gold], gold_label=label,
                    gold_answer=answer, origin=origin)  # type: ignore[arg-type]


def _lines(tag: str, n: int = 8) -> list[str]:
    return [f"P{i % 2}: {tag} remark number {i} regarding the zebra harvest schedule" for i in range(n)]


def _write(root: Path, split: str, scs: list[Scenario]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{split}.jsonl").write_text("".join(s.model_dump_json() + "\n" for s in scs), encoding="utf-8")


@pytest.fixture
def scen(tmp_path: Path) -> tuple[Path, Path]:
    """An invented train-only set covering every mode, and invented dev and test files to check it against."""
    train = [
        _scn(f"syn-rec-{k}", _lines(f"rec{k}"), question=f"What about rec{k}?", answer=f"Answer rec{k}.", gold=(3, 4))
        for k in range(10)
    ] + [
        _scn("syn-con-0", ["A: The launch is on Monday", "B: ok noted", "B: The launch is on Friday"],
             label="contradiction", gold=(0, 2)),
        _scn("syn-upd-0", ["A: Budget is ten units", "B: fine", "A: Budget is now twelve units"],
             label="update", gold=(0, 2)),
        _scn("syn-rsp-0", ["A: We chose the blue vendor for cost", "B: sure", "A: hmm", "B: What do you think?"],
             answer="You preferred the blue vendor for cost.", gold=(0,)),
        _scn("syn-ins-0", _lines("ins"), label="insufficient", question="Who owns the mango contract?", gold=()),
        _scn("syn-cons-0", _lines("cons"), label="consistent", gold=(0, 1)),
    ]
    held = tmp_path / "heldout"
    _write(held, "dev", [_scn("syn-dev-0", _lines("devonly"), split="dev")])
    _write(held, "test", [_scn("syn-test-0", _lines("testonly"), split="test")])
    _write(tmp_path / "scen", "train", train)
    return tmp_path / "scen", held


def _scenario_examples(data, scen, tmp_path) -> tuple[list[Example], Counter]:
    st: Counter = Counter()
    build_examples(*data, tmp_path / "cards", train_scenarios=scen[0], held_out_dir=scen[1], stats=st)
    return [e for s in ("train", "val") for e in load_examples(s, tmp_path / "cards", include_scenarios=True)
            if e.source == "scenarios_train"], st


def test_scenario_set_examples_cover_every_mode_and_mark_source(data, scen, tmp_path):
    exs, st = _scenario_examples(data, scen, tmp_path)
    cards = [Card.model_validate_json(e.completion) for e in exs]
    modes = {(c.mode, c.status) for c in cards}
    assert {("recall", "answered"), ("clarify", "answered"), ("respond", "answered"), ("recall", "abstained")} <= modes
    assert all(e.source == "scenarios_train" for e in exs)
    assert st["dropped_no_card_for_label"] == 1
    respond = next(c for c in cards if c.mode == "respond")
    assert respond.suggestion and respond.findings[0].evidence
    clarify = next(c for c in cards if c.mode == "clarify")
    assert clarify.findings[0].label in ("contradiction", "update") and len(clarify.findings[0].evidence) == 2
    ins = next(e for e in exs if e.id == "syn-ins-0")
    assert Card.model_validate_json(ins.completion).status == "abstained"
    qm = [e for s in ("train", "val") for e in load_examples(s, tmp_path / "cards")]
    assert qm and {e.source for e in qm} == {"qmsum"}  # the default load stays QMSum-only


def test_default_build_without_the_scenario_set_says_so(data, tmp_path):
    st: Counter = Counter()
    build_examples(*data, tmp_path / "cards", stats=st)
    assert st["scenario_set_absent"] == 1
    assert not (tmp_path / "cards" / "scenarios-train.jsonl").exists()


@pytest.mark.parametrize("bad", ["split", "origin", "id", "text", "overlap"])
def test_scenario_set_refuses_anything_that_could_leak(data, scen, tmp_path, bad):
    root, held = scen
    base = _scn("syn-bad-0", _lines("bad"), question="Q?", answer="A.", gold=(1,))
    if bad == "split":
        base = _scn("syn-bad-0", _lines("bad"), split="dev", question="Q?", answer="A.", gold=(1,))
    elif bad == "origin":
        base = _scn("syn-bad-0", _lines("bad"), origin="dataset", question="Q?", answer="A.", gold=(1,))
    elif bad == "id":
        base = _scn("syn-test-0", _lines("bad"), question="Q?", answer="A.", gold=(1,))
    elif bad == "text":
        base = _scn("syn-bad-0", _lines("testonly"), question="Q?", answer="A.", gold=(1,))
    elif bad == "overlap":
        base = _scn("syn-bad-0", _lines("devonly")[:6] + _lines("fresh", 2), question="Q?", answer="A.", gold=(1,))
    _write(root, "train", [*load_scenarios("train", root), base])
    with pytest.raises(ValueError):
        build_examples(*data, tmp_path / "cards", train_scenarios=root, held_out_dir=held)


def _ex(i: int, source: str, status: str, split: str = "train") -> Example:
    if status == "abstained":
        card = Card(mode="recall", status=status, findings=[], missing=["q"])
    else:
        ref = EvidenceRef(turn_id="t1", quote="hi", speaker="A", start_ms=0)
        card = Card(mode="recall", status=status, findings=[Finding(
            agent="recall", kind="recall", statement="s", evidence=[ref], confidence=1.0)])
    return Example(id=f"{source}-{status}-{i}", source=source, split=split, conversation_id=f"c{i}",
                   messages=[{"role": "user", "content": f"text {i}"}], completion=card.model_dump_json())


def _mix(n_ans: int, n_abs_q: int, n_abs_s: int, n_ans_s: int = 0) -> list[Example]:
    return ([_ex(i, "qmsum", "answered") for i in range(n_ans)]
            + [_ex(i, "scenarios_train", "answered") for i in range(n_ans_s)]
            + [_ex(i, "qmsum", "abstained") for i in range(n_abs_q)]
            + [_ex(i, "scenarios_train", "abstained") for i in range(n_abs_s)])


def _abstained(exs: list[Example]) -> list[Example]:
    return [e for e in exs if '"abstained"' in e.completion]


def test_cap_limits_abstained_share_and_is_deterministic():
    from contextlens.cards.dataset import cap_abstained
    exs = _mix(n_ans=20, n_abs_q=30, n_abs_s=10, n_ans_s=10)
    a, b = cap_abstained(exs, 0.25, seed=3), cap_abstained(exs, 0.25, seed=3)
    assert [e.id for e in a] == [e.id for e in b]
    assert len(_abstained(a)) / len(a) <= 0.25
    assert len(_abstained(a)) == 10  # 30 answered -> floor(30 / 3)
    assert len([e for e in a if e not in _abstained(a)]) == 30  # every answered example kept
    assert [e.id for e in a] != [e.id for e in cap_abstained(exs, 0.25, seed=4)]
    assert {e.id for e in a} <= {e.id for e in exs}
    assert all(x in exs for x in a)  # content untouched


def test_cap_keeps_every_source_with_abstained_examples():
    from contextlens.cards.dataset import cap_abstained
    exs = _mix(n_ans=30, n_abs_q=100, n_abs_s=2)  # cap allows 10
    out = _abstained(cap_abstained(exs, 0.25, seed=0))
    assert len(out) == 10
    assert Counter(e.source for e in out)["scenarios_train"] >= 1


def test_cap_drops_nothing_when_under_the_cap_and_handles_empty():
    from contextlens.cards.dataset import cap_abstained
    exs = _mix(n_ans=30, n_abs_q=5, n_abs_s=3)
    assert cap_abstained(exs, 0.25, seed=0) == exs
    assert cap_abstained([], 0.25, seed=0) == []
    only_abs = _mix(n_ans=0, n_abs_q=4, n_abs_s=0)
    assert cap_abstained(only_abs, 0.25, seed=0) == []  # no answered example: none may stay


def test_build_cap_applies_to_train_only_and_records_stats(data, tmp_path):
    raw, splits = data
    base_stats: Counter = Counter()
    base = build_examples(raw, splits, tmp_path / "a", stats=base_stats)
    st: Counter = Counter()
    out = tmp_path / "b"
    counts = build_examples(raw, splits, out, stats=st, abstain_cap=0.05, cap_seed=1)
    assert counts["val"] == base["val"]
    assert (out / "qmsum-val.jsonl").read_text(encoding="utf-8") == (
        tmp_path / "a" / "qmsum-val.jsonl").read_text(encoding="utf-8")
    train = load_examples("train", out)
    assert len(train) == counts["train"] == st["train_final_size"]
    n_abs = len(_abstained(train))
    assert st["train_class_abstained"] == n_abs and st["train_class_answered"] == len(train) - n_abs
    assert st["train_source_qmsum"] == len(train) and st["train_synthetic_share"] == 0
    assert n_abs * 3 <= len(train) - n_abs
    assert st["train_abstained_dropped_by_cap"] == base["train"] - counts["train"] > 0
