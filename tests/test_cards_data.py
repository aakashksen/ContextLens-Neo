import json
import re
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from contextlens.cards.dataset import Example, build_examples
from contextlens.cards.train_lora import attach_lora, run_steps, tokenize_example
from contextlens.llm.untrusted import unescape
from contextlens.schemas import Card

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


@pytest.mark.real
def test_real_model_loads_under_peft_and_vram_limit():
    from contextlens.cards.train_lora import train

    meta = train(Path("models/cards-check"), steps=1)
    assert meta["peak_vram_gb"] <= 7.0
