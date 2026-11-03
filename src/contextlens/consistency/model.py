"""The trained consistency model: a sequence classifier over (statement, recent earlier turns), its checkpoint
format, and `TrainedChecker`, which implements `ConsistencyChecker`."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contextlens.data.loaders import Pair
from contextlens.schemas import ConsistencyLabel, Scenario, Turn

BASE_MODEL = "microsoft/deberta-v3-base"  # MIT, see docs/datasets.md
LABELS: tuple[ConsistencyLabel, ...] = ("consistent", "update", "contradiction", "insufficient")
MAX_LENGTH = 256
CONTEXT_TURNS = 6
LABEL_MAP_FILE = "label_map.json"
THRESHOLD = 0.5  # below this top probability the checker abstains with "insufficient"


@dataclass(frozen=True)
class Example:
    statement: str
    speaker: str
    earlier: list[tuple[str, str]]  # (speaker, text), oldest first
    label: str | None
    group: str  # the conversation id
    source: str


def example_from_turns(statement: Turn, earlier: list[Turn], label: str | None = None) -> Example:
    return Example(statement.text, statement.speaker, [(t.speaker, t.text) for t in earlier], label,
                   statement.conversation_id, "turns")


def examples_from_scenarios(scenarios: list[Scenario], source: str = "scenarios") -> list[Example]:
    """One example per labelled scenario: the turn at as_of_seq is the statement, the turns before it the context."""
    out = []
    for s in scenarios:
        if s.gold_label is None:
            continue
        turns = [t for t in s.conversation.turns if t.seq <= s.request.as_of_seq]
        out.append(Example(turns[-1].text, turns[-1].speaker, [(t.speaker, t.text) for t in turns[:-1]],
                           s.gold_label, s.conversation.id, source))
    return out


def examples_from_pairs(pairs: list[Pair]) -> list[Example]:
    """Pairs hold no speakers, so the two sides of a dialogue are called A and B, counting back from the statement."""
    out = []
    for p in pairs:
        m = len(p.earlier)
        earlier = [("A" if (m - j) % 2 == 0 else "B", t) for j, t in enumerate(p.earlier)]
        out.append(Example(p.statement, "A", earlier, p.label, p.conversation_id, p.source))
    return out


def render(ex: Example) -> tuple[str, str]:
    """(statement, recent context) as two text segments for the tokenizer."""
    return f"{ex.speaker}: {ex.statement}", " ".join(f"{s}: {t}" for s, t in ex.earlier[-CONTEXT_TURNS:])


def load_pretrained(name_or_path: str, labels: tuple[str, ...] = LABELS) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # dtype is explicit: the stored DeBERTa weights are fp16 and are otherwise loaded as fp16, where AdamW's eps
    # underflows to 0 and the first optimizer step turns the weights into NaN. Training keeps fp32 weights.
    model = AutoModelForSequenceClassification.from_pretrained(
        name_or_path, dtype=torch.float32, num_labels=len(labels), id2label=dict(enumerate(labels)),
        label2id={n: i for i, n in enumerate(labels)})
    return model, AutoTokenizer.from_pretrained(name_or_path)


def save_checkpoint(model: Any, tokenizer: Any, directory: Path, meta: dict[str, Any] | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    (directory / LABEL_MAP_FILE).write_text(
        json.dumps({"labels": labels, "max_length": MAX_LENGTH, "meta": meta or {}}, indent=1) + "\n", encoding="utf-8")


def load_checkpoint(directory: Path) -> tuple[Any, Any, tuple[str, ...]]:
    saved = json.loads((directory / LABEL_MAP_FILE).read_text(encoding="utf-8"))
    model, tokenizer = load_pretrained(str(directory))
    labels = tuple(saved["labels"])
    if labels != tuple(model.config.id2label[i] for i in range(model.config.num_labels)):
        raise ValueError(f"the label map in {directory} does not match the model's output order")
    return model, tokenizer, labels


def predict_proba(model: Any, tokenizer: Any, examples: list[Example], batch_size: int = 32) -> list[list[float]]:
    import torch

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    out: list[list[float]] = []
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            first, second = zip(*(render(e) for e in examples[i : i + batch_size]))
            enc = tokenizer(list(first), list(second), truncation=True, max_length=MAX_LENGTH, padding=True,
                            return_tensors="pt").to(device)
            out += torch.softmax(model(**enc).logits.float(), dim=-1).tolist()
    model.train(was_training)
    return out


class TrainedChecker:
    """Gives the most probable label and its probability. When even the top probability is below `threshold` it
    abstains: it returns "insufficient" with the model's probability for that label."""

    def __init__(self, model: Any, tokenizer: Any, labels: tuple[str, ...] = LABELS, threshold: float = THRESHOLD) -> None:
        self.model, self.tokenizer, self.labels, self.threshold = model, tokenizer, labels, threshold

    @classmethod
    def from_checkpoint(cls, directory: Path, threshold: float = THRESHOLD, device: str = "cpu") -> "TrainedChecker":
        model, tokenizer, labels = load_checkpoint(directory)
        return cls(model.to(device), tokenizer, labels, threshold)

    def check(self, statement: Turn, earlier: list[Turn]) -> tuple[ConsistencyLabel, float]:
        probs = predict_proba(self.model, self.tokenizer, [example_from_turns(statement, earlier)])[0]
        best = max(range(len(probs)), key=probs.__getitem__)
        if probs[best] < self.threshold:
            return "insufficient", probs[self.labels.index("insufficient")]
        return self.labels[best], probs[best]  # type: ignore[return-value]
