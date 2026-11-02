from pathlib import Path

import pytest

from contextlens.data.loaders import load_dataset, load_gold, load_pairs
from contextlens.data.splits import dataset_hash, make_splits
from contextlens.schemas import Conversation, Turn

FIX = Path(__file__).parent / "fixtures" / "data"


def _convs(n: int = 40, text: str = "hello") -> list[Conversation]:
    return [
        Conversation(
            id=f"c{i}",
            source="synthetic",
            turns=[
                Turn(id=f"c{i}:t0", conversation_id=f"c{i}", seq=0, speaker="A", text=text,
                     start_ms=0, end_ms=1, source="synthetic")
            ],
        )
        for i in range(n)
    ]


def test_no_conversation_in_two_splits():
    ids = [c.id for c in _convs()]
    s = make_splits(_convs())
    assert sorted(s["train"] + s["dev"] + s["test"]) == sorted(ids)
    assert not set(s["train"]) & set(s["dev"])
    assert not set(s["train"]) & set(s["test"])
    assert not set(s["dev"]) & set(s["test"])
    assert (len(s["train"]), len(s["dev"]), len(s["test"])) == (20, 10, 10)


def test_split_is_deterministic():
    assert make_splits(_convs()) == make_splits(list(reversed(_convs())))
    assert make_splits(_convs(), seed=1) != make_splits(_convs(), seed=2)


def test_hash_changes_when_text_changes():
    assert dataset_hash(_convs()) == dataset_hash(list(reversed(_convs())))
    assert dataset_hash(_convs()) != dataset_hash(_convs(text="hellO"))


def test_longmemeval_loader():
    convs = load_dataset("longmemeval", FIX)
    assert [c.id for c in convs] == ["q1", "q2_abs"]
    assert [t.text for t in convs[0].turns][:3] == ["I live in Paris.", "Nice city.", "I moved to Berlin."]
    assert [t.seq for t in convs[0].turns] == [0, 1, 2, 3]
    assert convs[0].turns[0].speaker == "user" and convs[0].turns[0].source == "dataset"
    gold = load_gold("longmemeval", FIX)
    s = gold[0]
    assert s.origin == "dataset" and s.category == "knowledge-update"
    assert s.gold_evidence_turn_ids == ["q1:t0", "q1:t2"]
    assert s.gold_answer == "Berlin" and s.gold_label is None
    assert s.request.question == "Where do I live now?" and s.request.as_of_seq == 3
    assert gold[1].gold_answer == "42" and gold[1].gold_evidence_turn_ids == []


def test_ami_loader():
    (c,) = load_dataset("ami", FIX)
    assert c.id == "ES0001a"
    assert [(t.speaker, t.text, t.start_ms, t.end_ms) for t in c.turns] == [
        ("A", "Hi, everyone", 1000, 1500),
        ("B", "Hello there", 3000, 3800),
        ("A", "Agreed", 9000, 9400),
    ]
    assert [t.seq for t in c.turns] == [0, 1, 2]
    assert all(t.source == "dataset" for t in c.turns)


def test_qmsum_loader():
    (c,) = load_dataset("qmsum", FIX)
    assert [t.speaker for t in c.turns] == ["Alice", "Bob", "Alice"]
    assert c.split == "dev"
    (s,) = load_gold("qmsum", FIX)
    assert s.origin == "dataset"
    assert s.gold_evidence_turn_ids == [f"{c.id}:t1", f"{c.id}:t2"]
    assert s.gold_answer == "Ship on Friday."
    assert s.request.question == "What did Alice decide?" and s.request.as_of_seq == 2


def test_decode_loader():
    pairs = load_pairs(FIX)
    assert [p.label for p in pairs] == ["contradiction", "non_contradiction"]
    p = pairs[0]
    assert p.statement == "I have never had a pet."
    assert p.earlier == ["I have a dog.", "What breed?"]
    assert p.conversation_id == "d1" and p.source == "decode"
    assert len(load_dataset("decode", FIX)) == 2


def test_unknown_dataset_rejected():
    with pytest.raises(ValueError):
        load_dataset("nope", FIX)
