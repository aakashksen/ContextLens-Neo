from contextlens.context.budget import Section, fit_context
from contextlens.llm.client import FakeLLM
from contextlens.memory.extract import MemoryExtractor, MemoryStore
from contextlens.memory.summary import RollingSummary
from contextlens.schemas import MemoryItem, Turn


def turn(seq: int, text: str, speaker: str = "Ana") -> Turn:
    return Turn(
        id=f"t{seq}", conversation_id="c1", seq=seq, speaker=speaker, text=text,
        start_ms=seq * 1000, end_ms=seq * 1000 + 900, source="synthetic",
    )  # fmt: skip


T1 = turn(1, "Let us meet on Monday.")
T2 = turn(2, "Actually, make it Wednesday.", "Ben")


def item(kind="decision", subject="meeting day", statement="Meet on Monday", cites=("t1",)):
    return {"kind": kind, "subject": subject, "statement": statement, "source_turn_ids": list(cites)}


def correction_item():
    return item("correction", "meeting day", "Meet on Wednesday", ("t2",))


def build_store() -> MemoryStore:
    first = MemoryExtractor(FakeLLM([{"items": [item()]}])).update([T1], [])
    second = MemoryExtractor(FakeLLM([{"items": [correction_item()]}])).update([T2], first)
    store = MemoryStore()
    store.add(first + second)
    return store


def test_correction_links_supersedes():
    first = MemoryExtractor(FakeLLM([{"items": [item()]}])).update([T1], [])
    llm = FakeLLM([{"items": [correction_item()]}])
    second = MemoryExtractor(llm).update([T2], first)
    assert len(second) == 1
    assert second[0].supersedes == first[0].id
    assert second[0].kind == "correction"
    assert second[0].as_of_seq == 2
    assert "Wednesday" in second[0].statement
    # transcript text sits in a delimited data block, not in the system message
    assert "Wednesday" not in llm.calls[0][0]["content"]
    assert "<transcript>" in llm.calls[0][1]["content"]


def test_lookup_returns_current_value_by_default():
    got = build_store().lookup("c1", "meeting day", as_of_seq=2)
    assert [m.statement for m in got] == ["Meet on Wednesday"]
    # before the correction was said, the old value is still current
    early = build_store().lookup("c1", "meeting day", as_of_seq=1)
    assert [m.statement for m in early] == ["Meet on Monday"]


def test_history_still_available():
    got = build_store().lookup("c1", "meeting day", as_of_seq=2, include_superseded=True)
    assert [m.statement for m in got] == ["Meet on Monday", "Meet on Wednesday"]
    assert isinstance(got[0], MemoryItem)


def test_item_cites_real_turns():
    llm = FakeLLM([{"items": [
        item(cites=("t1", "t99")),
        item(subject="ghost", statement="Invented", cites=("t42",)),
        item(subject="none", statement="No cite", cites=()),
    ]}])  # fmt: skip
    out = MemoryExtractor(llm).update([T1], [])
    assert len(out) == 1
    assert out[0].source_turn_ids == ["t1"]
    assert out[0].speaker == "Ana"


def test_summary_is_not_citable():
    summary = RollingSummary(FakeLLM(["They plan to meet Monday."]))
    assert summary.update([T1]) == "They plan to meet Monday."
    assert summary.citable is False
    assert not hasattr(summary, "source_turn_ids")
    # a memory item that cites the summary instead of a turn is dropped
    out = MemoryExtractor(FakeLLM([{"items": [item(cites=("summary",))]}])).update([T1], [])
    assert out == []


def test_budget_keeps_evidence_turns():
    secs = [
        Section(text="a" * 40, seq=1, turn_id="t1", evidence=True),
        Section(text="b" * 40, seq=2, turn_id="t2"),
        Section(text="c" * 40, seq=3, turn_id="t3"),
        Section(text="d" * 40, seq=4, turn_id="t4"),
    ]
    out = fit_context(secs, max_tokens=20)  # 10 tokens each: room for two
    assert [s.turn_id for s in out] == ["t1", "t4"]
    tight = fit_context(secs, max_tokens=1)
    assert [s.turn_id for s in tight] == ["t1"]
