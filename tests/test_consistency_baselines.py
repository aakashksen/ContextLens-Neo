import json
import re
from pathlib import Path

import pytest

from contextlens.consistency.baselines import TfidfChecker, ZeroShotChecker
from contextlens.consistency.data import DECODE_LABEL_MAP, build_pairs, source_counts
from contextlens.data.loaders import Pair
from contextlens.data.scenarios import SCENARIOS_DIR
from contextlens.schemas import Turn

RAW = Path(__file__).parent / "fixtures" / "data"
DOCS = Path(__file__).resolve().parents[1] / "docs" / "datasets.md"


def _scenario_dir(tmp_path: Path) -> Path:
    """Hand-written scenarios in a dev file and a test file; invented text."""
    def sc(sid: str, split: str, label: str | None, lines: list[str]) -> str:
        turns = [{"id": f"{sid}:t{i}", "conversation_id": sid, "seq": i, "speaker": "A", "text": t,
                  "start_ms": i * 1000, "end_ms": i * 1000 + 900, "source": "synthetic"} for i, t in enumerate(lines)]
        return json.dumps({
            "id": sid, "conversation": {"id": sid, "turns": turns, "source": "synthetic", "split": split},
            "request": {"conversation_id": sid, "as_of_seq": len(lines) - 1}, "category": "x",
            "gold_evidence_turn_ids": [], "gold_label": label, "gold_answer": None, "origin": "synthetic"})
    dev = [sc("s-con", "dev", "contradiction", ["The deadline is Monday.", "Ok.", "The deadline is never."]),
           sc("s-upd", "dev", "update", ["The deadline is Monday.", "Ok.", "We moved the deadline to Wednesday."]),
           sc("s-ins", "dev", "insufficient", ["Who owns billing?", "Not sure.", "Maybe somebody."]),
           sc("s-qa", "dev", None, ["What was decided?"])]
    test = [sc("s-con2", "test", "contradiction", ["Budget is ten.", "Fine.", "Budget is zero."]),
            sc("s-ok", "test", "consistent", ["Budget is ten.", "Fine.", "Ten it is."])]
    (tmp_path / "dev.jsonl").write_text("\n".join(dev) + "\n", encoding="utf-8")
    (tmp_path / "test.jsonl").write_text("\n".join(test) + "\n", encoding="utf-8")
    return tmp_path


def _all(tmp_path: Path) -> dict[str, list[Pair]]:
    sd = _scenario_dir(tmp_path)
    return {s: build_pairs(s, RAW, sd) for s in ("train", "dev", "test")}


def test_conversation_split_preserved(tmp_path):
    by_split = _all(tmp_path)
    where: dict[str, set[str]] = {}
    for split, pairs in by_split.items():
        assert all(p.split == split for p in pairs)
        for p in pairs:
            where.setdefault(p.conversation_id, set()).add(split)
    assert all(len(v) == 1 for v in where.values())
    # c1 has two records and both stay in train; c5 is in dev and test upstream and ends up in test only
    assert [p.conversation_id for p in by_split["train"] if p.source == "decode"].count("c1") == 2
    assert where["c5"] == {"test"} and "c5" not in {p.conversation_id for p in by_split["dev"]}


def test_label_map_matches_datasets_doc():
    text = DOCS.read_text(encoding="utf-8")
    section = text.split("### DECODE label map and splits", 1)[1].split("\n## ", 1)[0]
    rows = re.findall(r"^\| `(\w+)` \| `(\w+)` \|", section, flags=re.M)
    assert dict(rows) == DECODE_LABEL_MAP
    assert "update" not in DECODE_LABEL_MAP.values() and "insufficient" not in DECODE_LABEL_MAP.values()


def test_sources_are_tracked(tmp_path):
    by_split = _all(tmp_path)
    assert source_counts(by_split["dev"]) == {
        "decode": {"contradiction": 1}, "scenarios": {"contradiction": 1, "insufficient": 1, "update": 1}}
    assert source_counts(by_split["test"]) == {
        "decode": {"consistent": 2, "contradiction": 2}, "scenarios": {"consistent": 1, "contradiction": 1}}
    assert source_counts(by_split["train"]) == {"decode": {"consistent": 2, "contradiction": 2}}


def test_default_scenarios_dir_is_the_written_one():
    assert (SCENARIOS_DIR / "dev.jsonl").is_file()


def _turn(i: int, text: str) -> Turn:
    return Turn(id=f"t{i}", conversation_id="c", seq=i, speaker="A", text=text, start_ms=i, end_ms=i + 1, source="dataset")


FOUR = [
    ("The deadline is Monday.", "The deadline is Monday and never changes.", "contradiction"),
    ("The deadline is Monday.", "We moved the deadline to Wednesday.", "update"),
    ("Who owns billing?", "Maybe somebody will know eventually.", "insufficient"),
    ("The deadline is Monday.", "Monday works fine for the deadline.", "consistent"),
]


def test_tfidf_fits_and_predicts_all_four_labels():
    pairs = [Pair(conversation_id=f"c{i}-{k}", statement=s, earlier=[e], label=lab, source="scenarios")
             for k in range(3) for i, (e, s, lab) in enumerate(FOUR)]
    checker = TfidfChecker().fit(pairs)
    got = {lab: checker.check(_turn(1, s), [_turn(0, e)])[0] for e, s, lab in FOUR}
    assert got == {lab: lab for _, _, lab in FOUR}
    assert 0.0 < checker.check(_turn(1, FOUR[0][1]), [_turn(0, FOUR[0][0])])[1] <= 1.0


def test_tfidf_rejects_unmapped_labels_and_unfitted_use():
    with pytest.raises(ValueError):
        TfidfChecker().fit([Pair(conversation_id="c", statement="x", earlier=[], label="non_contradiction", source="decode")])
    with pytest.raises(RuntimeError):
        TfidfChecker().check(_turn(1, "x"), [_turn(0, "y")])


def test_zero_shot_with_stub_scorer():
    def scorer(pairs):  # [contradiction, entailment, neutral]
        return [[0.9, 0.05, 0.05] if "never" in hyp else [0.05, 0.9, 0.05] for _, hyp in pairs]
    z = ZeroShotChecker(scorer=scorer)
    assert z.check(_turn(1, "I never had one."), [_turn(0, "I had a dog.")]) == ("contradiction", 0.9)
    label, conf = z.check(_turn(1, "A beagle."), [_turn(0, "I had a dog.")])
    assert label == "consistent" and conf == pytest.approx(0.95)
    assert z.check(_turn(0, "Hello."), [])[0] == "insufficient"


@pytest.mark.real
def test_zero_shot_real_model_flags_a_clear_contradiction():
    z = ZeroShotChecker()
    assert z.check(_turn(1, "I have never owned a pet."), [_turn(0, "I have a dog named Rex.")])[0] == "contradiction"
