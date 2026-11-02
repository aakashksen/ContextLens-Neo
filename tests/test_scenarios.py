import json
from collections import Counter
from pathlib import Path

from contextlens.data.scenarios import load_scenarios

ROOT = Path(__file__).parents[1]
CATEGORIES = {
    "recall", "contradiction", "legitimate_update", "insufficient_evidence", "wrong_speaker_trap",
    "hypothetical_or_joking", "prompt_injection", "respond_with_suggestion",
}


def _all():
    return {s: load_scenarios(s, ROOT / "datasets" / "scenarios") for s in ("dev", "test")}


def test_counts_per_category_in_dev_and_test():
    for split, scs in _all().items():
        assert len(scs) >= 120, split
        counts = Counter(s.category for s in scs)
        assert set(counts) == CATEGORIES
        assert min(counts.values()) >= 15, (split, counts)
        assert all(s.origin in ("dataset", "synthetic") for s in scs)
        assert len({s.id for s in scs}) == len(scs)


def test_gold_evidence_turns_exist_before_as_of_seq():
    for scs in _all().values():
        for s in scs:
            seqs = {t.id: t.seq for t in s.conversation.turns}
            assert s.request.as_of_seq == max(seqs.values())
            assert s.request.conversation_id == s.conversation.id
            for tid in s.gold_evidence_turn_ids:
                assert tid in seqs and seqs[tid] <= s.request.as_of_seq, (s.id, tid)
            if s.category != "insufficient_evidence":
                assert s.gold_evidence_turn_ids, s.id
            if s.category == "insufficient_evidence":
                assert s.gold_label == "insufficient" and not s.gold_evidence_turn_ids
            if s.origin == "synthetic":
                assert all(t.source == "synthetic" for t in s.conversation.turns)


def test_split_by_conversation():
    data = _all()
    ids = {k: {s.conversation.id for s in v} for k, v in data.items()}
    assert not ids["dev"] & ids["test"]
    manifest = json.loads((ROOT / "datasets" / "splits" / "longmemeval.json").read_text())["splits"]
    for split, scs in data.items():
        for s in scs:
            assert s.conversation.split == split
            if s.origin == "dataset":
                assert s.conversation.id in manifest[split], s.id


def _find(texts):
    out = []
    for scs in _all().values():
        for s in scs:
            if [t.text for t in s.conversation.turns] == texts:
                out.append(s)
    return out


def test_deadline_cases_have_expected_labels():
    upd = _find(["The deadline is Monday", "We moved the deadline to Wednesday"])
    con = _find(["The deadline is Monday", "We moved the deadline to Wednesday", "It has always been Wednesday"])
    assert [(s.gold_label, s.category) for s in upd] == [("update", "legitimate_update")]
    assert [(s.gold_label, s.category) for s in con] == [("contradiction", "contradiction")]
    assert upd[0].conversation.id != con[0].conversation.id
    assert upd[0].gold_evidence_turn_ids == [t.id for t in upd[0].conversation.turns]
    assert con[0].gold_evidence_turn_ids[-1] == con[0].conversation.turns[-1].id
