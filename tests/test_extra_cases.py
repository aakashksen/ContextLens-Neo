import json
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from contextlens.data import extra_cases
from contextlens.data.check_splits import check as check_splits
from contextlens.data.scenarios import load_scenarios
from contextlens.eval.check_run import check, recompute_metrics
from contextlens.eval.records import SCENARIO_SOURCES, labels_reviewed, split_hash, write_run
from contextlens.replay.runner import Prediction
from contextlens.schemas import Card

ROOT = Path(__file__).parents[1]
EXTRA_DIR = ROOT / "datasets" / "scenarios_extra"
# Pinned: this work must not change the existing dev or test scenarios.
DEV_HASH = "bc875601bad1d7030a637e9e814f9041d4d345e6335c02df84d8080dccaed3c1"
TEST_HASH = "9c6e5a2088e88632a65d80f9be03110cb849c57298802ffec6ef149eb7ee6692"
# Shapes whose insufficient case may contain the asked value, because it is only a guess, a hypothetical,
# a detail of another thing, or stated after the cut.
VALUE_MAY_APPEAR = {"hypothetical", "speculation", "other-entity", "after-cut"}


def _extra():
    return load_scenarios("dev", EXTRA_DIR)


def _shape(sid: str) -> str:
    return sid.removeprefix("syn-xtra-").rsplit("-", 2)[0]


def _pairs():
    by_id = {s.id: s for s in _extra()}
    return [(by_id[i], by_id[i.removesuffix("-ins") + "-ans"]) for i in by_id if i.endswith("-ins")]


def test_existing_dev_and_test_files_are_untouched():
    scen = ROOT / "datasets" / "scenarios"
    assert split_hash("dev", ROOT) == DEV_HASH and split_hash("test", ROOT) == TEST_HASH
    manifest = json.loads((ROOT / "datasets/splits/scenarios.json").read_text(encoding="utf-8"))["splits"]
    assert manifest["dev"]["hash"] == DEV_HASH and manifest["test"]["hash"] == TEST_HASH
    assert set(manifest) == {"dev", "test"} and (scen / "dev.jsonl").is_file()


def test_the_file_is_what_the_generator_writes():
    want = "".join(s.model_dump_json() + "\n" for s in extra_cases.build_extra())
    assert (EXTRA_DIR / "dev.jsonl").read_bytes() == want.encode("utf-8")


def test_counts_and_shapes():
    pairs = _pairs()
    scs = _extra()
    assert len(pairs) >= 50 and len(scs) == 2 * len(pairs)
    assert Counter(s.category for s in scs) == {"insufficient_evidence": len(pairs), "recall": len(pairs)}
    assert set(Counter(_shape(i.id) for i, _ in pairs)) == set(extra_cases.SHAPES)
    lengths = [len(i.conversation.turns) for i, _ in pairs]
    assert min(lengths) <= 4 and sum(n >= 15 for n in lengths) >= 15


def test_each_case_is_well_formed():
    for s in _extra():
        turns = {t.id: t for t in s.conversation.turns}
        assert s.origin == "synthetic" and s.conversation.split == "dev" and s.conversation.source == "synthetic"
        assert all(t.source == "synthetic" for t in turns.values())
        assert s.request.conversation_id == s.conversation.id == s.id and s.request.question
        assert s.request.as_of_seq <= len(turns) - 1
        for tid in s.gold_evidence_turn_ids:
            assert turns[tid].seq <= s.request.as_of_seq
        if s.category == "insufficient_evidence":
            assert s.gold_label == "insufficient" and not s.gold_evidence_turn_ids and s.gold_answer is None
        else:
            assert s.gold_label is None and len(s.gold_evidence_turn_ids) == 1 and s.gold_answer


def test_a_pair_differs_only_in_whether_the_evidence_is_there():
    for ins, ans in _pairs():
        shape = _shape(ins.id)
        i_turns, a_turns = ins.conversation.turns, ans.conversation.turns
        assert len(i_turns) == len(a_turns) and ins.request.question == ans.request.question
        assert [t.speaker for t in i_turns] == [t.speaker for t in a_turns] or shape == "after-cut"
        i_text, a_text = [t.text for t in i_turns], [t.text for t in a_turns]
        if shape == "after-cut":
            assert sorted(i_text) == sorted(a_text) and i_text != a_text
            assert ins.request.as_of_seq < len(i_turns) - 1  # the detail comes after the cut
            assert ans.request.as_of_seq == len(a_turns) - 1
        else:
            assert sum(x != y for x, y in zip(i_text, a_text)) == 1, ins.id
            assert ins.request.as_of_seq == ans.request.as_of_seq
        value = ans.gold_answer.rstrip(".").lower()
        evidence = next(t for t in a_turns if t.id == ans.gold_evidence_turn_ids[0])
        assert value in evidence.text.lower()
        seen = " ".join(t.text.lower() for t in i_turns if t.seq <= ins.request.as_of_seq)
        if shape not in VALUE_MAY_APPEAR:
            assert value not in seen, ins.id


def test_no_overlap_with_dev_or_test():
    existing = [s for split in ("dev", "test") for s in load_scenarios(split, ROOT / "datasets" / "scenarios")]
    old_ids = {s.id for s in existing} | {s.conversation.id for s in existing}
    old_texts = {t.text.strip().lower() for s in existing for t in s.conversation.turns}
    old_questions = {s.request.question for s in existing}
    new = _extra()
    assert len({s.id for s in new}) == len(new) and not ({s.id for s in new} | {s.conversation.id for s in new}) & old_ids
    for s in new:
        assert s.request.question not in old_questions
        for t in s.conversation.turns:
            assert len(t.text.split()) < 5 or t.text.strip().lower() not in old_texts, (s.id, t.text)
    # no five-word run of a new turn appears in an existing turn
    def grams(text: str) -> set[tuple[str, ...]]:
        w = re.findall(r"[a-z0-9']+", text.lower())
        return {tuple(w[i:i + 5]) for i in range(len(w) - 4)}
    old_grams = set().union(*(grams(t) for t in old_texts))
    for s in new:
        for t in s.conversation.turns:
            assert not grams(t.text) & old_grams, (s.id, t.text)


def test_manifest_and_split_checks_cover_the_new_source():
    assert "scenarios_extra" in SCENARIO_SOURCES
    manifest = json.loads((ROOT / "datasets/splits/scenarios-extra.json").read_text(encoding="utf-8"))
    assert manifest["source"] == "scenarios_extra" and list(manifest["splits"]) == ["dev"]
    assert manifest["splits"]["dev"]["ids"] == [s.id for s in _extra()]
    assert manifest["splits"]["dev"]["hash"] == split_hash("dev", ROOT, "scenarios_extra")
    assert [p for p in check_splits() if "scenarios-extra" in p] == []


def test_extra_labels_never_count_as_reviewed(tmp_path: Path):
    (tmp_path / "orchestration").mkdir()
    (tmp_path / "orchestration/tasks.json").write_text(
        json.dumps({"tasks": [{"id": "H2", "owner": "human", "status": "done"}]}), encoding="utf-8")
    assert labels_reviewed(tmp_path, "scenarios") is True  # the same state that signs the ordinary scenarios
    assert labels_reviewed(tmp_path, "scenarios_extra") is False


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=root, check=True, capture_output=True)


def test_check_run_validates_a_record_of_the_new_source(tmp_path: Path):
    (tmp_path / "datasets/splits").mkdir(parents=True)
    shutil.copytree(EXTRA_DIR, tmp_path / "datasets/scenarios_extra")
    shutil.copy(ROOT / "datasets/splits/scenarios-extra.json", tmp_path / "datasets/splits/scenarios-extra.json")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    scs = _extra()
    preds = [Prediction(scenario_id=s.id, card=Card(mode="recall", status="abstained", findings=[]),
                        stage_latency_ms={"total": 10}) for s in scs]
    path = write_run("E1", "dev", {}, {"llm": "qwen-real"}, {"p": "v1"}, {"main": recompute_metrics(scs, preds)}, {},
                     [{"setting": "a", "ok": True}], {"main": preds}, split_hash("dev", tmp_path, "scenarios_extra"),
                     scenario_source="scenarios_extra", root=tmp_path)
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert rec["labels_reviewed"] is False and rec["label_source"] == "unreviewed_extra"
    assert check("E1", root=tmp_path) == []
