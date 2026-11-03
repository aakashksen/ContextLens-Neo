import hashlib
import json
from pathlib import Path

import pytest

from contextlens.consistency import rescore as rs
from contextlens.replay.runner import Prediction
from contextlens.schemas import ButtonRequest, Conversation, Scenario, Turn

LABELS = ("consistent", "update", "contradiction", "insufficient")
SECRET = "the-quick-brown-secret-text"


def scenario(sid: str, label: str | None = "contradiction") -> Scenario:
    turns = [Turn(id=f"{sid}:{i}", conversation_id=f"c-{sid}", seq=i, speaker="AB"[i % 2], text=f"{SECRET} {sid} {i}",
                  start_ms=i, end_ms=i + 1, source="dataset") for i in range(4)]
    return Scenario(id=sid, conversation=Conversation(id=f"c-{sid}", turns=turns, source="decode", split="dev"),
                    request=ButtonRequest(conversation_id=f"c-{sid}", as_of_seq=2), category="decode",
                    gold_evidence_turn_ids=[], gold_label=label, gold_answer=None, origin="dataset")  # type: ignore[arg-type]


def write_dev(tmp_path: Path, scs: list[Scenario], ids: list[str] | None = None) -> tuple[Path, Path]:
    data = tmp_path / "dev.jsonl"
    data.write_text("".join(s.model_dump_json() + "\n" for s in scs), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"splits": {"dev": {
        "hash": hashlib.sha256(data.read_bytes()).hexdigest(), "n": len(scs), "ids": ids or [s.id for s in scs]}}}))
    return manifest, data


def test_checkpoint_sha_must_match_every_record(tmp_path: Path) -> None:
    ck = tmp_path / "ck"
    ck.mkdir()
    (ck / "model.safetensors").write_bytes(b"weights")
    sha = hashlib.sha256(b"weights").hexdigest()
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"config": {"checkpoint_sha256": sha}}))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"config": {"checkpoint_sha256": "0" * 64}}))
    assert rs.verify_checkpoint(ck, [good, good]) == sha
    with pytest.raises(ValueError, match="does not match"):
        rs.verify_checkpoint(ck, [good, bad])


def test_dev_file_must_match_its_manifest_hash_and_ids(tmp_path: Path) -> None:
    scs = [scenario("a"), scenario("b")]
    manifest, data = write_dev(tmp_path, scs)
    assert [s.id for s in rs.load_dev(manifest, data)] == ["a", "b"]
    wrong_ids, data2 = write_dev(tmp_path, scs, ids=["a", "zzz"])
    with pytest.raises(ValueError, match="ids"):
        rs.load_dev(wrong_ids, data2)
    data.write_text(data.read_text() + "\n")
    with pytest.raises(ValueError, match="hash"):
        rs.load_dev(manifest, data)


def test_rescore_scores_the_statement_with_its_earlier_turns_and_keeps_ids_and_probabilities_only(tmp_path: Path) -> None:
    seen = []

    def proba(examples):
        seen.extend(examples)
        return [[0.1, 0.2, 0.6, 0.1] for _ in examples]

    rows = rs.rescore([scenario("a"), scenario("b", None)], proba, LABELS)
    assert rows == [{"id": s, "probs": dict(zip(LABELS, [0.1, 0.2, 0.6, 0.1]))} for s in ("a", "b")]
    assert seen[0].statement == f"{SECRET} a 2" and len(seen[0].earlier) == 2  # turns after as_of_seq are not seen
    out = tmp_path / "out.jsonl"
    rs.write_rows(out, rows)
    assert SECRET not in out.read_text()
    assert rs.read_rows(out) == rows


def test_agreement_with_the_saved_trained_predictions() -> None:
    rows = [{"id": "a", "probs": dict(zip(LABELS, [0.1, 0.1, 0.7, 0.1]))},
            {"id": "b", "probs": dict(zip(LABELS, [0.6, 0.1, 0.2, 0.1]))},
            {"id": "c", "probs": dict(zip(LABELS, [0.3, 0.3, 0.3, 0.1]))}]  # below 0.5: the trained checker abstains
    saved = [Prediction(scenario_id="a", label="contradiction"), Prediction(scenario_id="b", label="contradiction"),
             Prediction(scenario_id="c", label="consistent")]
    # DECODE rows are binary: contradiction or not
    assert rs.agreement(rows, saved, binary=True) == (2, 3)
    four = [Prediction(scenario_id="a", label="contradiction"), Prediction(scenario_id="b", label="consistent"),
            Prediction(scenario_id="c", label="insufficient")]
    assert rs.agreement(rows, four, binary=False) == (3, 3)


def test_agreement_stops_when_ids_differ() -> None:
    with pytest.raises(ValueError, match="ids"):
        rs.agreement([{"id": "a", "probs": {}}], [Prediction(scenario_id="b", label="consistent")], binary=True)


def test_a_manifest_without_ids_is_checked_by_hash_and_count(tmp_path: Path) -> None:
    manifest, data = write_dev(tmp_path, [scenario("a"), scenario("b")])
    entry = json.loads(manifest.read_text())
    del entry["splits"]["dev"]["ids"]
    manifest.write_text(json.dumps(entry))
    assert len(rs.load_dev(manifest, data)) == 2
    entry["splits"]["dev"]["n"] = 3
    manifest.write_text(json.dumps(entry))
    with pytest.raises(ValueError, match="count"):
        rs.load_dev(manifest, data)
