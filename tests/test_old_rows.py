"""Rows written before Prediction gained cloud_supported must still rebuild to the same bytes, and a row that
does carry the field keeps it, null or not."""
import json
from pathlib import Path

from contextlens.experiments import rebuild_records
from contextlens.replay.runner import Prediction
from contextlens.schemas import Card

# the shape of a saved cards row: every field the model then had, none named cloud_supported
_ROW = Prediction(scenario_id="s1", card=Card(mode="recall", status="abstained", findings=[], missing=["no evidence"]),
                  claim_supported=[], stage_latency_ms={"llm": 5})
_FIELDS = json.loads(_ROW.model_dump_json())
OLD_ROW = json.dumps({k: v for k, v in _FIELDS.items() if k != "cloud_supported"}, separators=(",", ":"))


def _rec(tmp_path: Path, line: str) -> dict:
    (tmp_path / "rows.jsonl").write_bytes((line + "\n").encode("utf-8"))
    return {"predictions": {"recent": "rows.jsonl"}}


def test_a_row_without_cloud_supported_writes_back_to_the_same_bytes(tmp_path: Path) -> None:
    rows, hashes = rebuild_records._rows(tmp_path, _rec(tmp_path, OLD_ROW))
    assert rows["recent"][0].cloud_supported is None and "recent" in hashes
    assert "cloud_supported" not in rows["recent"][0].model_dump_json()


def test_a_row_with_an_explicit_null_keeps_it(tmp_path: Path) -> None:
    line = OLD_ROW.replace('"raw_output"', '"cloud_supported":null,"raw_output"')
    rows, _ = rebuild_records._rows(tmp_path, _rec(tmp_path, line))
    assert rows["recent"][0].model_dump_json() == line


def test_a_row_with_cloud_verdicts_keeps_them(tmp_path: Path) -> None:
    line = OLD_ROW.replace('"raw_output"', '"cloud_supported":[true],"raw_output"')
    rows, _ = rebuild_records._rows(tmp_path, _rec(tmp_path, line))
    assert rows["recent"][0].cloud_supported == [True]


def test_a_new_row_has_no_invented_value() -> None:
    p = Prediction(scenario_id="s", ranked_turn_ids=["t"])
    assert "cloud_supported" not in p.model_dump_json()
    assert '"cloud_supported":null' in Prediction(scenario_id="s", ranked_turn_ids=["t"], cloud_supported=None).model_dump_json()
