import subprocess
from pathlib import Path

from contextlens.consistency import hand_read as hr
from contextlens.consistency.select_cascade import Case
from contextlens.schemas import ButtonRequest, Conversation, Scenario, Turn

ROOT = Path(__file__).resolve().parents[1]


def scn(sid: str) -> Scenario:
    turns = [Turn(id=f"{sid}:{i}", conversation_id="c", seq=i, speaker="AB"[i % 2], text=f"line {sid} {i}",
                  start_ms=i, end_ms=i + 1, source="dataset") for i in range(6)]
    return Scenario(id=sid, conversation=Conversation(id="c", turns=turns, source="decode", split="dev"),
                    request=ButtonRequest(conversation_id="c", as_of_seq=4), category="decode",
                    gold_evidence_turn_ids=[], gold_label="consistent", gold_answer=None, origin="dataset")


def test_file_holds_ids_gold_and_the_statement_with_its_last_earlier_turns(tmp_path: Path) -> None:
    out = tmp_path / "fa.md"
    sample = [Case("decode", "a", False, 0.8, True)]
    ids = hr.write_hand_read(out, sample, {"decode": [scn("a"), scn("zz")]})
    assert ids == ["a"]
    text = out.read_text(encoding="utf-8")
    assert "## 1. a (decode)" in text and "gold: consistent" in text
    assert "line a 4" in text and "line a 2" in text and "line a 1" not in text  # statement plus 2 earlier turns
    assert "line a 5" not in text  # turns after as_of_seq are not shown


def test_the_output_folder_is_git_ignored() -> None:
    r = subprocess.run(["git", "check-ignore", "-q", "results/local/false-alarms.md"], cwd=ROOT)
    assert r.returncode == 0
