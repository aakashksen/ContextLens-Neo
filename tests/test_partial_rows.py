import json

from contextlens.experiments import partial_rows as pr
from contextlens.replay.runner import Prediction
from contextlens.schemas import Card, EvidenceRef, Finding


def card(status="answered", mode="recall", statement="SECRET WORDS"):
    f = [Finding(agent="single", kind="recall", statement=statement, confidence=0.9,
                 evidence=[EvidenceRef(turn_id="t0", quote="q", speaker="a", start_ms=0)])] * (status == "answered")
    return Card(mode=mode, status=status, findings=f)


def line(commit, sid, c, system="single"):
    row = json.loads(Prediction(scenario_id=sid, card=c).model_dump_json())
    return json.dumps({"commit": commit, "subset": "scenarios", "winner": system, "model": "m", "calls": 0, "row": row})


def test_it_compares_single_rows_between_two_commits_and_prints_ids_only(tmp_path, capsys, monkeypatch):
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join([
        line("aaa", "c1", card()), line("bbb", "c1", card()),                              # same
        line("aaa", "c2", card()), line("bbb", "c2", card("abstained")),                    # status (and findings) differ
        line("aaa", "c3", card()), line("bbb", "c3", card(mode="clarify")),                 # mode differs
        line("aaa", "c4", card()), line("bbb", "c4", card(statement="other")),              # findings only
        line("aaa", "c5", card()), line("bbb", "c6", card()), line("bbb", "c1", card(), "team"),
        '{"commit": "aaa", "cut'], ) + "\n", encoding="utf-8")
    got = pr.compare(path, "aaa", "bbb")
    assert got["cases_in_both"] == 4 and got["identical"] == 1
    assert got["differ"]["status"]["ids"] == ["c2"] and got["differ"]["mode"]["ids"] == ["c3"]
    assert got["differ"]["findings"]["ids"] == ["c2", "c4"]
    assert got["only_in_first"] == ["c5"] and got["only_in_second"] == ["c6"]
    monkeypatch.setattr(pr, "ROOT", tmp_path)
    monkeypatch.setattr(pr, "PARTIAL_REL", "rows.jsonl")
    assert pr.main(["aaa", "bbb"]) == 0
    out = capsys.readouterr().out
    assert "c4" in out and "SECRET" not in out and "other" not in out
