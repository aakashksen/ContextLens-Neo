"""E7's questions: generated from AMI's annotations by a fixed rule (docs/ami-questions.md)."""
import json
from pathlib import Path

import pytest

from contextlens.data import ami_questions as aq
from contextlens.experiments import e7_realtime as e7

ROOT = Path(__file__).resolve().parents[1]
NITE = 'xmlns:nite="http://nite.sourceforge.net/"'
RAW = ROOT / "datasets" / "raw"


def write_ami_fixture(raw: Path) -> None:
    """Meetings M1 (dev) and M3 (test) with the same annotations; speakers A and B."""
    ami = raw / "ami"
    words = {"A": [("hello", 1.0, 1.4), ("there", 1.5, 2.0), ("ship", 10.0, 10.4), ("friday", 10.5, 11.0)],
             "B": [("agreed", 5.0, 5.5), ("fine", 5.6, 6.0)]}
    segs = {"A": [(0, 1), (2, 3)], "B": [(0, 1)]}
    for d in ("words", "segments", "dialogueActs", "extractive", "abstractive"):
        (ami / d).mkdir(parents=True, exist_ok=True)
    for m in ("M1", "M3"):
        for spk, ws in words.items():
            body = "".join(f'<w nite:id="{m}.{spk}.words{i}" starttime="{a}" endtime="{b}">{w}</w>'
                           for i, (w, a, b) in enumerate(ws))
            (ami / "words" / f"{m}.{spk}.words.xml").write_text(f"<nite:root {NITE}>{body}</nite:root>")
            body = "".join(f'<segment nite:id="{m}.sync.{k}"><nite:child href="{m}.{spk}.words.xml#id({m}.{spk}.words{a})'
                           f'..id({m}.{spk}.words{b})"/></segment>' for k, (a, b) in enumerate(segs[spk]))
            (ami / "segments" / f"{m}.{spk}.segments.xml").write_text(f"<nite:root {NITE}>{body}</nite:root>")
        da = {"A": [(1, 0, 1), (2, 2, 3)], "B": [(1, 0, 0)]}
        for spk, items in da.items():
            body = "".join(f'<dact nite:id="{m}.{spk}.dialog-act.x.{k}"><nite:child href="{m}.{spk}.words.xml#id('
                           f'{m}.{spk}.words{a})..id({m}.{spk}.words{b})"/></dact>' for k, a, b in items)
            (ami / "dialogueActs" / f"{m}.{spk}.dialog-act.xml").write_text(f"<nite:root {NITE}>{body}</nite:root>")
        links = [("A", 1, 1), ("B", 1, 2), ("A", 2, 3)]  # (speaker, dialogue act, sentence)
        body = "".join(f'<summlink nite:id="{m}.l{k}"><nite:pointer role="extractive" href="{m}.{s}.dialog-act.xml#id('
                       f'{m}.{s}.dialog-act.x.{d})"/><nite:pointer role="abstractive" href="{m}.abssumm.xml#id('
                       f'{m}.s.{t})"/></summlink>' for k, (s, d, t) in enumerate(links))
        (ami / "extractive" / f"{m}.summlink.xml").write_text(f"<nite:root {NITE}>{body}</nite:root>")
        sec = {"abstract": [], "decisions": [(1, "Ship on Friday.")], "actions": [(2, "B will confirm.")],
               "problems": [(3, "Nobody owns QA.")]}
        body = "".join(f"<{name}>" + "".join(f'<sentence nite:id="{m}.s.{i}">{t}</sentence>' for i, t in items)
                       + f"</{name}>" for name, items in sec.items())
        (ami / "abstractive" / f"{m}.abssumm.xml").write_text(f"<nite:root {NITE}>{body}</nite:root>")


def test_generation_is_deterministic_dev_only_and_evidence_is_in_the_transcript(tmp_path: Path) -> None:
    write_ami_fixture(tmp_path / "raw")
    splits = tmp_path / "ami.json"
    splits.write_text(json.dumps({"splits": {"train": [], "dev": ["M1"], "test": ["M3"]}}))
    first = aq.generate(tmp_path / "raw", splits, n_meetings=5)
    assert first == aq.generate(tmp_path / "raw", splits, n_meetings=5)
    assert {q["meeting"] for q in first} == {"M1"}  # the test meeting has the same annotations and is not used
    by = {q["category"]: q for q in first}
    assert by["decisions"]["gold_answer"] == "Ship on Friday." and by["decisions"]["gold_evidence_turn_ids"] == ["M1:t0"]
    assert by["actions"]["gold_evidence_turn_ids"] == ["M1:t1"] and by["problems"]["as_of_seq"] == 2
    conv = {t.id: t for t in aq.load_dataset("ami", tmp_path / "raw")[0].turns}
    for q in first:
        assert all(i in conv and conv[i].seq <= q["as_of_seq"] for i in q["gold_evidence_turn_ids"])
    out = tmp_path / "out" / "dev.json"
    aq.write_questions(first, out)
    doc = json.loads(out.read_text())
    assert doc["labels_reviewed"] is False and "CC BY 4.0" in doc["source"]
    sample = aq.review_sample(first, conv, n=3)
    assert "Gold answer: Ship on Friday." in sample and sample == aq.review_sample(first, conv, n=3)


def test_the_committed_questions() -> None:
    doc = json.loads((ROOT / e7.QUESTIONS_REL).read_text(encoding="utf-8"))
    qs = doc["questions"]
    assert doc["labels_reviewed"] is False and "CC BY 4.0" in doc["source"]
    assert len(qs) >= 30 and len({q["meeting"] for q in qs}) >= 10
    dev = set(json.loads((ROOT / "datasets/splits/ami.json").read_text())["splits"]["dev"])
    assert {q["meeting"] for q in qs} <= dev
    for q in qs:
        assert q["gold_answer"] and q["category"] and q["gold_evidence_turn_ids"]
        seqs = [int(i.rsplit(":t", 1)[1]) for i in q["gold_evidence_turn_ids"]]
        assert max(seqs) == q["as_of_seq"] and all(i.startswith(q["meeting"] + ":t") for i in q["gold_evidence_turn_ids"])
    assert (ROOT / "datasets/ami-questions/review-sample.md").read_text(encoding="utf-8").count("\n## ") == 15


@pytest.mark.skipif(not (RAW / "ami/abstractive").is_dir(), reason="the local AMI annotations are not on this machine")
def test_committed_questions_regenerate_from_the_local_annotations() -> None:
    doc = json.loads((ROOT / e7.QUESTIONS_REL).read_text(encoding="utf-8"))
    assert aq.generate(RAW, ROOT / "datasets/splits/ami.json") == doc["questions"]
