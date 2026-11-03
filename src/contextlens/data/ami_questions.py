"""E7's questions, derived from AMI's own annotations by a fixed rule. No speech output and no model is involved.

Rule (also written in docs/ami-questions.md):
1. Only meetings in the `dev` split of datasets/splits/ami.json are used; no train or test meeting is read.
2. AMI's manual abstractive summary of a meeting has three sections of sentences that answer a general question:
   `decisions`, `actions` and `problems`. A section qualifies for a meeting when at least one of its sentences is
   linked (by the corpus's own summlink file) to a dialogue act whose first word falls inside a transcript turn.
   Only meetings where all three sections qualify are candidates.
3. Candidates are sorted by meeting id and `N_MEETINGS` of them are drawn with random.Random(SEED). Each drawn
   meeting gives one question per section, so the questions spread over meetings.
4. The question is a fixed sentence per section, naming no content of the answer. The gold answer is the section's
   summary sentences in corpus order. The gold evidence is the clean-transcript turns holding the linked dialogue
   acts' first words, in turn order. `as_of_seq` is the last of those turns, so the question is asked when the last
   piece of evidence has been said and every evidence turn is at or before it.
5. `labels_reviewed` is false until a person has read a sample (H2).

Data: AMI Meeting Corpus (https://groups.inf.ed.ac.uk/ami/corpus/), manual annotations v1.6.2, CC BY 4.0.
"""
import json
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from contextlens.data.loaders import load_dataset
from contextlens.schemas import Turn

SEED = 13
N_MEETINGS = 11
SPLIT = "dev"
SECTIONS = {
    "decisions": "What did the team decide so far?",
    "actions": "What actions were assigned to team members so far?",
    "problems": "What problems or open issues were raised so far?",
}
ATTRIBUTION = ("AMI Meeting Corpus, manual annotations v1.6.2, CC BY 4.0, https://groups.inf.ed.ac.uk/ami/corpus/. "
               "Questions are derived from its abstractive summaries and summary links; see docs/ami-questions.md.")
_NITE = "{http://nite.sourceforge.net/}"
_HREF = re.compile(r"#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?")
_DA_FILE = re.compile(r"^[^.]+\.([A-Z])\.dialog-act\.xml$")


def _word_times(path: Path) -> tuple[list[str], dict[str, tuple[int, int | None]]]:
    """Element ids in file order, and for each its index and start time in ms (None when untimed)."""
    ids, info = [], {}
    for el in ET.parse(path).getroot():
        wid = el.get(_NITE + "id")
        start = el.get("starttime")
        info[wid] = (len(ids), round(float(start) * 1000) if start else None)
        ids.append(wid)
    return ids, info


def _first_start(raw: Path, meeting: str, speaker: str, first: str, last: str, cache: dict) -> int | None:
    if speaker not in cache:
        cache[speaker] = _word_times(raw / "words" / f"{meeting}.{speaker}.words.xml")
    ids, info = cache[speaker]
    if first not in info or last not in info:
        return None
    for wid in ids[info[first][0] : info[last][0] + 1]:
        if info[wid][1] is not None:
            return info[wid][1]
    return None


def _dialogue_acts(raw: Path, meeting: str) -> dict[str, tuple[str, str, str]]:
    out = {}
    for path in sorted((raw / "dialogueActs").glob(f"{meeting}.*.dialog-act.xml")):
        speaker = _DA_FILE.match(path.name).group(1)  # type: ignore[union-attr]
        for da in ET.parse(path).getroot().iter("dact"):
            child = da.find(_NITE + "child")
            m = _HREF.search(child.get("href")) if child is not None else None
            if m:
                out[da.get(_NITE + "id")] = (speaker, m.group(1), m.group(2) or m.group(1))
    return out


def _sentences(raw: Path, meeting: str) -> dict[str, tuple[str, str]]:
    """Sentence id -> (section, text) for the sections the rule uses, in corpus order."""
    root = ET.parse(raw / "abstractive" / f"{meeting}.abssumm.xml").getroot()
    return {s.get(_NITE + "id"): (sec.tag, " ".join((s.text or "").split()))
            for sec in root if sec.tag in SECTIONS for s in sec.iter("sentence")}


def _links(raw: Path, meeting: str) -> list[tuple[str, str]]:
    """(dialogue act id, abstractive sentence id) pairs from the summlink file."""
    out = []
    for link in ET.parse(raw / "extractive" / f"{meeting}.summlink.xml").getroot().iter("summlink"):
        ends = {p.get("role"): _HREF.search(p.get("href")).group(1)  # type: ignore[union-attr]
                for p in link.iter(_NITE + "pointer")}
        if "extractive" in ends and "abstractive" in ends:
            out.append((ends["extractive"], ends["abstractive"]))
    return out


def meeting_questions(raw: Path, meeting: str, turns: list[Turn]) -> list[dict[str, Any]]:
    """The questions of one meeting, one per qualifying section; empty if a source file is missing."""
    try:
        sentences, links, acts = _sentences(raw, meeting), _links(raw, meeting), _dialogue_acts(raw, meeting)
    except (FileNotFoundError, ET.ParseError):
        return []
    cache: dict = {}
    by_section: dict[str, tuple[list[str], set[int]]] = {s: ([], set()) for s in SECTIONS}
    linked: dict[str, set[int]] = {}
    for da_id, sent_id in links:
        if da_id not in acts or sent_id not in sentences:
            continue
        speaker, first, last = acts[da_id]
        start = _first_start(raw, meeting, speaker, first, last, cache)
        seq = next((t.seq for t in turns if t.speaker == speaker and start is not None
                    and t.start_ms <= start <= t.end_ms), None)
        if seq is not None:
            linked.setdefault(sent_id, set()).add(seq)
    for sent_id, (section, text) in sentences.items():
        if sent_id in linked and text:
            by_section[section][0].append(text)
            by_section[section][1].update(linked[sent_id])
    out = []
    for section, (texts, seqs) in by_section.items():
        if texts:
            ordered = sorted(seqs)
            out.append({"meeting": meeting, "as_of_seq": ordered[-1], "question": SECTIONS[section],
                        "category": section, "gold_evidence_turn_ids": [f"{meeting}:t{s}" for s in ordered],
                        "gold_label": None, "gold_answer": " ".join(texts)})
    return out


def generate(raw_root: Path, splits_path: Path, n_meetings: int = N_MEETINGS, seed: int = SEED) -> list[dict[str, Any]]:
    """Questions for `n_meetings` dev meetings, or for fewer when fewer have all three sections."""
    dev = set(json.loads(splits_path.read_text(encoding="utf-8"))["splits"][SPLIT])
    raw = raw_root / "ami"
    conversations = {c.id: c for c in load_dataset("ami", raw_root) if c.id in dev}
    candidates = []
    for meeting in sorted(conversations):
        qs = meeting_questions(raw, meeting, conversations[meeting].turns)
        if {q["category"] for q in qs} == set(SECTIONS):
            candidates.append(qs)
    chosen = random.Random(seed).sample(candidates, min(n_meetings, len(candidates)))
    return [q for qs in sorted(chosen, key=lambda qs: qs[0]["meeting"]) for q in qs]


def write_questions(questions: list[dict[str, Any]], path: Path) -> None:
    doc = {"source": ATTRIBUTION, "seed": SEED, "labels_reviewed": False, "questions": questions}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")


def review_sample(questions: list[dict[str, Any]], turns_by_id: dict[str, Turn], n: int = 15, seed: int = SEED) -> str:
    """A fixed-seed sample of questions with their gold answer and the evidence turns quoted, for a person to read."""
    picked = random.Random(seed).sample(range(len(questions)), min(n, len(questions)))
    lines = ["# E7 question sample for review", "",
             f"{len(picked)} of {len(questions)} questions, drawn with seed {seed}. For each: is the gold answer "
             "supported by the quoted turns, and are those turns the place to find it? Source: " + ATTRIBUTION, ""]
    for k, i in enumerate(sorted(picked), start=1):
        q = questions[i]
        ids = q["gold_evidence_turn_ids"]
        lines += [f"## {k}. {q['meeting']} ({q['category']}), asked after turn {q['as_of_seq']}", "",
                  f"Question: {q['question']}", "", f"Gold answer: {q['gold_answer']}", "",
                  f"Evidence ({len(ids)} turns" + (", first 8 shown" if len(ids) > 8 else "") + "):"]
        lines += [f"- {turns_by_id[t].speaker} [{turns_by_id[t].start_ms // 1000} s]: {turns_by_id[t].text}"
                  for t in ids[:8]]
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from contextlens.eval.records import ROOT

    ap = argparse.ArgumentParser(prog="ami_questions")
    ap.add_argument("--raw", type=Path, default=ROOT / "datasets/raw", help="folder holding ami/")
    a = ap.parse_args(argv)
    questions = generate(a.raw, ROOT / "datasets/splits/ami.json")
    out = ROOT / "datasets/ami-questions"
    write_questions(questions, out / "dev.json")
    turns = {t.id: t for c in load_dataset("ami", a.raw) for t in c.turns
             if c.id in {q["meeting"] for q in questions}}
    (out / "review-sample.md").write_text(review_sample(questions, turns), encoding="utf-8")
    print(len(questions), "questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
