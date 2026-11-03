import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from contextlens.schemas import ButtonRequest, Conversation, Scenario, Turn

_NITE = "{http://nite.sourceforge.net/}"
_SPLIT_FROM_FILE = {"train": "train", "val": "dev", "valid": "dev", "dev": "dev", "test": "test"}
_HREF = re.compile(r"#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?")


class Pair(BaseModel):
    conversation_id: str
    statement: str
    earlier: list[str]
    label: Literal["contradiction", "non_contradiction"]
    source: str


def _turn(cid: str, seq: int, speaker: str, text: str, start_ms: int | None = None, end_ms: int | None = None) -> Turn:
    # Datasets without timings get a nominal 1 s per turn so ordering is still carried by time.
    start = seq * 1000 if start_ms is None else start_ms
    return Turn(id=f"{cid}:t{seq}", conversation_id=cid, seq=seq, speaker=speaker, text=text,
                start_ms=start, end_ms=start + 1000 if end_ms is None else end_ms, source="dataset")


def _longmemeval_records(root: Path) -> list[dict]:
    return json.loads((root / "longmemeval" / "longmemeval_oracle.json").read_text(encoding="utf-8"))


def _longmemeval_conv(rec: dict) -> tuple[Conversation, list[str]]:
    cid = rec["question_id"]
    turns: list[Turn] = []
    evidence: list[str] = []
    for session in rec["haystack_sessions"]:
        for msg in session:
            t = _turn(cid, len(turns), msg["role"], msg["content"])
            turns.append(t)
            if msg.get("has_answer"):
                evidence.append(t.id)
    return Conversation(id=cid, turns=turns, source="longmemeval"), evidence


def _load_longmemeval(root: Path) -> list[Conversation]:
    return [_longmemeval_conv(r)[0] for r in _longmemeval_records(root)]


def _gold_longmemeval(root: Path) -> list[Scenario]:
    out = []
    for rec in _longmemeval_records(root):
        conv, evidence = _longmemeval_conv(rec)
        out.append(Scenario(
            id=conv.id, conversation=conv, category=rec["question_type"],
            request=ButtonRequest(conversation_id=conv.id, as_of_seq=len(conv.turns) - 1, question=rec["question"]),
            gold_evidence_turn_ids=evidence, gold_label=None, gold_answer=str(rec["answer"]), origin="dataset",
        ))
    return out


def _load_ami(root: Path) -> list[Conversation]:
    base = root / "ami"
    meetings: dict[str, list[tuple[int, int, str, str]]] = {}
    for seg_path in sorted((base / "segments").glob("*.segments.xml")):
        meeting, speaker = seg_path.name.split(".")[:2]
        # Segments may start or end on non-word elements (disfmarker, gap, vocalsound), so index every element.
        items = list(ET.parse(base / "words" / f"{meeting}.{speaker}.words.xml").getroot())
        index = {el.get(_NITE + "id"): i for i, el in enumerate(items)}
        for seg in ET.parse(seg_path).getroot().iter("segment"):
            child = seg.find(_NITE + "child")
            m = _HREF.search(child.get("href")) if child is not None else None
            if not m or m.group(1) not in index or (m.group(2) or m.group(1)) not in index:
                continue
            first, last = index[m.group(1)], index[m.group(2) or m.group(1)]
            span = [
                (el.text.strip(), float(el.get("starttime")), float(el.get("endtime")), el.get("punc") == "true")
                for el in items[first : last + 1]
                if el.tag == "w" and el.text and el.text.strip() and el.get("starttime") and el.get("endtime")
            ]
            if not span:
                continue
            text = ""
            for tok, _, _, punc in span:
                text += tok if punc or not text else " " + tok
            meetings.setdefault(meeting, []).append(
                (round(span[0][1] * 1000), round(span[-1][2] * 1000), speaker, text)
            )
    convs = []
    for meeting, segs in sorted(meetings.items()):
        segs.sort(key=lambda s: (s[0], s[2]))
        convs.append(Conversation(
            id=meeting, source="ami",
            turns=[_turn(meeting, i, sp, tx, st, en) for i, (st, en, sp, tx) in enumerate(segs)],
        ))
    return convs


def _qmsum_records(root: Path) -> list[tuple[str, str | None, dict]]:
    out = []
    for path in sorted((root / "qmsum").glob("*.jsonl")):
        split = _SPLIT_FROM_FILE.get(path.stem)
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if line.strip():
                out.append((f"qmsum-{path.stem}-{i}", split, json.loads(line)))
    return out


def _qmsum_conv(cid: str, split: str | None, rec: dict) -> Conversation:
    turns = [_turn(cid, i, t["speaker"], t["content"]) for i, t in enumerate(rec["meeting_transcripts"])]
    return Conversation(id=cid, turns=turns, source="qmsum", split=split)


def _load_qmsum(root: Path) -> list[Conversation]:
    return [_qmsum_conv(*r) for r in _qmsum_records(root)]


def _gold_qmsum(root: Path) -> list[Scenario]:
    out = []
    for cid, split, rec in _qmsum_records(root):
        conv = _qmsum_conv(cid, split, rec)
        queries = [("general", q) for q in rec.get("general_query_list", [])]
        queries += [("specific", q) for q in rec.get("specific_query_list", [])]
        for n, (kind, q) in enumerate(queries):
            # Spans are [start, end] as strings, both ends inclusive, and may overlap (checked on the real data);
            # general queries have no span.
            indices = [i for start, end in q.get("relevant_text_span", []) for i in range(int(start), int(end) + 1)]
            evidence = [f"{cid}:t{i}" for i in dict.fromkeys(indices)]
            out.append(Scenario(
                id=f"{cid}-q{n}", conversation=conv, category=f"qmsum-{kind}",
                request=ButtonRequest(conversation_id=cid, as_of_seq=len(conv.turns) - 1, question=q["query"]),
                gold_evidence_turn_ids=evidence, gold_label=None, gold_answer=q.get("answer"), origin="dataset",
            ))
    return out


def _decode_records(root: Path) -> list[tuple[str | None, dict]]:
    out = []
    for path in sorted((root / "decode").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append((_SPLIT_FROM_FILE.get(path.stem), json.loads(line)))
    return out


def _load_decode(root: Path) -> list[Conversation]:
    return [
        Conversation(
            id=rec["id"], source="decode", split=split,
            turns=[_turn(rec["id"], i, t["speaker"], t["text"]) for i, t in enumerate(rec["dialogue"])],
        )
        for split, rec in _decode_records(root)
    ]


def load_pairs(root: Path) -> list[Pair]:
    pairs = []
    for _, rec in _decode_records(root):
        texts = [t["text"] for t in rec["dialogue"]]
        pairs.append(Pair(
            conversation_id=rec["id"], statement=texts[-1], earlier=texts[:-1], source="decode",
            label="contradiction" if rec["is_contradiction"] else "non_contradiction",
        ))
    return pairs


_LOADERS = {"longmemeval": _load_longmemeval, "ami": _load_ami, "qmsum": _load_qmsum, "decode": _load_decode}
_GOLD = {"longmemeval": _gold_longmemeval, "qmsum": _gold_qmsum}


def load_dataset(name: str, root: Path) -> list[Conversation]:
    if name not in _LOADERS:
        raise ValueError(f"unknown dataset {name!r}; expected one of {sorted(_LOADERS)}")
    return _LOADERS[name](root)


def load_gold(name: str, root: Path) -> list[Scenario]:
    if name not in _GOLD:
        raise ValueError(f"no gold scenarios for {name!r}; expected one of {sorted(_GOLD)}")
    return _GOLD[name](root)
