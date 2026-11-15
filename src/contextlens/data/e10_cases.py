"""E10's case set: long meetings, long-range evidence, conflicts far apart, answer-present/answer-removed twins, short controls.

Four case types, all pressed after a long stretch of real meeting text (AMI, QMSum), split dev/test by meeting:

  qmsum    A QMSum specific question whose gold span ends at least 10 turns before the press at the end of the meeting.
           Everything (transcript, question, gold span, reference answer) is the publisher's; nothing is invented.
  conflict Triplets on one real stretch of meeting: a statement A, 20 to 60 real turns, then a statement B that
           contradicts A / legitimately updates A / agrees with A. A and B are invented (templates below); the turns
           between are real. Gold evidence is the two turns. No question is pressed.
  twin     Pairs on one real stretch of 40 to 100 turns, a question pressed at the end. In the answerable twin one
           invented turn near the start states the answer; in the insufficient twin that turn is replaced by a turn on
           the same topic that states nothing (a near miss), so the answer is removed and gold evidence is empty.
  control  A seeded sample of short E8-style cases from datasets/scenarios/dev.jsonl, unchanged.

The scenario files hold real meeting text, so they sit in git-ignored datasets/raw/scenarios/e10; the committed
manifest datasets/splits/e10-scenarios.json holds ids, hashes and how each case was made, never text.
Rebuilding with the same seed gives the same files byte for byte (the manifest hash proves it).
"""

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from contextlens.data.loaders import _turn, load_dataset, load_gold
from contextlens.eval.records import ROOT, write_atomic
from contextlens.schemas import ButtonRequest, Conversation, Scenario, Turn

SEED = 10
MANIFEST_REL = Path("datasets/splits/e10-scenarios.json")
DATA_REL = Path("datasets/raw/scenarios/e10")
RAW_REL = Path("datasets/raw")
CONTROL_SOURCE_REL = Path("datasets/scenarios/dev.jsonl")
BUILDER_VERSION = "e10-cases-v1"

QUOTA = {"qmsum": 100, "conflict_triplets": 30, "twin_pairs": 50, "control": 40}
QMSUM_MIN_BACK = 10  # turns between the last gold turn and the press
QMSUM_MAX_GOLD = 8
QMSUM_MAX_TURNS = 900
QMSUM_PER_MEETING = 2
CONFLICT_GAP = (20, 60)  # real turns between statement A and statement B
CONFLICT_LEAD = 5  # real turns before statement A
TWIN_WINDOW = (40, 100)
TWIN_FACT_AT = (3, 12)  # position of the answer turn from the start of the window
MIN_WORDS = 3  # a distractor turn shorter than this is a backchannel; the window must still hold mostly real talk


def side(meeting: str) -> str:
    """dev or test, by the meeting only, so a meeting is never on both sides."""
    return "dev" if int(hashlib.sha256(f"e10:{meeting}".encode()).hexdigest()[:8], 16) % 2 == 0 else "test"


# ------------------------------------------------------------------ invented statements (conflict and twins)

_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
_NAMES = ["Priya", "Tomas", "Wen", "Ifeoma", "Lars", "Marguerite", "Dmitri", "Aiko", "Beatrix", "Santiago"]
_ITEMS = ["crates", "boxes", "pallets", "cartons"]
_PLACES = ["Rotterdam", "Gdansk", "Porto", "Leeds", "Bologna", "Tampere", "Graz", "Cork"]

# Each: topic, A, contradiction B (flat denial, no sign of a change), update B (the same new value, with a stated change and
# reason), consistent B (agrees with A). {v1} is A's value, {v2} the conflicting one.
CONFLICTS: list[dict[str, Any]] = [
    {"values": lambda r: (f"{(a := r.choice([12, 18, 24, 30]))}", f"{a + r.choice([6, 10, 14])}"),
     "A": "The venue deposit for the offsite is {v1} hundred euros.",
     "contra": "The venue deposit for the offsite is {v2} hundred euros, that is the figure on the contract.",
     "upd": "The venue deposit for the offsite went up to {v2} hundred euros, they revised the quote after we added the catering.",
     "cons": "Right, the venue deposit for the offsite is {v1} hundred euros, that matches the contract."},
    {"values": lambda r: tuple(r.sample(_DAYS, 2)),
     "A": "The supplier delivered the {item} on {v1}.",
     "contra": "The supplier delivered the {item} on {v2}, I signed for them myself.",
     "upd": "The delivery of the {item} has since moved to {v2}, the supplier rescheduled it because of a strike.",
     "cons": "Yes, the supplier delivered the {item} on {v1}, I signed for them."},
    {"values": lambda r: (f"{(a := r.choice([4, 5, 6, 7]))}", f"{a + r.choice([2, 3])}"),
     "A": "The audit team has {v1} people on it.",
     "contra": "The audit team has {v2} people on it, I counted them this morning.",
     "upd": "The audit team grew to {v2} people, two more joined this week after the extension.",
     "cons": "Correct, the audit team has {v1} people on it."},
    {"values": lambda r: tuple(r.sample(_NAMES, 2)),
     "A": "{v1} owns the migration checklist.",
     "contra": "{v2} owns the migration checklist, it was assigned to them at the start.",
     "upd": "{v2} owns the migration checklist now, {v1} handed it over because of the reorganisation.",
     "cons": "As agreed, {v1} owns the migration checklist."},
    {"values": lambda r: (f"{(a := r.choice([2, 3, 4]))}.{r.choice([0, 1, 5])}", f"{(a + r.choice([1, 2]))}.{r.choice([0, 2, 4])}"),
     "A": "The shared library is on version {v1}.",
     "contra": "The shared library is on version {v2}, that is what the lockfile says.",
     "upd": "The shared library is on version {v2} after the upgrade we did yesterday.",
     "cons": "Yes, the shared library is on version {v1}, the lockfile agrees."},
    {"values": lambda r: tuple(r.sample(_PLACES, 2)),
     "A": "The regional warehouse is in {v1}.",
     "contra": "The regional warehouse is in {v2}, I visited it last spring.",
     "upd": "The regional warehouse moved to {v2} last month because the lease in the old town ended.",
     "cons": "That is right, the regional warehouse is in {v1}."},
    {"values": lambda r: (f"{(a := r.choice([8, 12, 15]))}", f"{a + r.choice([5, 7, 9])}"),
     "A": "The licence costs {v1} euros per seat.",
     "contra": "The licence costs {v2} euros per seat, I have the invoice here.",
     "upd": "The licence now costs {v2} euros per seat, the vendor raised the price from the first of the month.",
     "cons": "Yes, the licence costs {v1} euros per seat, per the invoice."},
    {"values": lambda r: (f"{(a := r.choice([14, 21, 28]))}", f"{a + r.choice([7, 14])}"),
     "A": "The trial period lasts {v1} days.",
     "contra": "The trial period lasts {v2} days, it says so in the terms.",
     "upd": "The trial period was extended to {v2} days, the vendor offered it after we asked.",
     "cons": "Right, the trial period lasts {v1} days, as the terms say."},
    {"values": lambda r: (f"{(a := r.choice([3, 5, 7]))}", f"{a + r.choice([2, 3])}"),
     "A": "Room {v1}0 is booked for the workshop.",
     "contra": "Room {v2}0 is booked for the workshop, I have the confirmation email.",
     "upd": "The workshop moved to room {v2}0, because room {v1}0 turned out to be too small.",
     "cons": "Yes, room {v1}0 is booked for the workshop, I saw the confirmation."},
    {"values": lambda r: (f"{(a := r.choice([20, 25, 40]))}", f"{a + r.choice([10, 15])}"),
     "A": "The pilot group is capped at {v1} testers.",
     "contra": "The pilot group is capped at {v2} testers, that is what legal signed off.",
     "upd": "The pilot group cap went up to {v2} testers, legal approved a larger group this week.",
     "cons": "Correct, the pilot group is capped at {v1} testers, as legal signed off."},
]

# Twins: the answer turn, a near miss on the same topic that states nothing, the question, and the reference answer.
TWINS: list[dict[str, Any]] = [
    {"values": lambda r: (f"{r.choice([4, 6, 8, 12])}{r.choice(['00', '50'])}",),
     "fact": "The training budget for the new hires is {v} dollars.",
     "near": "We still have to settle the training budget for the new hires.",
     "q": "What is the training budget for the new hires?", "a": "{v} dollars"},
    {"values": lambda r: (r.choice(_DAYS),),
     "fact": "The vendor demo is booked for {v}.",
     "near": "Somebody should ask when the vendor demo can happen.",
     "q": "When is the vendor demo?", "a": "{v}"},
    {"values": lambda r: (r.choice(_NAMES),),
     "fact": "{v} will run the customer interviews.",
     "near": "We need someone to run the customer interviews.",
     "q": "Who will run the customer interviews?", "a": "{v}"},
    {"values": lambda r: (r.choice(_PLACES),),
     "fact": "The next planning week will be held in {v}.",
     "near": "We have not picked a city for the next planning week.",
     "q": "Where will the next planning week be held?", "a": "{v}"},
    {"values": lambda r: (f"{r.choice([40, 60, 75, 90])}",),
     "fact": "The print run for the brochure is {v} copies.",
     "near": "The size of the print run for the brochure is still open.",
     "q": "How many copies is the brochure print run?", "a": "{v} copies"},
    {"values": lambda r: (r.choice(["Friday", "Thursday", "Wednesday"]),),
     "fact": "The security review has to be finished by {v}.",
     "near": "The security review needs a deadline, nobody has set one yet.",
     "q": "By when does the security review have to be finished?", "a": "{v}"},
    {"values": lambda r: (r.choice(["Aurora", "Basalt", "Cobalt", "Meridian", "Tundra"]),),
     "fact": "The codename for the second release is {v}.",
     "near": "We have not agreed a codename for the second release.",
     "q": "What is the codename for the second release?", "a": "{v}"},
    {"values": lambda r: (f"{r.choice([2, 3, 5, 6])} weeks",),
     "fact": "The onboarding programme will take {v}.",
     "near": "How long the onboarding programme should take has not been decided.",
     "q": "How long will the onboarding programme take?", "a": "{v}"},
    {"values": lambda r: (r.choice(["the east entrance", "the loading dock", "the main lobby", "the car park"]),),
     "fact": "The visitors will be met at {v}.",
     "near": "Nobody has said where the visitors will be met.",
     "q": "Where will the visitors be met?", "a": "{v}"},
    {"values": lambda r: (f"{r.choice([3, 4, 5, 7])} percent",),
     "fact": "The contingency reserve is set at {v} of the budget.",
     "near": "The size of the contingency reserve is not set yet.",
     "q": "How big is the contingency reserve?", "a": "{v} of the budget"},
]


# ------------------------------------------------------------------ real text

def _speakers(turns: list[Turn]) -> list[str]:
    return sorted({t.speaker for t in turns})


def _eligible_window(turns: list[Turn], start: int, length: int) -> bool:
    w = turns[start:start + length]
    return len(w) == length and len(_speakers(w)) >= 3 and sum(len(t.text.split()) >= MIN_WORDS for t in w) >= 0.6 * length


def _pick_window(rng: random.Random, turns: list[Turn], length: int, taken: list[tuple[int, int]]) -> int | None:
    """A start index whose window of `length` turns has 3+ speakers and mostly real talk, not overlapping an earlier window."""
    for _ in range(60):
        start = rng.randrange(0, max(1, len(turns) - length))
        if any(start < b and a < start + length for a, b in taken):
            continue
        if _eligible_window(turns, start, length):
            return start
    return None


def _rebuild(cid: str, turns: list[Turn]) -> list[Turn]:
    """New conversation: the same words in order, renumbered, ids under the new conversation id."""
    return [_turn(cid, i, t.speaker, t.text).model_copy(update={"source": t.source}) for i, t in enumerate(turns)]


def _scenario(sid: str, turns: list[Turn], category: str, *, question: str | None, gold_ids: list[int], gold_label: Any,
              gold_answer: str | None, split: str, source: str) -> Scenario:
    conv = Conversation(id=sid, turns=turns, source=source, split=split)  # type: ignore[arg-type]
    return Scenario(
        id=sid, conversation=conv, category=category, gold_label=gold_label, gold_answer=gold_answer, origin="synthetic",
        request=ButtonRequest(conversation_id=sid, as_of_seq=len(turns) - 1, question=question),
        gold_evidence_turn_ids=[f"{sid}:t{i}" for i in gold_ids])


def _inject(cid: str, speaker: str, text: str, seq: int) -> Turn:
    t = _turn(cid, seq, speaker, text)
    return t.model_copy(update={"source": "synthetic"})


# ------------------------------------------------------------------ the four types

def build_qmsum(raw_root: Path) -> tuple[list[Scenario], dict[str, dict[str, Any]]]:
    rng = random.Random(f"{SEED}:qmsum")
    pool = []
    for s in load_gold("qmsum", raw_root):
        if s.category != "qmsum-specific" or not s.gold_evidence_turn_ids or not s.request.question:
            continue
        n = len(s.conversation.turns)
        idx = [int(t.rsplit(":t", 1)[1]) for t in s.gold_evidence_turn_ids]
        if max(idx) <= n - 1 - QMSUM_MIN_BACK and len(idx) <= QMSUM_MAX_GOLD and n <= QMSUM_MAX_TURNS and s.gold_answer:
            pool.append((s, max(idx)))
    out: list[Scenario] = []
    meta: dict[str, dict[str, Any]] = {}
    for sp in ("dev", "test"):
        mine = sorted((p for p in pool if side(p[0].conversation.id) == sp), key=lambda p: p[0].id)
        rng.shuffle(mine)
        per: Counter[str] = Counter()
        made = 0
        for s, last in mine:
            if made >= QUOTA["qmsum"] // 2:
                break
            if per[s.conversation.id] >= QMSUM_PER_MEETING:
                continue
            per[s.conversation.id] += 1
            made += 1
            sid = f"e10-qm-{len(out):03d}"
            old = s.conversation.id
            turns = [t.model_copy(update={"id": f"{sid}:t{t.seq}", "conversation_id": sid}) for t in s.conversation.turns]
            gold = [g.replace(old, sid, 1) for g in s.gold_evidence_turn_ids]
            conv = Conversation(id=sid, turns=turns, source="qmsum", split=sp)  # type: ignore[arg-type]
            out.append(Scenario(
                id=sid, conversation=conv, category="qmsum_long", origin="dataset", gold_label=None, gold_answer=s.gold_answer,
                request=ButtonRequest(conversation_id=sid, as_of_seq=len(turns) - 1, question=s.request.question),
                gold_evidence_turn_ids=gold))
            meta[sid] = {"type": "qmsum", "split": sp, "meeting": old, "source_case": s.id, "turns": len(turns),
                         "gold_turns": len(gold), "turns_back": len(turns) - 1 - last,
                         "how": "QMSum specific question and published gold span; pressed at the end of the meeting"}
    return out, meta


def _meetings(raw_root: Path) -> dict[str, list[Turn]]:
    out: dict[str, list[Turn]] = {}
    for name in ("ami", "qmsum"):
        for c in load_dataset(name, raw_root):
            if len(c.turns) >= 200 and len(_speakers(c.turns)) >= 3 and (name == "ami" or len(c.turns) <= QMSUM_MAX_TURNS):
                out[f"{name}:{c.id}"] = sorted(c.turns, key=lambda t: t.seq)
    return out


def _fill(tpl: str, v: tuple[str, ...], item: str = "") -> str:
    d = {"v": v[0], "item": item}
    if len(v) > 1:
        d.update(v1=v[0], v2=v[1])
    else:
        d.update(v1=v[0])
    return tpl.format(**d)


def build_conflicts(meetings: dict[str, list[Turn]]) -> tuple[list[Scenario], dict[str, dict[str, Any]]]:
    rng = random.Random(f"{SEED}:conflict")
    out: list[Scenario] = []
    meta: dict[str, dict[str, Any]] = {}
    names = sorted(meetings)
    taken: dict[str, list[tuple[int, int]]] = {}
    for sp in ("dev", "test"):
        pool = [m for m in names if side(m) == sp]
        rng.shuffle(pool)
        made = 0
        for k in range(QUOTA["conflict_triplets"] // 2):
            for meeting in pool[k * 6:] + pool[:k * 6]:  # the next meeting, else any that still has a window
                gap = rng.randint(*CONFLICT_GAP)
                turns = meetings[meeting]
                start = _pick_window(rng, turns, CONFLICT_LEAD + gap, taken.setdefault(meeting, []))
                if start is not None:
                    break
            else:
                raise RuntimeError("no window for a conflict triplet")
            taken[meeting].append((start, start + CONFLICT_LEAD + gap))
            tpl = CONFLICTS[made % len(CONFLICTS)]
            made += 1
            v = tpl["values"](rng)
            item = rng.choice(_ITEMS)
            lead, middle = turns[start:start + CONFLICT_LEAD], turns[start + CONFLICT_LEAD:start + CONFLICT_LEAD + gap]
            spk = _speakers(middle)
            a_spk = rng.choice(spk)
            other_speaker = rng.random() < 1 / 3
            b_spk = rng.choice([s for s in spk if s != a_spk]) if other_speaker else a_spk
            n = len(out) // 3
            a_text = _fill(tpl["A"], v, item)
            for kind, label in (("contra", "contradiction"), ("upd", "update"), ("cons", "consistent")):
                sid = f"e10-cf-{n:02d}-{label}"
                texts = _rebuild(sid, [*lead, _inject(sid, a_spk, a_text, 0), *middle,
                                       _inject(sid, b_spk, _fill(tpl[kind], v, item), 0)])
                a_seq, b_seq = len(lead), len(texts) - 1
                sc = _scenario(sid, texts, f"conflict_{label}", question=None, gold_ids=[a_seq, b_seq], gold_label=label,
                               gold_answer=None, split=sp, source=meeting.split(":")[0])
                out.append(sc)
                meta[sid] = {"type": "conflict", "split": sp, "meeting": meeting, "triplet": n, "label": label, "template": CONFLICTS.index(tpl),
                             "window": [start, start + CONFLICT_LEAD + gap], "distractor_turns": gap, "speakers_in_window": len(spk),
                             "different_speakers": b_spk != a_spk, "gold_seq": [a_seq, b_seq],
                             "how": "real meeting turns with two invented statements inserted; gold = both"}
    return out, meta


def build_twins(meetings: dict[str, list[Turn]]) -> tuple[list[Scenario], dict[str, dict[str, Any]]]:
    rng = random.Random(f"{SEED}:twin")
    out: list[Scenario] = []
    meta: dict[str, dict[str, Any]] = {}
    taken: dict[str, list[tuple[int, int]]] = {}
    pairs = 0
    names = sorted(meetings)
    for sp in ("dev", "test"):
        pool = [m for m in names if side(m) == sp]
        rng.shuffle(pool)
        for k in range(QUOTA["twin_pairs"] // 2):
            for meeting in pool[k * 3:] + pool[:k * 3]:
                length = rng.randint(*TWIN_WINDOW)
                turns = meetings[meeting]
                start = _pick_window(rng, turns, length, taken.setdefault(meeting, []))
                if start is not None:
                    break
            else:
                raise RuntimeError("no window for a twin pair")
            taken[meeting].append((start, start + length))
            tpl = TWINS[pairs % len(TWINS)]
            v = tpl["values"](rng)
            window = turns[start:start + length]
            spk = rng.choice(_speakers(window))
            at = rng.randint(*TWIN_FACT_AT)
            for label, text in (("answerable", _fill(tpl["fact"], v)), ("insufficient", _fill(tpl["near"], v))):
                sid = f"e10-tw-{pairs:02d}-{label}"
                texts = _rebuild(sid, [*window[:at], _inject(sid, spk, text, 0), *window[at:]])
                answerable = label == "answerable"
                out.append(_scenario(
                    sid, texts, f"twin_{label}", question=tpl["q"], gold_ids=[at] if answerable else [],
                    gold_label=None if answerable else "insufficient", gold_answer=_fill(tpl["a"], v) if answerable else None,
                    split=sp, source=meeting.split(":")[0]))
                meta[sid] = {"type": "twin", "split": sp, "meeting": meeting, "pair": pairs, "label": label, "template": TWINS.index(tpl),
                             "window": [start, start + length], "turns": len(texts), "answer_turn": at,
                             "how": "real meeting turns with one invented turn: the answer, or a near miss that states nothing"}
            pairs += 1
    return out, meta


def build_controls(root: Path) -> tuple[list[Scenario], dict[str, dict[str, Any]]]:
    rng = random.Random(f"{SEED}:control")
    rows = [Scenario.model_validate_json(ln) for ln in (root / CONTROL_SOURCE_REL).read_text(encoding="utf-8").splitlines() if ln.strip()]
    by_cat: dict[str, list[Scenario]] = {}
    for s in sorted(rows, key=lambda s: s.id):
        by_cat.setdefault(s.category, []).append(s)
    cats = sorted(by_cat)
    quota = {c: QUOTA["control"] // len(cats) + (i < QUOTA["control"] % len(cats)) for i, c in enumerate(cats)}
    out, meta = [], {}
    for c in cats:
        pick = rng.sample(by_cat[c], quota[c])
        for i, s in enumerate(pick):
            sp = "dev" if i % 2 == 0 else "test"
            conv = s.conversation.model_copy(update={"split": sp})
            sid = f"e10-ct-{s.id}"
            out.append(s.model_copy(update={"id": sid, "conversation": conv, "category": f"control_{c}"}))
            meta[sid] = {"type": "control", "split": sp, "meeting": s.conversation.id, "source_case": s.id,
                         "how": "seeded sample of E8's dev short cases, unchanged"}
    return out, meta


# ------------------------------------------------------------------ build, freeze, load

def build_all(root: Path = ROOT, raw_root: Path | None = None) -> tuple[dict[str, list[Scenario]], dict[str, dict[str, Any]]]:
    raw_root = raw_root or root / RAW_REL
    meetings = _meetings(raw_root)
    parts = [build_qmsum(raw_root), build_conflicts(meetings), build_twins(meetings), build_controls(root)]
    meta: dict[str, dict[str, Any]] = {}
    splits: dict[str, list[Scenario]] = {"dev": [], "test": []}
    for scs, m in parts:
        meta.update(m)
        for s in scs:
            splits[m[s.id]["split"]].append(s)
    shared = {m["meeting"] for m in meta.values() if m["split"] == "dev"} & {m["meeting"] for m in meta.values() if m["split"] == "test"}
    if shared:
        raise RuntimeError(f"meetings on both sides: {sorted(shared)}")
    for sp in splits:
        splits[sp].sort(key=lambda s: s.id)
    return splits, meta


def counts(splits: dict[str, list[Scenario]], meta: dict[str, dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for sp, scs in splits.items():
        c = Counter(s.category for s in scs)
        out[sp] = {**dict(sorted(c.items())), "total": len(scs)}
    return out


def freeze(root: Path = ROOT, raw_root: Path | None = None, out_root: Path | None = None) -> dict[str, Any]:
    """Build the cases, write them (git-ignored) and the manifest with their hashes (committed). Refuses to overwrite a
    manifest whose hashes differ, so a frozen set is never silently replaced."""
    out_root = out_root or root
    splits, meta = build_all(root, raw_root)
    directory = out_root / DATA_REL
    directory.mkdir(parents=True, exist_ok=True)
    entries = {}
    for sp, scs in splits.items():
        text = "".join(s.model_dump_json() + "\n" for s in scs)
        write_atomic(directory / f"{sp}.jsonl", text)
        entries[sp] = {"hash": hashlib.sha256(text.encode("utf-8")).hexdigest(), "n": len(scs), "ids": [s.id for s in scs]}
    manifest = {"kind": "scenarios", "source": "e10", "builder": BUILDER_VERSION, "seed": SEED, "quota": QUOTA,
                "counts": counts(splits, meta), "splits": entries, "cases": meta,
                "note": "Real AMI and QMSum text sits in git-ignored datasets/raw/scenarios/e10; this file holds ids, hashes and how each "
                        "case was made. Conflict and twin statements are invented from templates, so their labels are unreviewed."}
    path = out_root / MANIFEST_REL
    if path.is_file():
        old = json.loads(path.read_text(encoding="utf-8"))
        if {k: v["hash"] for k, v in old["splits"].items()} != {k: v["hash"] for k, v in entries.items()}:
            raise RuntimeError("the frozen E10 cases would change; delete the manifest only if the set was never run")
    write_atomic(path, json.dumps(manifest, indent=1) + "\n")
    return manifest


def manifest_sha(root: Path) -> str:
    """Hash of the manifest's content (parsed, so line endings do not matter): the cases' hashes, ids and notes."""
    m = json.loads((root / MANIFEST_REL).read_text(encoding="utf-8"))
    return hashlib.sha256(json.dumps(m, sort_keys=True).encode("utf-8")).hexdigest()


def data_dir(root: Path) -> Path:
    """Where the case files are: this checkout's, else the main folder's (a linked worktree has no datasets/raw)."""
    from contextlens.llm.client import shared_root

    here = root / DATA_REL
    return here if (here / "dev.jsonl").is_file() else shared_root(root) / DATA_REL


def load_cases(split: str, root: Path = ROOT) -> tuple[list[Scenario], dict[str, dict[str, Any]], str]:
    """The split's scenarios, the manifest's per-case notes, and the hash; fails if the files do not match the manifest."""
    manifest = json.loads((root / MANIFEST_REL).read_text(encoding="utf-8"))
    raw = (data_dir(root) / f"{split}.jsonl").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != manifest["splits"][split]["hash"]:
        raise RuntimeError(f"the E10 {split} file does not match its manifest")
    scs = [Scenario.model_validate_json(ln) for ln in raw.decode("utf-8").splitlines() if ln.strip()]
    return scs, manifest["cases"], digest


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    raw = Path(args[0]) if args else None
    m = freeze(ROOT, raw)
    print(json.dumps(m["counts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
