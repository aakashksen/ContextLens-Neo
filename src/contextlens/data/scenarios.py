import json
import random
from pathlib import Path

from contextlens.data import scenario_cases as cases
from contextlens.data.loaders import load_gold
from contextlens.schemas import ButtonRequest, Conversation, ConsistencyLabel, Scenario, Turn

SCENARIOS_DIR = Path("datasets/scenarios")
SPLITS = ("dev", "test")
DATASET_RECALL_PER_SPLIT = 10


def _conversation(cid: str, lines: tuple[str, ...] | list[str], split: str) -> Conversation:
    turns = []
    for i, line in enumerate(lines):
        speaker, text = line.split(": ", 1)
        turns.append(Turn(id=f"{cid}:t{i}", conversation_id=cid, seq=i, speaker=speaker, text=text,
                          start_ms=i * 4000, end_ms=i * 4000 + 3500, source="synthetic"))
    return Conversation(id=cid, turns=turns, source="synthetic", split=split)  # type: ignore[arg-type]


def _scenario(key: str, i: int, split: str, category: str, lines, evidence: list[int], *, label: ConsistencyLabel | None,
              question: str | None = None, answer: str | None = None) -> Scenario:
    cid = f"syn-{key}-{i:02d}"
    conv = _conversation(cid, lines, split)
    last = len(conv.turns) - 1
    return Scenario(
        id=cid, conversation=conv, category=category,
        request=ButtonRequest(conversation_id=cid, as_of_seq=last, question=question),
        gold_evidence_turn_ids=[conv.turns[j].id for j in sorted({e % len(conv.turns) for e in evidence})],
        gold_label=label, gold_answer=answer, origin="synthetic",
    )


def _split_of(i: int) -> str:
    return SPLITS[i % 2]


def _written() -> dict[str, list[Scenario]]:
    out: dict[str, list[Scenario]] = {s: [] for s in SPLITS}
    for sc in [*_labelled_cases(), *_question_cases(), *_deadline_cases()]:
        out[sc.conversation.split].append(sc)  # type: ignore[index]
    return out


def _labelled_cases() -> list[Scenario]:
    out = []
    for i, t in enumerate(cases.CONTRADICTION):
        out.append(_scenario("con", i, _split_of(i), "contradiction", t, [0, -1], label="contradiction"))
    for i, t in enumerate(cases.UPDATE):
        out.append(_scenario("upd", i, _split_of(i), "legitimate_update", t, [0, -1], label="update"))
    for i, t in enumerate(cases.HYPOTHETICAL):
        out.append(_scenario("hyp", i, _split_of(i), "hypothetical_or_joking", t, [0, -1], label="consistent"))
    for i, t in enumerate(cases.WRONG_SPEAKER_CONSISTENT):
        out.append(_scenario("wsc", i, _split_of(i), "wrong_speaker_trap", t, [0, -1], label="consistent"))
    return out


def _question_cases() -> list[Scenario]:
    out = []
    for i, (t, q, a, ev) in enumerate(cases.WRONG_SPEAKER_QUESTION):
        out.append(_scenario("wsq", i, _split_of(i), "wrong_speaker_trap", t, [ev], label=None, question=q, answer=a))
    for i, (t, q, a, ev) in enumerate(cases.INJECTION):
        out.append(_scenario("inj", i, _split_of(i), "prompt_injection", t, [ev], label=None, question=q, answer=a))
    for i, (t, q) in enumerate(cases.INSUFFICIENT):
        out.append(_scenario("ins", i, _split_of(i), "insufficient_evidence", t, [], label="insufficient", question=q))
    for i, (t, a, ev) in enumerate(cases.RESPOND):
        out.append(_scenario("rsp", i, _split_of(i), "respond_with_suggestion", t, list(ev), label=None, answer=a))
    for i, (t, q, a, ev) in enumerate(cases.RECALL):
        out.append(_scenario("rec", i, _split_of(i), "recall", t, list(ev), label=None, question=q, answer=a))
    return out


def _deadline_cases() -> list[Scenario]:
    return [
        _scenario("deadline-update", 0, "dev", "legitimate_update",
                  ["Ravi: The deadline is Monday", "Sue: We moved the deadline to Wednesday"], [0, 1], label="update"),
        _scenario("deadline-contradiction", 0, "dev", "contradiction",
                  ["Ravi: The deadline is Monday", "Sue: We moved the deadline to Wednesday",
                   "Sue: It has always been Wednesday"], [0, 1, 2], label="contradiction"),
    ]


def _dataset_recall(raw_root: Path, splits_dir: Path) -> dict[str, list[Scenario]]:
    manifest = json.loads((splits_dir / "longmemeval.json").read_text(encoding="utf-8"))["splits"]
    gold = [s for s in load_gold("longmemeval", raw_root) if s.gold_evidence_turn_ids]
    out: dict[str, list[Scenario]] = {}
    for split in SPLITS:
        members = set(manifest[split])
        pool = sorted((s for s in gold if s.conversation.id in members),
                      key=lambda s: (sum(len(t.text) for t in s.conversation.turns), s.id))
        picked = []
        for s in pool[:DATASET_RECALL_PER_SPLIT]:
            conv = s.conversation.model_copy(update={"split": split})
            picked.append(s.model_copy(update={"conversation": conv, "category": "recall"}))
        out[split] = picked
    return out


def build_scenarios(raw_root: Path, splits_dir: Path, out_dir: Path = SCENARIOS_DIR) -> dict[str, int]:
    written, dataset = _written(), _dataset_recall(raw_root, splits_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in SPLITS:
        scs = written[split] + dataset[split]
        (out_dir / f"{split}.jsonl").write_text(
            "".join(s.model_dump_json() + "\n" for s in scs), encoding="utf-8")
        counts[split] = len(scs)
    return counts


def load_scenarios(split: str, root: Path = SCENARIOS_DIR) -> list[Scenario]:
    path = root / f"{split}.jsonl"
    return [Scenario.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_review_sample(root: Path = SCENARIOS_DIR, n: int = 50, seed: int = 13) -> Path:
    pool = [s for split in SPLITS for s in load_scenarios(split, root)]
    sample = random.Random(seed).sample(sorted(pool, key=lambda s: s.id), n)
    lines = [f"# Scenario review sample ({n} random scenarios, seed {seed})", "",
             "Check that each gold label and answer is defensible. Mark any that are not.", ""]
    for k, s in enumerate(sample, 1):
        lines += [f"## {k}. {s.id}", f"- category: {s.category}; origin: {s.origin}; split: {s.conversation.split}"]
        turns, gold = s.conversation.turns, set(s.gold_evidence_turn_ids)
        shown = [t for i, t in enumerate(turns) if i >= len(turns) - 6 or t.id in gold]
        lines += [f"- {t.id} [{t.speaker}]: {t.text if t.id in gold else t.text[:300]}" for t in shown]
        if len(shown) < len(turns):
            lines.append(f"- (showing {len(shown)} of {len(turns)} turns: last 6 and all gold evidence)")
        lines += [f"- question: {s.request.question}", f"- gold_label: {s.gold_label}",
                  f"- gold_answer: {s.gold_answer}", f"- gold_evidence: {s.gold_evidence_turn_ids}", ""]
    path = root / "review-sample.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(build_scenarios(Path("datasets/raw"), Path("datasets/splits")))
    print(write_review_sample())
