"""The train-only scenario set at datasets/scenarios_train/: invented, hand-written, origin "synthetic".
It covers all four consistency labels and the three card modes (recall, clarify, respond), abstention and injected
turns. It reuses no dev or test text; `overlap_problems` checks that and the experiment runner enforces it."""
import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from contextlens.data import train_cases as cases
from contextlens.data.scenarios import SCENARIOS_DIR, _conversation, load_scenarios
from contextlens.schemas import ButtonRequest, ConsistencyLabel, Scenario

TRAIN_DIR = Path("datasets/scenarios_train")
RUN = 8  # a run of this many words in a row counts as shared text

# The card mode a case is for, by category. A consistent exchange has nothing to flag, so its mode is the
# clarify check that finds no conflict; abstention is a recall question the turns cannot answer.
CARD_MODE = {
    "contradiction": "clarify", "legitimate_update": "clarify", "hypothetical_or_joking": "clarify",
    "wrong_speaker_trap": "recall", "injected_consistent": "clarify", "prompt_injection": "recall",
    "insufficient_evidence": "recall", "recall": "recall", "respond_with_suggestion": "respond",
}


def _scenario(key: str, i: int, category: str, lines, evidence: list[int], *, label: ConsistencyLabel | None,
              question: str | None = None, answer: str | None = None) -> Scenario:
    cid = f"trn-{key}-{i:02d}"
    conv = _conversation(cid, lines, "train")
    last = len(conv.turns) - 1
    return Scenario(
        id=cid, conversation=conv, category=category,
        request=ButtonRequest(conversation_id=cid, as_of_seq=last, question=question),
        gold_evidence_turn_ids=[conv.turns[j].id for j in sorted({e % len(conv.turns) for e in evidence})],
        gold_label=label, gold_answer=answer, origin="synthetic",
    )


def build_train_scenarios() -> list[Scenario]:
    return [*_labelled_train(), *_question_train()]


def _labelled_train() -> list[Scenario]:
    out: list[Scenario] = []
    for i, t in enumerate(cases.CONTRADICTION):
        out.append(_scenario("con", i, "contradiction", t, [0, -1], label="contradiction"))
    for i, t in enumerate(cases.UPDATE):
        out.append(_scenario("upd", i, "legitimate_update", t, [0, -1], label="update"))
    for i, t in enumerate(cases.HYPOTHETICAL):
        out.append(_scenario("hyp", i, "hypothetical_or_joking", t, [0, -1], label="consistent"))
    for i, t in enumerate(cases.WRONG_SPEAKER_CONSISTENT):
        out.append(_scenario("wsc", i, "wrong_speaker_trap", t, [0, -1], label="consistent"))
    for i, t in enumerate(cases.INJECTED_CONSISTENT):
        out.append(_scenario("ijc", i, "injected_consistent", t, [0, -1], label="consistent"))
    for i, (t, q) in enumerate(cases.INSUFFICIENT):
        out.append(_scenario("ins", i, "insufficient_evidence", t, [], label="insufficient", question=q))
    return out


def _question_train() -> list[Scenario]:
    out: list[Scenario] = []
    for i, (t, q, a, ev) in enumerate(cases.RECALL):
        out.append(_scenario("rec", i, "recall", t, list(ev), label=None, question=q, answer=a))
    for i, (t, q, a, ev) in enumerate(cases.WRONG_SPEAKER_QUESTION):
        out.append(_scenario("wsq", i, "wrong_speaker_trap", t, [ev], label=None, question=q, answer=a))
    for i, (t, q, a, ev) in enumerate(cases.INJECTION):
        out.append(_scenario("inj", i, "prompt_injection", t, [ev], label=None, question=q, answer=a))
    for i, (t, a, ev) in enumerate(cases.RESPOND):
        out.append(_scenario("rsp", i, "respond_with_suggestion", t, list(ev), label=None, answer=a))
    return out


def mode_of(s: Scenario) -> str:
    return CARD_MODE[s.category]


def counts(scenarios: list[Scenario]) -> dict[str, dict[str, int]]:
    """Distinct cases per label, per card mode and per category. Every case is its own conversation."""
    return {
        "cases": {"total": len(scenarios), "conversations": len({s.conversation.id for s in scenarios})},
        "by_label": dict(sorted(Counter(s.gold_label or "none" for s in scenarios).items())),
        "by_mode": dict(sorted(Counter(mode_of(s) for s in scenarios).items())),
        "by_category": dict(sorted(Counter(s.category for s in scenarios).items())),
        "by_mode_and_label": dict(sorted(Counter(f"{mode_of(s)}/{s.gold_label or 'none'}" for s in scenarios).items())),
        "abstention_cases": {"insufficient": sum(s.gold_label == "insufficient" for s in scenarios)},
        "injected_turn_cases": {"with_injection": sum(s.category in ("prompt_injection", "injected_consistent")
                                                      for s in scenarios)},
    }


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def word_runs(texts: Iterable[str], n: int = RUN) -> set[str]:
    out: set[str] = set()
    for t in texts:
        w = _words(t)
        out |= {" ".join(w[i : i + n]) for i in range(len(w) - n + 1)}
    return out


def _texts(scenarios: list[Scenario]) -> list[str]:
    return [t.text for s in scenarios for t in s.conversation.turns] + [
        x for s in scenarios for x in (s.request.question, s.gold_answer) if x]


def overlap_problems(train: list[Scenario], others: list[Scenario]) -> list[str]:
    """Shared scenario ids, conversation ids, turn texts, or runs of RUN words between the training set and others."""
    out = []
    ids = {s.id for s in others} | {s.conversation.id for s in others}
    out += [f"id {s.id} is also in the held-out scenarios" for s in train
            if s.id in ids or s.conversation.id in ids]
    # stock replies such as "Good." or "Noted." are allowed to recur; any turn of five words or more is not
    texts = {t.text.strip().lower() for s in others for t in s.conversation.turns if len(_words(t.text)) >= 5}
    out += [f"{s.id} turn {t.id} repeats a held-out turn" for s in train for t in s.conversation.turns
            if t.text.strip().lower() in texts]
    shared = word_runs(_texts(train)) & word_runs(_texts(others))
    if shared:
        out.append(f"{len(shared)} shared {RUN}-word runs, e.g. {sorted(shared)[0]!r}")
    return out


def held_out(scenarios_root: Path = SCENARIOS_DIR) -> list[Scenario]:
    """Every dev and test scenario; the training set must share nothing with them."""
    return [s for split in ("dev", "test") for s in load_scenarios(split, scenarios_root)]


def held_out_present(scenarios_root: Path = SCENARIOS_DIR) -> list[Scenario]:
    """Like `held_out`, for whichever of the dev and test files exist under `scenarios_root`."""
    return [s for split in ("dev", "test") if (scenarios_root / f"{split}.jsonl").is_file()
            for s in load_scenarios(split, scenarios_root)]


def load_train_scenarios(root: Path = TRAIN_DIR) -> list[Scenario]:
    return load_scenarios("train", root)


def write_review_sample(scenarios: list[Scenario], path: Path) -> Path:
    """A short sample for the person to read: the first case of each category, in full."""
    lines = ["# Training scenario review sample", "",
             "Invented text, written for training only (origin: synthetic). Check each gold label and mode.", ""]
    seen: set[str] = set()
    for s in scenarios:
        if s.category in seen:
            continue
        seen.add(s.category)
        lines += [f"## {s.id} ({s.category}, card mode: {mode_of(s)})"]
        lines += [f"- {t.id} [{t.speaker}]: {t.text}" for t in s.conversation.turns]
        lines += [f"- question: {s.request.question}", f"- gold_label: {s.gold_label}",
                  f"- gold_answer: {s.gold_answer}", f"- gold_evidence: {s.gold_evidence_turn_ids}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build(out_dir: Path = TRAIN_DIR) -> dict:
    scs = build_train_scenarios()
    problems = overlap_problems(scs, held_out())
    if problems:
        raise RuntimeError("the training set overlaps the held-out scenarios: " + "; ".join(problems))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.jsonl").write_text("".join(s.model_dump_json() + "\n" for s in scs), encoding="utf-8")
    write_review_sample(scs, out_dir / "review-sample.md")
    summary = counts(scs)
    (out_dir / "counts.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(build(), indent=1))
