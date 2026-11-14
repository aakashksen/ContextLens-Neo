from pathlib import Path

from contextlens.data.loaders import Pair, load_pairs
from contextlens.data.scenarios import SCENARIOS_DIR, load_scenarios
from contextlens.data.splits import RAW_DIR
from contextlens.schemas import ConsistencyLabel

# Mirrored by the table in docs/datasets.md (section "DECODE label map"); test_label_map_matches_datasets_doc keeps them equal.
DECODE_LABEL_MAP: dict[str, ConsistencyLabel] = {"contradiction": "contradiction", "non_contradiction": "consistent"}


def _decode_pairs(split: str, raw_root: Path) -> list[Pair]:
    return [
        p.model_copy(update={"label": DECODE_LABEL_MAP[p.label]})
        for p in load_pairs(raw_root) if p.split == split
    ]


def _scenario_pairs(split: str, scenarios_root: Path) -> list[Pair]:
    if not (scenarios_root / f"{split}.jsonl").is_file():
        return []  # the written scenarios exist for dev and test only
    out = []
    for s in load_scenarios(split, scenarios_root):
        if s.gold_label is None:
            continue
        texts = [t.text for t in s.conversation.turns[: s.request.as_of_seq + 1]]
        out.append(Pair(conversation_id=s.conversation.id, statement=texts[-1], earlier=texts[:-1],
                        label=s.gold_label, source="scenarios", split=split))  # type: ignore[arg-type]
    return out


def build_pairs(split: str, raw_root: Path = RAW_DIR, scenarios_root: Path = SCENARIOS_DIR) -> list[Pair]:
    """DECODE pairs (labels mapped by DECODE_LABEL_MAP) followed by our scenario pairs, each tagged with its source."""
    return _decode_pairs(split, raw_root) + _scenario_pairs(split, scenarios_root)


def assert_trainable(pairs: list[Pair], raw_root: Path = RAW_DIR, scenarios_root: Path = SCENARIOS_DIR) -> None:
    """Stop if any pair's conversation is also in the dev or test split. Every training path calls this."""
    for split in ("dev", "test"):
        held = {p.conversation_id for p in build_pairs(split, raw_root, scenarios_root)}
        if held & {p.conversation_id for p in pairs}:
            raise RuntimeError(f"a {split} conversation reached training")

