"""The local QMSum scenario file for retrieval experiments: dev conversations only, specific queries only
(general queries have no gold span). The text stays in git-ignored datasets/raw/; only the manifest is committed."""
import json
from pathlib import Path

from contextlens.data.loaders import load_gold
from contextlens.eval.records import ROOT, write_local_manifest
from contextlens.schemas import Scenario

SPLIT = "dev"


def qmsum_scenarios(split: str, raw_root: Path = Path("datasets/raw"), splits_dir: Path = Path("datasets/splits")) -> list[Scenario]:
    """The specific-query scenarios of one split's conversations; the same rule for dev and for E8's test."""
    members = set(json.loads((splits_dir / "qmsum.json").read_text(encoding="utf-8"))["splits"][split])
    scenarios = []
    for s in load_gold("qmsum", raw_root):
        if s.conversation.id in members and s.gold_evidence_turn_ids:
            conv = s.conversation.model_copy(update={"split": split})
            scenarios.append(s.model_copy(update={"conversation": conv}))
    return scenarios


def build_qmsum_dev(raw_root: Path = Path("datasets/raw"), splits_dir: Path = Path("datasets/splits"),
                    root: Path = ROOT) -> int:
    scenarios = qmsum_scenarios(SPLIT, raw_root, splits_dir)
    write_local_manifest("qmsum", {SPLIT: scenarios}, root)
    return len(scenarios)


if __name__ == "__main__":
    print(build_qmsum_dev())
