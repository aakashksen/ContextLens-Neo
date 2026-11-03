"""The local QMSum scenario file for retrieval experiments: dev conversations only, specific queries only
(general queries have no gold span). The text stays in git-ignored datasets/raw/; only the manifest is committed."""
import json
from pathlib import Path

from contextlens.data.loaders import load_gold
from contextlens.eval.records import ROOT, write_local_manifest

SPLIT = "dev"


def build_qmsum_dev(raw_root: Path = Path("datasets/raw"), splits_dir: Path = Path("datasets/splits"),
                    root: Path = ROOT) -> int:
    members = set(json.loads((splits_dir / "qmsum.json").read_text(encoding="utf-8"))["splits"][SPLIT])
    scenarios = []
    for s in load_gold("qmsum", raw_root):
        if s.conversation.id in members and s.gold_evidence_turn_ids:
            conv = s.conversation.model_copy(update={"split": SPLIT})
            scenarios.append(s.model_copy(update={"conversation": conv}))
    write_local_manifest("qmsum", {SPLIT: scenarios}, root)
    return len(scenarios)


if __name__ == "__main__":
    print(build_qmsum_dev())
