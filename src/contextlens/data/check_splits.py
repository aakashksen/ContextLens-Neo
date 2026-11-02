import hashlib
import json
import sys
from pathlib import Path

from contextlens.data.loaders import load_dataset
from contextlens.data.splits import RAW_DIR, SPLITS_DIR, dataset_hash, make_splits


def check(splits_dir: Path = SPLITS_DIR, raw_dir: Path = RAW_DIR) -> list[str]:
    problems: list[str] = []
    manifests = sorted(splits_dir.glob("*.json"))
    if not manifests:
        problems.append(f"no split manifests in {splits_dir}")
    for path in manifests:
        m = json.loads(path.read_text(encoding="utf-8"))
        if m.get("kind") == "scenarios":
            for name, entry in m["splits"].items():
                f = splits_dir.parent / "scenarios" / f"{name}.jsonl"
                if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest() != entry["hash"]:
                    problems.append(f"{path.name}: scenario file {name} differs from the manifest")
            continue
        s = m["splits"]
        everything = [i for part in s.values() for i in part]
        if len(everything) != len(set(everything)):
            problems.append(f"{path.name}: a conversation is in two splits")
        if not (raw_dir / m["dataset"]).is_dir():
            print(f"{path.name}: raw data not present, only disjointness checked")
            continue
        convs = load_dataset(m["dataset"], raw_dir)
        if dataset_hash(convs) != m["hash"]:
            problems.append(f"{path.name}: dataset hash differs from the manifest")
        if make_splits(convs, seed=m["seed"]) != s:
            problems.append(f"{path.name}: splits differ from a fresh split")
    return problems


def main() -> int:
    problems = check()
    for p in problems:
        print("FAIL", p)
    if not problems:
        print("splits ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
