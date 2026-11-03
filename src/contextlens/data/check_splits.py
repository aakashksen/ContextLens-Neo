import hashlib
import json
import sys
from pathlib import Path

from contextlens.data.loaders import decode_split_hashes, load_dataset
from contextlens.eval.records import SCENARIO_SOURCES
from contextlens.data.splits import RAW_DIR, SPLITS_DIR, dataset_hash, make_splits


def _check_scenarios(path: Path, m: dict, root: Path) -> list[str]:
    source = m.get("source", "scenarios")
    if source not in SCENARIO_SOURCES:
        return [f"{path.name}: unknown scenario source {source!r}; expected one of {sorted(SCENARIO_SOURCES)}"]
    directory = root / SCENARIO_SOURCES[source][1]
    problems: list[str] = []
    for name, entry in m["splits"].items():
        f = directory / f"{name}.jsonl"
        want = entry.get("ids")
        if not f.is_file():
            if source == "scenarios":
                problems.append(f"{path.name}: committed scenario file {name} is missing")
                continue
            print(f"{path.name}: local scenario file {name} not present, only the manifest's ids checked")
            if want is not None and len(want) != len(set(want)):
                problems.append(f"{path.name}: {name} has duplicate ids in the manifest")
            continue
        raw = f.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["hash"]:
            problems.append(f"{path.name}: scenario file {name} differs from the manifest")
        ids = [json.loads(line)["id"] for line in raw.decode("utf-8").splitlines() if line.strip()]
        if len(ids) != len(set(ids)):
            problems.append(f"{path.name}: {name} has duplicate ids in the file")
        if "n" in entry and entry["n"] != len(ids):
            problems.append(f"{path.name}: {name} has n={entry['n']} in the manifest but {len(ids)} cases in the file")
        if want is not None and want != ids:
            problems.append(f"{path.name}: {name} ids differ from the manifest")
    return problems


def check(splits_dir: Path = SPLITS_DIR, raw_dir: Path = RAW_DIR) -> list[str]:
    problems: list[str] = []
    manifests = sorted(splits_dir.glob("*.json"))
    if not manifests:
        problems.append(f"no split manifests in {splits_dir}")
    for path in manifests:
        m = json.loads(path.read_text(encoding="utf-8"))
        if m.get("kind") == "scenarios":
            problems += _check_scenarios(path, m, splits_dir.parent.parent)
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
        fresh = decode_split_hashes(raw_dir) if m.get("split_rule") == "official" else make_splits(convs, seed=m["seed"])
        if fresh != s:
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
