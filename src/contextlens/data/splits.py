import hashlib
import json
import random
from pathlib import Path

from contextlens.data.loaders import load_dataset
from contextlens.schemas import Conversation

SPLITS_DIR = Path("datasets/splits")
RAW_DIR = Path("datasets/raw")


def make_splits(
    convs: list[Conversation], seed: int = 13, ratios: tuple[float, float, float] = (0.5, 0.25, 0.25)
) -> dict[str, list[str]]:
    ids = sorted(c.id for c in convs)
    random.Random(seed).shuffle(ids)
    n_train = round(len(ids) * ratios[0])
    n_dev = round(len(ids) * ratios[1])
    return {
        "train": sorted(ids[:n_train]),
        "dev": sorted(ids[n_train : n_train + n_dev]),
        "test": sorted(ids[n_train + n_dev :]),
    }


def dataset_hash(convs: list[Conversation]) -> str:
    h = hashlib.sha256()
    for c in sorted(convs, key=lambda c: c.id):
        h.update(json.dumps([c.id, [(t.id, t.speaker, t.text) for t in c.turns]]).encode())
    return h.hexdigest()


def write_manifest(name: str, root: Path = RAW_DIR, out_dir: Path = SPLITS_DIR) -> Path:
    convs = load_dataset(name, root)
    manifest = {"dataset": name, "seed": 13, "hash": dataset_hash(convs), "splits": make_splits(convs)}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    for ds in ("longmemeval", "ami"):
        print(write_manifest(ds))
