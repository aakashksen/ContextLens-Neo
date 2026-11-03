"""Rescore the chosen E4 checkpoint on dev and save its probabilities, with no training and no test split.

T5.2's DECODE records hold only labels for the trained model. This saves the full probability vector per case, so a
threshold can be chosen afterwards. Output is ids and probabilities only (DECODE text must not be tracked), written
under the git-ignored results/local/.

Run (CPU is enough): CUDA_VISIBLE_DEVICES= python -m contextlens.consistency.rescore --data-root <folder with models/ and datasets/raw/>
"""
import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

from contextlens.consistency import model as cm
from contextlens.eval.records import ROOT, SCENARIO_SOURCES
from contextlens.replay.runner import Prediction
from contextlens.schemas import Scenario

CHECKPOINT_REL = Path("models/e4/lr3e-05_ep3")
OUT_REL = Path("results/local")
# T5.2's records that chose this checkpoint (the rerun after the float16 fix).
RECORDS = {"scenarios": "E4-20261103T102658868337Z", "decode": "E4-20261103T102930546849Z"}
SPLIT = "dev"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checkpoint(checkpoint: Path, records: list[Path]) -> str:
    """The checkpoint's weights hash, after checking it equals `checkpoint_sha256` in every record."""
    actual = sha256_file(checkpoint / "model.safetensors")
    for r in records:
        want = json.loads(r.read_text(encoding="utf-8"))["config"]["checkpoint_sha256"]
        if want != actual:
            raise ValueError(f"the checkpoint's sha256 does not match {r.name}: {actual} vs {want}")
    return actual


def load_dev(manifest: Path, data: Path) -> list[Scenario]:
    """The dev scenarios in `data`, after checking the file hash and the case ids against the committed manifest."""
    entry = json.loads(manifest.read_text(encoding="utf-8"))["splits"][SPLIT]
    if sha256_file(data) != entry["hash"]:
        raise ValueError(f"{data.name} does not match the hash in {manifest.name}")
    scs = [Scenario.model_validate_json(ln) for ln in data.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if [s.id for s in scs] != entry["ids"]:
        raise ValueError(f"the ids in {data.name} do not match {manifest.name}")
    return scs


def rescore(scenarios: list[Scenario], proba: Callable[[list[cm.Example]], list[list[float]]],
            labels: tuple[str, ...]) -> list[dict]:
    """One row per scenario, the way E4 scored it: the turn at as_of_seq is the statement, the turns before it the context."""
    from contextlens.experiments.e4_consistency import _judged

    examples = []
    for s in scenarios:
        statement, earlier = _judged(s)
        examples.append(cm.example_from_turns(statement, earlier))
    return [{"id": s.id, "probs": dict(zip(labels, p))} for s, p in zip(scenarios, proba(examples))]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def agreement(rows: list[dict], saved: list[Prediction], *, binary: bool, threshold: float = cm.THRESHOLD) -> tuple[int, int]:
    """How many rescored rows give the label T5.2 saved for the trained model (same ids, same order). `binary`:
    DECODE rows only say contradiction or not."""
    from contextlens.eval.series import predicted_label

    if [r["id"] for r in rows] != [p.scenario_id for p in saved]:
        raise ValueError("the rescored ids and the saved prediction ids differ")
    agree = 0
    for r, p in zip(rows, saved):
        best = max(r["probs"], key=r["probs"].__getitem__)
        label = best if r["probs"][best] >= threshold else "insufficient"
        got = predicted_label(p)
        agree += (label == "contradiction") == (got == "contradiction") if binary else label == got
    return agree, len(rows)


def read_saved(root: Path, run_id: str) -> list[Prediction]:
    path = root / "results/runs" / f"{run_id}.trained.predictions.jsonl"
    return [Prediction.model_validate_json(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=ROOT,
                    help="folder holding models/e4/ and datasets/raw/ (git-ignored, so not in a worktree)")
    ap.add_argument("--root", type=Path, default=ROOT, help="folder holding the committed manifests and results/runs/")
    args = ap.parse_args()
    import torch

    if torch.cuda.is_available():
        print("note: a GPU is visible; set CUDA_VISIBLE_DEVICES= to score on the CPU", file=sys.stderr)
    ckpt = args.data_root / CHECKPOINT_REL
    sha = verify_checkpoint(ckpt, [args.root / "results/runs" / f"{r}.json" for r in RECORDS.values()])
    print(f"checkpoint sha256 matches both records: {sha}")
    start = time.perf_counter()
    model, tokenizer, labels = cm.load_checkpoint(ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    for source, run_id in RECORDS.items():
        manifest_rel, dir_rel = SCENARIO_SOURCES[source]
        scs = load_dev(args.root / manifest_rel, args.data_root / dir_rel / f"{SPLIT}.jsonl")
        t0 = time.perf_counter()
        rows = rescore(scs, lambda ex: cm.predict_proba(model, tokenizer, ex), labels)
        write_rows(args.root / OUT_REL / f"rescored-{source}-{SPLIT}.jsonl", rows)
        agree, n = agreement(rows, read_saved(args.root, run_id), binary=source == "decode")
        print(f"{source}: {n} cases rescored in {time.perf_counter() - t0:.0f} s on {device}; "
              f"{agree}/{n} agree with the label T5.2 saved")
    print(f"total {time.perf_counter() - start:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
