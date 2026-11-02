import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contextlens.replay.runner import Prediction

ROOT = Path(__file__).resolve().parents[3]
RUNS_REL = Path("results/runs")


def git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)


RECORD_FILE = re.compile(r"^\?\? results/runs/[^/ ]+-\d{8}T\d{12}Z(\.predictions\.jsonl|\.json)$")


def is_dirty(root: Path) -> bool:
    """Uncommitted changes, except new untracked run-record and predictions files."""
    out = git(root, "status", "--porcelain", "-uall")
    if out.returncode != 0:
        raise RuntimeError(f"git status failed: {out.stderr.strip()}")
    return any(ln and not RECORD_FILE.match(ln) for ln in out.stdout.splitlines())


def head_commit(root: Path) -> str:
    out = git(root, "rev-parse", "HEAD")
    if out.returncode != 0:
        raise RuntimeError(f"git rev-parse failed: {out.stderr.strip()}")
    return out.stdout.strip()


def split_hash(split: str, root: Path = ROOT) -> str:
    """Hash of the scenario file for a split: the cases a run is scored on."""
    return hashlib.sha256((root / "datasets/scenarios" / f"{split}.jsonl").read_bytes()).hexdigest()


MANIFEST_REL = Path("datasets/splits/scenarios.json")


def write_scenarios_manifest(root: Path = ROOT) -> Path:
    """Fix each scenario split by content hash and case count."""
    from contextlens.data.scenarios import load_scenarios

    splits = {}
    for f in sorted((root / "datasets/scenarios").glob("*.jsonl")):
        splits[f.stem] = {
            "hash": split_hash(f.stem, root),
            "n": len(load_scenarios(f.stem, root / "datasets/scenarios")),
        }
    path = root / MANIFEST_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"kind": "scenarios", "splits": splits}, indent=1) + "\n", encoding="utf-8")
    return path


def write_run(
    experiment: str,
    split: str,
    config: dict[str, Any],
    models: dict[str, str],
    prompt_versions: dict[str, str],
    metrics: dict[str, float],
    intervals: dict[str, tuple[float, float]],
    attempts: list[dict[str, Any]],
    predictions: list[Prediction],
    dataset_hash: str,
    *,
    root: Path = ROOT,
    allow_dirty: bool = False,
) -> Path:
    dirty = is_dirty(root)
    if dirty and not allow_dirty:
        raise RuntimeError("working tree has uncommitted changes; commit first")
    now = datetime.now(timezone.utc)
    run_id = f"{experiment}-{now.strftime('%Y%m%dT%H%M%S%fZ')}"
    runs = root / RUNS_REL
    runs.mkdir(parents=True, exist_ok=True)
    pred_path = runs / f"{run_id}.predictions.jsonl"
    pred_path.write_text("".join(p.model_dump_json() + "\n" for p in predictions), encoding="utf-8")
    record = {
        "run_id": run_id,
        "experiment": experiment,
        "git_commit": head_commit(root),
        "dirty": dirty,
        "dataset_hash": dataset_hash,
        "split": split,
        "config": config,
        "models": models,
        "prompt_versions": prompt_versions,
        "n": len(predictions),
        "metrics": metrics,
        "intervals": {k: list(v) for k, v in intervals.items()},
        "attempts": attempts,
        "claude_calls": config.get("claude_calls", 0),
        "labels_reviewed": config.get("labels_reviewed", False),
        "predictions_path": pred_path.relative_to(root).as_posix(),
        "started_at": config.get("started_at", now.isoformat()),
        "finished_at": now.isoformat(),
    }
    path = runs / f"{run_id}.json"
    path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return path
