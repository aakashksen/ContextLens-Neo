import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contextlens.replay.runner import Prediction
from contextlens.schemas import Scenario

ROOT = Path(__file__).resolve().parents[3]
RUNS_REL = Path("results/runs")


def git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)


RECORD_FILE = re.compile(r"^\?\? results/runs/[^/ ]+-\d{8}T\d{12}Z(\.[A-Za-z0-9_-]+\.predictions\.jsonl|\.json)$")


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


MANIFEST_REL = Path("datasets/splits/scenarios.json")
# Where each scenario source keeps its cases and the committed manifest that pins them. QMSum text cannot be
# committed, so its scenario file sits in the git-ignored datasets/raw/ and only its manifest is committed.
SCENARIO_SOURCES = {
    "scenarios": (MANIFEST_REL, Path("datasets/scenarios")),
    "qmsum": (Path("datasets/splits/qmsum-scenarios.json"), Path("datasets/raw/scenarios/qmsum")),
    "decode": (Path("datasets/splits/decode-scenarios.json"), Path("datasets/raw/scenarios/decode")),
}
# Prediction rows of these sources may hold ids and metrics data only (a bare label is fine): no quotes, cards or raw output.
IDS_ONLY_SOURCES = {"qmsum", "decode"}


def split_hash(split: str, root: Path = ROOT, source: str = "scenarios") -> str:
    """Hash of the scenario file for a split: the cases a run is scored on."""
    return hashlib.sha256((root / SCENARIO_SOURCES[source][1] / f"{split}.jsonl").read_bytes()).hexdigest()


def write_local_manifest(source: str, scenarios: dict[str, list[Scenario]], root: Path = ROOT) -> Path:
    """Write a source's scenario files (git-ignored) and its manifest of hash, count and ids (committed)."""
    manifest_rel, directory = SCENARIO_SOURCES[source]
    (root / directory).mkdir(parents=True, exist_ok=True)
    splits = {}
    for split, scs in scenarios.items():
        (root / directory / f"{split}.jsonl").write_text("".join(s.model_dump_json() + "\n" for s in scs), encoding="utf-8")
        splits[split] = {"hash": split_hash(split, root, source), "n": len(scs), "ids": [s.id for s in scs]}
    path = root / manifest_rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"kind": "scenarios", "source": source, "splits": splits}, indent=1) + "\n", encoding="utf-8")
    return path


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


SYSTEM_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def write_run(
    experiment: str,
    split: str,
    config: dict[str, Any],
    models: dict[str, str],
    prompt_versions: dict[str, str],
    metrics: dict[str, dict[str, float | None]],
    intervals: dict[str, dict[str, tuple[float, float]]],
    attempts: list[dict[str, Any]],
    predictions: dict[str, list[Prediction]],
    dataset_hash: str,
    *,
    comparisons: list[dict[str, Any]] | None = None,
    supersedes: list[dict[str, str]] | None = None,
    subset: str | None = None,
    scenario_source: str = "scenarios",
    root: Path = ROOT,
    allow_dirty: bool = False,
) -> Path:
    if scenario_source not in SCENARIO_SOURCES:
        raise ValueError(f"unknown scenario source {scenario_source!r}; expected one of {sorted(SCENARIO_SOURCES)}")
    bad = [name for name in predictions if not SYSTEM_NAME.match(name)]
    if not predictions or bad:
        raise ValueError(f"predictions must map system names (letters, digits, _ and -) to rows: {bad}")
    dirty = is_dirty(root)
    if dirty and not allow_dirty:
        raise RuntimeError("working tree has uncommitted changes; commit first")
    now = datetime.now(timezone.utc)
    run_id = f"{experiment}-{now.strftime('%Y%m%dT%H%M%S%fZ')}"
    runs = root / RUNS_REL
    runs.mkdir(parents=True, exist_ok=True)
    pred_paths = {}
    for name, rows in predictions.items():
        pred_path = runs / f"{run_id}.{name}.predictions.jsonl"
        pred_path.write_text("".join(p.model_dump_json() + "\n" for p in rows), encoding="utf-8")
        pred_paths[name] = pred_path.relative_to(root).as_posix()
    record = {
        "run_id": run_id,
        "experiment": experiment,
        "git_commit": head_commit(root),
        "dirty": dirty,
        "dataset_hash": dataset_hash,
        "split": split,
        "scenario_source": scenario_source,
        "config": config,
        "models": models,
        "prompt_versions": prompt_versions,
        "n": len(next(iter(predictions.values()))),
        "metrics": metrics,
        "intervals": {sys: {k: list(v) for k, v in d.items()} for sys, d in intervals.items()},
        "comparisons": comparisons or [],
        "supersedes": supersedes or [],
        **({"subset": subset} if subset is not None else {}),
        "attempts": attempts,
        "claude_calls": config.get("claude_calls", 0),
        "labels_reviewed": config.get("labels_reviewed", False),
        "predictions": pred_paths,
        "started_at": config.get("started_at", now.isoformat()),
        "finished_at": now.isoformat(),
    }
    path = runs / f"{run_id}.json"
    path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return path
