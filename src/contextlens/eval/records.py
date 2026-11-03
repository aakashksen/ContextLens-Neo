import hashlib
import json
import os
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


RECORD_FILE = re.compile(
    r"^\?\? results/runs/([^/ ]+-\d{8}T\d{12}Z(\.[A-Za-z0-9_-]+\.predictions\.jsonl|\.json)|\.[^/ ]+\.tmp)$"
    r"|^(\?\?| M) results/error-analysis-e2\.md$"  # E2 writes it before its cloud record, so it must not block that record
)


def write_atomic(path: Path, text: str) -> None:
    """Write beside the target, fsync, then rename. The temp name starts with a dot and ends in .tmp, so no
    record glob (`*Z.json`) matches it; on failure it is removed and the target is untouched."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def is_dirty(root: Path) -> bool:
    """Uncommitted changes, except new untracked run-record and predictions files and E2's error analysis."""
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
        write_atomic(root / directory / f"{split}.jsonl", "".join(s.model_dump_json() + "\n" for s in scs))
        splits[split] = {"hash": split_hash(split, root, source), "n": len(scs), "ids": [s.id for s in scs]}
    path = root / manifest_rel
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, json.dumps({"kind": "scenarios", "source": source, "splits": splits}, indent=1) + "\n")
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
    write_atomic(path, json.dumps({"kind": "scenarios", "splits": splits}, indent=1) + "\n")
    return path


TASKS_REL = Path("orchestration/tasks.json")


def labels_reviewed(root: Path, scenario_source: str = "scenarios") -> bool:
    """True only when the person has signed H2 (status done in tasks.json) and the record's labels are the
    scenario labels H2 reviewed. Labels from another source (QMSum, DECODE, ...) were not in H2's sample, so
    they stay unreviewed here. This is the only place the flag is set; a runner's config cannot override it."""
    if scenario_source != "scenarios":
        return False
    try:
        tasks = json.loads((root / TASKS_REL).read_text(encoding="utf-8"))["tasks"]
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return any(isinstance(t, dict) and t.get("id") == "H2" and t.get("status") == "done" for t in tasks)


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
        write_atomic(pred_path, "".join(p.model_dump_json() + "\n" for p in rows))
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
        "labels_reviewed": labels_reviewed(root, scenario_source),
        "predictions": pred_paths,
        "started_at": config.get("started_at", now.isoformat()),
        "finished_at": now.isoformat(),
    }
    path = runs / f"{run_id}.json"
    write_atomic(path, json.dumps(record, indent=1) + "\n")
    return path
