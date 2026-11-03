"""Find a run record's chosen checkpoint and check it is the one that was scored."""
import hashlib
from pathlib import Path
from typing import Any

from contextlens.eval.records import ROOT


def chosen_checkpoint(record: dict[str, Any], root: Path = ROOT) -> Path:
    """The winner's checkpoint directory under `root`, after its model.safetensors matches `config.checkpoint_sha256`.

    Older records hold an absolute path from a worktree that is gone; those are read from their `models/...` tail."""
    setting = record["config"]["winner"]
    raw = next(a["checkpoint"] for a in record["attempts"] if a.get("kind") == "training" and a["setting"] == setting)
    parts = Path(raw).parts
    path = root / raw if not Path(raw).is_absolute() else root.joinpath(*parts[parts.index("models"):])
    weights = path / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"no checkpoint at {path}")
    if hashlib.sha256(weights.read_bytes()).hexdigest() != record["config"]["checkpoint_sha256"]:
        raise ValueError(f"{weights} does not match the record's checkpoint_sha256")
    return path
