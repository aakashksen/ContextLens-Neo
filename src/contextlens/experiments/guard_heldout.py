"""One-shot check of the instruction guard on a separately written set of invented turns.

The set (`datasets/guard_heldout/turns.jsonl`) holds injected orders and ordinary talk that looks like them. It was
written and committed before the bare-order patterns, and the patterns were never run against it while they were being
written. This module is the only thing that does, once: the record is named after the set's hash and an existing record
makes a second run refuse. Any change to the patterns afterwards needs a new set, so this one stays an honest figure.

Run (the person starts it; no GPU, no network):
  .venv/Scripts/python -m contextlens.experiments.guard_heldout
"""
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contextlens.eval.records import ROOT, git, head_commit, is_dirty, write_atomic
from contextlens.guard.instructions import instruction_kind

SET_REL = Path("datasets/guard_heldout/turns.jsonl")
GUARD_REL = Path("src/contextlens/guard/instructions.py")
OUT_DIR = Path("results/guard-heldout")
NOTE = ("The author wrote both the patterns and this set, so the two are not independent: the set was written first and "
        "committed, and the patterns were not run against it before this record, but the writer knew what kinds of text "
        "it held. Read the rates as an upper bound on how well the patterns generalise, from one writer and a small set.")


def evaluate(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Catch rate on the injected orders and false-drop rate on the ordinary turns, with the ids of every miss and drop."""
    inj = [r for r in rows if r["kind"] == "injection"]
    ordinary = [r for r in rows if r["kind"] == "ordinary"]
    caught = [r for r in inj if instruction_kind(r["text"])]
    dropped = [r for r in ordinary if instruction_kind(r["text"])]
    return {"injections": len(inj), "injections_caught": len(caught),
            "catch_rate": len(caught) / len(inj) if inj else None,
            "caught_by_class": dict(sorted(Counter(instruction_kind(r["text"]) for r in caught).items())),
            "missed": [{"id": r["id"], "text": r["text"]} for r in inj if r not in caught],
            "ordinary": len(ordinary), "ordinary_dropped": len(dropped),
            "false_drop_rate": len(dropped) / len(ordinary) if ordinary else None,
            "dropped": [{"id": r["id"], "text": r["text"], "class": instruction_kind(r["text"])} for r in dropped]}


def load_set(root: Path = ROOT) -> tuple[list[dict[str, str]], str]:
    raw = (root / SET_REL).read_bytes()
    rows = [json.loads(ln) for ln in raw.decode("utf-8").splitlines() if ln.strip()]
    if not rows or any(r["kind"] not in ("injection", "ordinary") for r in rows):
        raise ValueError(f"{SET_REL} must hold turns of kind injection or ordinary")
    return rows, hashlib.sha256(raw).hexdigest()


def commit_of(root: Path, rel: Path, *flags: str) -> str | None:
    """The oldest commit that touched `rel`, or with "-1" the newest."""
    out = git(root, "log", *flags, "--format=%H", "--", rel.as_posix())
    return out.stdout.split()[-1] if out.returncode == 0 and out.stdout.split() else None


def run(root: Path = ROOT) -> Path:
    if is_dirty(root):
        raise RuntimeError("the working tree has uncommitted changes; commit first")
    rows, digest = load_set(root)
    path = root / OUT_DIR / f"{digest[:12]}.json"
    if path.exists():
        raise RuntimeError(f"{path.name} already exists: this set has been run once and is not rerun after any change; "
                           "write a new set")
    record = {"set": SET_REL.as_posix(), "set_sha256": digest, "set_first_commit": commit_of(root, SET_REL),
              "guard_sha256": hashlib.sha256((root / GUARD_REL).read_bytes()).hexdigest(),
              "guard_last_commit": commit_of(root, GUARD_REL, "-1"), "head": head_commit(root),
              "run_at": datetime.now(timezone.utc).isoformat(), "claude_calls": 0,
              "result": evaluate(rows), "independence_note": NOTE}
    (root / OUT_DIR).mkdir(parents=True, exist_ok=True)
    write_atomic(path, json.dumps(record, indent=1) + "\n")
    return path


def main() -> int:
    try:
        print(run())
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        print(f"guard held-out check stopped: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
