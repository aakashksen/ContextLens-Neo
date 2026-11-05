"""One-shot check of the instruction guard on a set of turns written by someone who has not seen the patterns.

The set is given as a path (`--set`, no default): JSONL lines of `id`, `text` and `kind` (`injection` or `ordinary`). The
record is named after the set's hash and an existing record makes a second run refuse; any change to the patterns
afterwards needs a new set. The record holds the set's hash and the last commit of `guard/instructions.py`, which should be
the commit the patterns were frozen at.

Run (the person starts it; no GPU, no network):
  .venv/Scripts/python -m contextlens.experiments.guard_heldout --set <path to the set>
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contextlens.eval.records import ROOT, git, head_commit, is_dirty, write_atomic
from contextlens.guard.instructions import instruction_kind

GUARD_REL = Path("src/contextlens/guard/instructions.py")
OUT_DIR = Path("results/guard-heldout")
NOTE = ("The set was written by an independent writer who had not seen the patterns, and the patterns were frozen at "
        "`guard_last_commit` before this run. Independence is still limited: one writer, a small set, and a writer who knew "
        "what the guard is for. Read the rates as one blind writer's view, not as a general figure.")


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


def load_set(path: Path) -> tuple[list[dict[str, str]], str]:
    raw = path.read_bytes()
    rows = [json.loads(ln) for ln in raw.decode("utf-8").splitlines() if ln.strip()]
    if not rows or any(r.get("kind") not in ("injection", "ordinary") or not r.get("text") for r in rows):
        raise ValueError(f"{path} must hold lines with a text and a kind of injection or ordinary")
    return rows, hashlib.sha256(raw).hexdigest()


def commit_of(root: Path, rel: Path) -> str | None:
    """The newest commit that touched `rel`."""
    out = git(root, "log", "-1", "--format=%H", "--", rel.as_posix())
    return out.stdout.strip() or None if out.returncode == 0 else None


def run(set_path: Path, root: Path = ROOT) -> Path:
    if is_dirty(root):
        raise RuntimeError("the working tree has uncommitted changes; commit first")
    rows, digest = load_set(set_path)
    path = root / OUT_DIR / f"{digest[:12]}.json"
    if path.exists():
        raise RuntimeError(f"{path.name} already exists: this set has been run once and is not rerun after any change; "
                           "write a new set")
    record = {"set_sha256": digest, "set_rows": len(rows), "guard_sha256": hashlib.sha256((root / GUARD_REL).read_bytes()).hexdigest(),
              "guard_last_commit": commit_of(root, GUARD_REL), "head": head_commit(root),
              "run_at": datetime.now(timezone.utc).isoformat(), "claude_calls": 0,
              "result": evaluate(rows), "independence_note": NOTE}
    (root / OUT_DIR).mkdir(parents=True, exist_ok=True)
    write_atomic(path, json.dumps(record, indent=1) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", type=Path, required=True, dest="set_path", help="JSONL of id, text, kind (injection or ordinary)")
    args = ap.parse_args(argv)
    try:
        print(run(args.set_path))
    except (RuntimeError, OSError, ValueError, KeyError) as exc:
        print(f"guard held-out check stopped: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
