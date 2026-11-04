"""Rebuild the derived fields of finished records from their saved rows, with no model call.

A record keeps its predictions; everything else in it (metrics, intervals, comparisons, notes) is computed from them.
When a rule for computing those fields was missing or wrong (here: McNemar for the paired answered/abstained outcomes,
and the label wording), the record is rebuilt rather than edited: a new record is written with byte-identical
predictions, the old one is listed as superseded with the reason, and the old file stays on disk.

What is checked before anything is written: the dataset hash and the case ids are the old record's; the saved rows read
back and write out to the same bytes as the old files; the metrics and intervals recompute to the saved values. After the
write the new prediction files are hashed against the old ones. Nothing here calls a model, Ollama or the cloud client.

Run (the tree must be clean and this file committed): python -m contextlens.experiments.rebuild_records [RUN_ID ...]
Records are rebuilt in the order given, so a cloud record can name the rebuilt local record it was made from.
"""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from contextlens.eval.check_run import _same_pair, _same_value, recompute_comparison, recompute_intervals, recompute_metrics
from contextlens.eval.records import (
    ROOT,
    RUNS_REL,
    SUBSETS_REL,
    is_dirty,
    label_text,
    load_verified,
    write_run,
)
from contextlens.experiments import e3_team as e3
from contextlens.experiments.e2_context import RunBroken
from contextlens.replay.runner import Prediction

REASON = ("rebuilt from the same saved rows with no model call: exact McNemar was missing on the paired answered/abstained "
          "outcomes (answer_when_answerable, abstain_when_insufficient), and the label wording in the notes did not follow "
          "labels_reviewed. Predictions are byte-identical; only derived fields differ")
DEFAULT_IDS = (
    "E3-20261103T221004658482Z",  # cards
    "E2-20261103T224548534556Z",  # E2 local
    "E2-20261103T230638094732Z",  # E2 cloud arm, made from the local one above
    "E3-20261103T200704046158Z",  # consistency, scenarios
    "E3-20261103T201657902326Z",  # consistency, DECODE
)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def _load(root: Path, run_id: str) -> dict[str, Any]:
    path = root / RUNS_REL / f"{run_id}.json"
    if not path.is_file():
        raise RunBroken(f"no record {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(root: Path, rec: dict[str, Any]) -> tuple[dict[str, list[Prediction]], dict[str, str]]:
    """The saved rows per system, after checking they write back to the same bytes; and each file's hash."""
    out: dict[str, list[Prediction]] = {}
    hashes: dict[str, str] = {}
    for name, rel in rec["predictions"].items():
        raw = (root / rel).read_bytes()
        out[name] = [Prediction.model_validate_json(ln) for ln in raw.decode("utf-8").splitlines() if ln.strip()]
        again = "".join(p.model_dump_json() + "\n" for p in out[name]).encode("utf-8")
        if _digest(again) != _digest(raw):
            raise RunBroken(f"system {name}: the saved rows do not write back to the same bytes, so they cannot be rebuilt as they are")
        hashes[name] = _digest(raw)
    return out, hashes


def rebuild(root: Path, run_id: str, reason: str = REASON, renames: dict[str, str] | None = None,
            say: Any = print) -> Path:
    """Write the rebuilt record for `run_id` and return its path."""
    if is_dirty(root):
        raise RunBroken("the working tree has uncommitted changes; commit first, or the record cannot be written")
    old = _load(root, run_id)
    scs, digest = load_verified(old["scenario_source"], old["split"], root)
    if digest != old["dataset_hash"]:
        raise RunBroken(f"{run_id}: the scenario file is not the one the record used")
    if "subset" in old:
        ids = set(json.loads((root / SUBSETS_REL / f"{old['subset']}.json").read_text(encoding="utf-8"))["ids"])
        scs = [s for s in scs if s.id in ids]
    rows, before = _rows(root, old)
    if any([p.scenario_id for p in r] != [s.id for s in scs] for r in rows.values()):
        raise RunBroken(f"{run_id}: the rows do not cover the record's cases in order")

    config = copy.deepcopy(old["config"])
    cards = old["experiment"] == e3.EXPERIMENT and config.get("part") == "cards"
    if cards:
        metrics, intervals, comparisons = e3.card_stats(scs, rows)
        extras = e3.cards_extras(scs, rows, metrics, comparisons, label_text(root, old["scenario_source"]))
        config["notes"][0] = extras["finding"]
        if e3.INJECTION_VACUOUS not in config["notes"]:
            config["notes"].append(e3.INJECTION_VACUOUS)
        config.update({k: extras[k] for k in ("paired_binary", "abstention_cases", "duplicate_findings")})
    else:
        metrics = {n: {m: recompute_metrics(scs, r).get(m) for m in old["metrics"][n]} for n, r in rows.items()}
        intervals = {n: {m: iv for m, iv in recompute_intervals(scs, rows[n], list(d)).items() if iv}
                     for n, d in old["intervals"].items()}
        comparisons = []
        for c in old["comparisons"]:
            fresh = recompute_comparison(scs, rows[c["a"]], rows[c["b"]], c["metric"])
            if fresh is None:
                raise RunBroken(f"{run_id}: comparison {c['a']} vs {c['b']} on {c['metric']} cannot be recomputed")
            comparisons.append({"a": c["a"], "b": c["b"], **fresh})
        if old["experiment"] == e3.EXPERIMENT:  # the consistency part
            config["selection_data"] = "dev (selection data, not held out)"
            config["label_note"] = label_text(root, old["scenario_source"])
            tables = {sy: {src: {k: (kn[0], kn[1]) for k, kn in t.items()} for src, t in by.items()}
                      for sy, by in config["catch_far"].items()}
            config["alert_precision_note"] = e3.precision_note(tables)
            hand = config.get("hand_read_file")
            if isinstance(hand, str) and hand.endswith("/" + e3.HAND_READ_REL.as_posix()):
                config["hand_read_file"] = e3.HAND_READ_REL.as_posix()  # it named a worktree that no longer exists
    for name, saved in old["metrics"].items():  # derived values that are not being corrected must come out the same
        for m, v in saved.items():
            if not _same_value(metrics[name].get(m), v):
                raise RunBroken(f"{run_id}: {name} {m} recomputes to {metrics[name].get(m)!r}, not the saved {v!r}")
    for name, saved_iv in old["intervals"].items():
        for m, v in saved_iv.items():
            got = intervals.get(name, {}).get(m)
            if got is None or not _same_pair(got, v):
                raise RunBroken(f"{run_id}: {name} interval {m} does not recompute to the saved one")
    if "local_run" in config:
        config["local_run"] = (renames or {}).get(config["local_run"], config["local_run"])
    config["rebuilt"] = {"from": run_id, "reason": reason, "model_calls": 0,
                         "prediction_hashes_sha256": before,
                         "claude_calls_of_the_source_record": old.get("claude_calls", 0)}
    config["claude_calls"] = 0  # this rebuild made none; the source record keeps its own count
    path = write_run(
        old["experiment"], old["split"], config, old["models"], old["prompt_versions"], metrics,
        {n: {m: tuple(v) for m, v in d.items()} for n, d in intervals.items()}, old["attempts"], rows, old["dataset_hash"],
        comparisons=comparisons, supersedes=[{"run_id": run_id, "reason": reason}], subset=old.get("subset"),
        scenario_source=old["scenario_source"], root=root)
    new = json.loads(path.read_text(encoding="utf-8"))
    after = {n: _digest((root / rel).read_bytes()) for n, rel in new["predictions"].items()}
    if after != before:
        raise RunBroken(f"{path.name}: the written predictions differ from the old ones: "
                        f"{sorted(n for n in before if before[n] != after.get(n))}")
    say(f"{run_id} -> {new['run_id']}: predictions identical ({len(after)} files)")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_ids", nargs="*", default=list(DEFAULT_IDS), help="records to rebuild, in order")
    args = ap.parse_args(argv)
    renames: dict[str, str] = {}
    try:
        for rid in args.run_ids:
            renames[rid] = rebuild(ROOT, rid, renames=renames).stem
    except (RunBroken, OSError, ValueError, KeyError) as exc:
        print(f"rebuild stopped: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(renames, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
