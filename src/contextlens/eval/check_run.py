import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from contextlens.data.scenarios import load_scenarios
from contextlens.eval import metrics as M
from contextlens.eval.records import MANIFEST_REL, ROOT, RUNS_REL, git, split_hash
from contextlens.eval.stats import percentile
from contextlens.replay.runner import Prediction
from contextlens.schemas import Scenario

REQUIRED = (
    "run_id experiment git_commit dirty dataset_hash split config models prompt_versions n metrics "
    "intervals attempts claude_calls labels_reviewed predictions_path started_at finished_at"
).split()
TOLERANCE = 1e-9


def _label(p: Prediction) -> str:
    for f in p.card.findings:
        if f.label:
            return f.label
    return "insufficient"


def recompute_metrics(scenarios: list[Scenario], predictions: list[Prediction]) -> dict[str, float]:
    by_id = {p.scenario_id: p for p in predictions}
    pairs = [(s, by_id[s.id]) for s in scenarios if s.id in by_id]
    labelled = [(s.gold_label, _label(p)) for s, p in pairs if s.gold_label]
    gold, pred = [g for g, _ in labelled], [q for _, q in labelled]
    abst = M.abstention_accuracy(
        [p.card.status == "abstained" for _, p in pairs],
        [s.gold_label == "insufficient" for s, _ in pairs],
    )
    totals = [float(p.stage_latency_ms["total"]) for _, p in pairs if "total" in p.stage_latency_ms]
    out = {
        "macro_f1": M.macro_f1(gold, pred),
        "false_alert_rate": M.false_alert_rate(gold, pred),
        **abst,
    }
    if totals:
        out["latency_p50_ms"] = percentile(totals, 50)
        out["latency_p95_ms"] = percentile(totals, 95)
    return out


def _strings(x: Any) -> list[str]:
    if isinstance(x, str):
        return [x]
    if isinstance(x, dict):
        return [s for v in x.values() for s in _strings(v)]
    if isinstance(x, list):
        return [s for v in x for s in _strings(v)]
    return []


def records_for(experiment: str, root: Path = ROOT) -> list[Path]:
    return sorted((root / RUNS_REL).glob(f"{experiment}-*Z.json"))


def check_record(rec: dict[str, Any], root: Path, min_n: int = 0) -> list[str]:
    problems = [f"missing field {k}" for k in REQUIRED if k not in rec]
    if problems:
        return problems
    if rec["dirty"]:
        problems.append("record was made on a dirty tree")
    if any(re.search(r"fake|stub", s, re.I) for s in _strings(rec["models"])):
        problems.append("a model name matches fake or stub")
    if not rec["attempts"]:
        problems.append("attempts is empty")
    if git(root, "merge-base", "--is-ancestor", rec["git_commit"], "HEAD").returncode != 0:
        problems.append("git_commit is not an ancestor of HEAD")
    if min_n and rec["n"] < min_n:
        problems.append(f"n={rec['n']} is below --min-n {min_n}")

    if not rec["metrics"]:
        problems.append("metrics is empty")

    scenarios: list[Scenario] = []
    split = rec["split"]
    manifest_path = root / MANIFEST_REL
    entry = None
    if manifest_path.is_file():
        entry = json.loads(manifest_path.read_text(encoding="utf-8")).get("splits", {}).get(split)
    if entry is None:
        problems.append(f"no manifest entry for split {split}")
    elif not (root / "datasets/scenarios" / f"{split}.jsonl").is_file():
        problems.append(f"no scenario file for split {split}")
    else:
        if rec["dataset_hash"] != entry["hash"]:
            problems.append("dataset_hash does not match the split manifest")
        if split_hash(split, root) != entry["hash"]:
            problems.append("scenario file no longer matches the split manifest")
        else:
            scenarios = load_scenarios(split, root / "datasets/scenarios")
        if rec["n"] != entry["n"]:
            problems.append(f"n={rec['n']} differs from the split's {entry['n']} cases")

    pred_path = root / rec["predictions_path"]
    if not pred_path.is_file():
        problems.append("predictions file is missing")
        return problems
    lines = pred_path.read_text(encoding="utf-8").splitlines()
    preds = [Prediction.model_validate_json(x) for x in lines if x.strip()]
    if len(preds) != rec["n"]:
        problems.append(f"predictions hold {len(preds)} rows but n={rec['n']}")
    ids_ok = True
    if scenarios:
        pred_ids = [p.scenario_id for p in preds]
        if len(set(pred_ids)) != len(pred_ids) or set(pred_ids) != {s.id for s in scenarios}:
            ids_ok = False
            problems.append("prediction ids do not match the split's scenario ids exactly once each")
    if scenarios and ids_ok:
        fresh = recompute_metrics(scenarios, preds)
        for name, saved in rec["metrics"].items():
            if name not in fresh:
                problems.append(f"metric {name} cannot be recomputed from predictions")
            elif abs(fresh[name] - saved) > TOLERANCE:
                problems.append(f"metric {name} does not recompute from predictions")
    return problems


def check(experiment: str, *, root: Path = ROOT, min_n: int = 0, split: str | None = None) -> list[str]:
    recs = [(p, json.loads(p.read_text(encoding="utf-8"))) for p in records_for(experiment, root)]
    if split:
        recs = [(p, r) for p, r in recs if r.get("split") == split]
    if not recs:
        return [f"no run record for {experiment}"]
    path, latest = recs[-1]
    problems = [f"{path.name}: {m}" for m in check_record(latest, root, min_n)]
    tests = [r for _, r in recs if r.get("split") == "test"]
    for r in tests[1:]:
        if not (r.get("config") or {}).get("override_note"):
            problems.append(f"{r.get('run_id')}: more than one test record and no override_note in config")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--min-n", type=int, default=0)
    ap.add_argument("--split")
    a = ap.parse_args(argv)
    problems = check(a.experiment, min_n=a.min_n, split=a.split)
    for p in problems:
        print("FAIL", p)
    if not problems:
        print(f"{a.experiment} ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
