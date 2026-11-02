import argparse
import json
import re
import sys
from pathlib import Path
from collections.abc import Iterable
from typing import Any

from contextlens.data.scenarios import load_scenarios
from contextlens.eval import metrics as M
from contextlens.eval.records import MANIFEST_REL, ROOT, RUNS_REL, git, split_hash
from contextlens.eval.stats import percentile
from contextlens.replay.runner import Prediction
from contextlens.schemas import Card, Scenario

REQUIRED = (
    "run_id experiment git_commit dirty dataset_hash split config models prompt_versions n metrics "
    "intervals attempts claude_calls labels_reviewed predictions_path started_at finished_at"
).split()
TOLERANCE = 1e-9


def _label(card: Card) -> str:
    for f in card.findings:
        if f.label:
            return f.label
    return "insufficient"


Metrics = dict[str, float | None]
K_DEFAULT = (1, 3, 5, 10)


def _base_metrics(rows: list[tuple[Scenario, Prediction]], ks: Iterable[int]) -> Metrics:
    carded = [(s, p, p.card) for s, p in rows if p.card is not None]
    labelled = [(s.gold_label, _label(c)) for s, _, c in carded if s.gold_label]
    gold, pred = [g for g, _ in labelled], [q for _, q in labelled]
    out: Metrics = {
        "macro_f1": M.macro_f1(gold, pred),
        "false_alert_rate": M.false_alert_rate(gold, pred),
        **M.abstention_accuracy(
            [c.status == "abstained" for _, _, c in carded],
            [s.gold_label == "insufficient" for s, _, _ in carded],
        ),
        "tokens_per_card": M.mean([float(c.tokens) for _, _, c in carded]),
        "cost_usd_per_card": M.mean([c.cost_usd for _, _, c in carded]),
    }

    pred_sp: list[str] = []
    gold_sp: list[str] = []
    for s, _, c in carded:
        speaker_of = {t.id: t.speaker for t in s.conversation.turns}
        for ev in (e for f in c.findings for e in f.evidence):
            if ev.turn_id in speaker_of:
                pred_sp.append(ev.speaker)
                gold_sp.append(speaker_of[ev.turn_id])
    out["speaker_accuracy"] = M.speaker_accuracy(pred_sp, gold_sp)

    # "supports the claim" is a saved rubric judgement; without one per finding the metric is omitted
    if all(p.claim_supported is not None and len(p.claim_supported) == len(c.findings)
           for _, p, c in carded if c.findings):
        per_card: list[list[bool]] = []
        for s, p, c in carded:
            judged = {id(f): j for f, j in zip(c.findings, p.claim_supported or [])}
            turn_seq = {t.id: t.seq for t in s.conversation.turns}
            per_card.append(M.grounded_flags(
                [c], turn_seq, s.request.as_of_seq, lambda f, judged=judged: judged[id(f)]))
        flat = [x for fl in per_card for x in fl]
        out["grounded_rate"] = sum(flat) / len(flat) if flat else None
        out["unsupported_claim_rate"] = M.unsupported_claim_rate(per_card)

    raws = [p.raw_output for _, p in rows if p.raw_output is not None]
    out["schema_valid_rate"] = M.mean([float(_schema_valid(r)) for r in raws])

    ranked = [(s, p.ranked_turn_ids) for s, p in rows
              if p.ranked_turn_ids is not None and s.gold_evidence_turn_ids]
    ranks = [M.first_gold_rank(r, s.gold_evidence_turn_ids) for s, r in ranked]
    out["mrr"] = M.mrr(ranks)
    for k in ks:
        out[f"recall@{k}"] = M.mean(
            [v for s, r in ranked if (v := M.recall_at_k(r, s.gold_evidence_turn_ids, k)) is not None])

    stages: dict[str, list[float]] = {}
    for _, p in rows:
        for name, ms in p.stage_latency_ms.items():
            stages.setdefault(name, []).append(float(ms))
    for name, vals in stages.items():
        suffix = "" if name == "total" else f"[stage={name}]"
        out[f"latency_p50_ms{suffix}"] = percentile(vals, 50)
        out[f"latency_p95_ms{suffix}"] = percentile(vals, 95)
    if "total" not in stages:
        out["latency_p50_ms"] = out["latency_p95_ms"] = None
    return out


def _schema_valid(raw: str) -> bool:
    try:
        Card.model_validate_json(raw)
    except ValueError:
        return False
    return True


def recompute_metrics(
    scenarios: list[Scenario], predictions: list[Prediction], ks: Iterable[int] = K_DEFAULT
) -> Metrics:
    """Every metric an experiment may save, recomputed from the saved rows with the functions in
    metrics.py. Also gives each metric restricted to one scenario category or origin as
    `name[category=x]` / `name[origin=y]`."""
    ks = tuple(ks)
    by_id = {p.scenario_id: p for p in predictions}
    rows = [(s, by_id[s.id]) for s in scenarios if s.id in by_id]
    out = _base_metrics(rows, ks)
    for key in ("category", "origin"):
        for value in sorted({getattr(s, key) for s, _ in rows}):
            sub = _base_metrics([(s, p) for s, p in rows if getattr(s, key) == value], ks)
            out.update({f"{name}[{key}={value}]": v for name, v in sub.items()})
    return out


def _same_value(fresh: float | None, saved: object) -> bool:
    if fresh is None or saved is None:
        return fresh is None and saved is None
    return isinstance(saved, (int, float)) and abs(fresh - saved) <= TOLERANCE


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
        ks = {int(k) for name in rec['metrics'] for k in re.findall(r'recall@(\d+)', name)}
        fresh = recompute_metrics(scenarios, preds, K_DEFAULT + tuple(sorted(ks)))
        for name, saved in rec["metrics"].items():
            if name not in fresh:
                problems.append(f"metric {name} cannot be recomputed from predictions")
            elif not _same_value(fresh[name], saved):
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
