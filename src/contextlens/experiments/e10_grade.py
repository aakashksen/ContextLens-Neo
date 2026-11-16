"""E10's correctness grading by Sonnet, in two offline steps. This module calls no API: the coordinator runs the grading
(Sonnet subagents read the batch files and write the graded files); this module only prepares and reads files.

  export   .venv/Scripts/python -m contextlens.experiments.e10_grade export --split test --out <dir>
           Picks about 200 cards: about 50 cases (whole conflict triplets and twin pairs, stratified by case type) times every arm,
           so the same case is graded for every arm and the grades can be compared in pairs. Arm identity is hidden: each card
           gets a random item id and the batches are shuffled. Writes batch-001.json ... (about 25 items each), the rubric path
           inside each, and key.json (item id -> arm and case; NEVER given to a grader).
  import   .venv/Scripts/python -m contextlens.experiments.e10_grade import --split test --dir <dir>
           Reads graded-batch-NNN.json files, checks them against the batches, and writes the correctness summary.

Batch file: {"batch": 1, "rubric": "docs/e10-grading-rubric.md", "items": [item, ...]}.
Item: {"item_id", "case_type", "question", "gold_answer", "gold_label", "gold_turns": [{"turn_id","speaker","text"}],
       "card": {"status","mode","suggestion","missing","findings":[{"kind","label","statement","evidence":[{"turn_id","quote"}]}]},
       "cited_turns": [{"turn_id","speaker","text"}]}   (the turns the card cites that are not gold, so support can be checked)
Graded file: a JSON list of {"item_id", "verdict": "correct" | "partial" | "wrong", "reason": "<one line>"}.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contextlens.data import e10_cases
from contextlens.eval.records import ROOT, write_atomic
from contextlens.eval.stats import bootstrap_ci, mcnemar_exact, paired_bootstrap_diff
from contextlens.experiments import e10_long as run
from contextlens.replay.runner import Prediction
from contextlens.schemas import Scenario

RUBRIC_REL = "docs/e10-grading-rubric.md"
VERDICTS = ("correct", "partial", "wrong")
VALUE = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}
SHARE = {"qmsum": 0.30, "conflict": 0.30, "twin": 0.26, "control": 0.14}  # of the cases; conflict and twin come whole
UNIT = {"qmsum": 1, "conflict": 3, "twin": 2, "control": 1}  # cases per sampled unit
MAX_CITED, TURN_CHARS = 12, 700


def _turn_dict(sc: Scenario, tid: str) -> dict[str, str] | None:
    t = next((t for t in sc.conversation.turns if t.id == tid), None)
    return None if t is None else {"turn_id": t.id, "speaker": t.speaker, "text": t.text[:TURN_CHARS]}


def item_for(item_id: str, sc: Scenario, p: Prediction, case_type: str) -> dict[str, Any]:
    card = p.card
    gold = [d for g in sc.gold_evidence_turn_ids if (d := _turn_dict(sc, g))]
    cited = []
    for f in (card.findings if card else []):
        for e in f.evidence:
            if e.turn_id not in sc.gold_evidence_turn_ids and e.turn_id not in {c["turn_id"] for c in cited}:
                d = _turn_dict(sc, e.turn_id)
                if d and len(cited) < MAX_CITED:
                    cited.append(d)
    body = None if card is None else {
        "status": card.status, "mode": card.mode, "suggestion": card.suggestion, "missing": card.missing,
        "findings": [{"kind": f.kind, "label": f.label, "statement": f.statement,
                      "evidence": [{"turn_id": e.turn_id, "quote": e.quote} for e in f.evidence]} for f in card.findings]}
    return {"item_id": item_id, "case_type": sc.category, "question": sc.request.question, "gold_answer": sc.gold_answer,
            "gold_label": sc.gold_label, "gold_turns": gold, "card": body, "cited_turns": cited}


def sample_cases(scs: list[Scenario], meta: dict[str, dict[str, Any]], n_cases: int, seed: int) -> list[Scenario]:
    """About n_cases cases: each type's share of them, in whole units (a triplet or a pair stays together)."""
    rng = random.Random(f"{seed}:grade")
    units: dict[str, dict[str, list[Scenario]]] = {t: defaultdict(list) for t in SHARE}
    for s in scs:
        m = meta[s.id]
        unit = f"{m['type']}:{m.get('triplet', m.get('pair', s.id))}"
        units[m["type"]][unit].append(s)
    out: list[Scenario] = []
    for t, share in SHARE.items():
        keys = sorted(units[t])
        rng.shuffle(keys)
        want = max(UNIT[t], round(n_cases * share / UNIT[t])) if keys else 0
        for k in keys[:want]:
            out.extend(units[t][k])
    return out


def export(root: Path, split: str, out_dir: Path, *, n_cards: int = 200, batch_size: int = 25, seed: int = 13,
           commit: str | None = None, config: Path | None = None) -> dict[str, Any]:
    cfg = run.load_config(root, config)
    arms = cfg["arms"]
    scs, meta, digest = e10_cases.load_cases(split, root)
    rows = run.saved_rows(root, run.run_key(root, cfg, commit), split)
    have = [a for a in arms if rows.get(a)]
    if len(have) < len(arms):
        raise RuntimeError(f"saved rows exist for {have} only; the arms are {arms}. Run E10 first.")
    complete = [s for s in scs if all(s.id in rows[a] for a in arms)]
    chosen = sample_cases(complete, meta, n_cards // len(arms), seed)
    by_id = {s.id: s for s in chosen}
    rng = random.Random(f"{seed}:order")
    cards = [(s, a) for s in chosen for a in arms]
    rng.shuffle(cards)
    ids = rng.sample(range(10_000, 100_000), len(cards))
    key, items = {}, []
    for (s, a), n in zip(cards, ids, strict=True):
        item_id = f"g{n}"
        key[item_id] = {"split": split, "case": s.id, "arm": a, "type": meta[s.id]["type"]}
        items.append(item_for(item_id, by_id[s.id], rows[a][s.id], s.category))
    out_dir.mkdir(parents=True, exist_ok=True)
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    for i, b in enumerate(batches, 1):
        write_atomic(out_dir / f"batch-{i:03d}.json", json.dumps({"batch": i, "rubric": RUBRIC_REL, "items": b}, indent=1) + "\n")
    write_atomic(out_dir / "key.json", json.dumps({"split": split, "experiment": cfg.get("experiment", "E10"), "cases_hash": digest, "rows_key": run.run_key(root, cfg, commit), "key": key}, indent=1) + "\n")
    return {"cases": len(chosen), "cards": len(items), "batches": len(batches), "dir": str(out_dir)}


def read_graded(dir_: Path, key: dict[str, Any]) -> dict[str, dict[str, str]]:
    """item id -> {verdict, reason}. Every item of every batch must be graded exactly once with a valid verdict and a reason."""
    graded: dict[str, dict[str, str]] = {}
    for f in sorted(dir_.glob("graded-batch-*.json")):
        for e in json.loads(f.read_text(encoding="utf-8")):
            iid, verdict, reason = e.get("item_id"), e.get("verdict"), (e.get("reason") or "").strip()
            if iid not in key:
                raise ValueError(f"{f.name}: unknown item {iid!r}")
            if verdict not in VERDICTS:
                raise ValueError(f"{f.name}: {iid} has verdict {verdict!r}; expected one of {VERDICTS}")
            if not reason:
                raise ValueError(f"{f.name}: {iid} has no reason")
            if iid in graded:
                raise ValueError(f"{f.name}: {iid} is graded twice")
            graded[iid] = {"verdict": verdict, "reason": reason.splitlines()[0][:300]}
    expected = {i["item_id"] for f in sorted(dir_.glob("batch-*.json")) for i in json.loads(f.read_text(encoding="utf-8"))["items"]}
    if missing := sorted(expected - set(graded)):
        raise ValueError(f"{len(missing)} items have no grade, for example {missing[:3]}")
    return graded


def summarize_grades(key: dict[str, dict[str, str]], graded: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Per arm and per case type: strict (correct only) and lenient (correct or partial) rates with 95% intervals over meetings,
    and paired McNemar on strict correctness between arms, on the cases every arm was graded on."""
    by_arm: dict[str, dict[str, Any]] = defaultdict(dict)
    for iid, g in graded.items():
        k = key[iid]
        by_arm[k["arm"]][k["case"]] = {"verdict": g["verdict"], "type": k["type"], "group": k.get("group", k["case"])}
    out: dict[str, Any] = {"n_graded": len(graded), "arms": {}, "comparisons": []}
    for arm, cases in sorted(by_arm.items()):
        entry: dict[str, Any] = {}
        for label, sel in (("all", lambda c: True), *((t, lambda c, t=t: c["type"] == t) for t in SHARE)):
            rows = [c for c in cases.values() if sel(c)]
            if not rows:
                continue
            strict = [float(c["verdict"] == "correct") for c in rows]
            lenient = [VALUE[c["verdict"]] > 0 and 1.0 or 0.0 for c in rows]
            g = [c["group"] for c in rows]
            entry[label] = {"n": len(rows), "strict": sum(strict) / len(rows), "strict_ci": list(bootstrap_ci(strict, g, n=2000)),
                            "lenient": sum(lenient) / len(rows), "verdicts": {v: sum(c["verdict"] == v for c in rows) for v in VERDICTS}}
        out["arms"][arm] = entry
    arms = sorted(by_arm)
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            ids = sorted(set(by_arm[a]) & set(by_arm[b]))
            if not ids:
                continue
            xa = [by_arm[a][c]["verdict"] == "correct" for c in ids]
            xb = [by_arm[b][c]["verdict"] == "correct" for c in ids]
            diff, lo, hi = paired_bootstrap_diff([float(v) for v in xa], [float(v) for v in xb], [by_arm[a][c]["group"] for c in ids], n=5000)
            out["comparisons"].append({"a": a, "b": b, "metric": "graded_correct_strict", "n": len(ids), "diff": diff, "ci": [lo, hi],
                                       "mcnemar_p": mcnemar_exact(xa, xb), "a_only": sum(p and not q for p, q in zip(xa, xb)),
                                       "b_only": sum(q and not p for p, q in zip(xa, xb))})
    return out


def fold_in(root: Path, split: str, dir_: Path) -> Path:
    """Read the graded files and write the summary (counts and rates, no meeting text) beside the code-scored records."""
    meta_file = json.loads((dir_ / "key.json").read_text(encoding="utf-8"))
    cases = json.loads((root / e10_cases.MANIFEST_REL).read_text(encoding="utf-8"))["cases"]
    key = {i: {**v, "group": cases[v["case"]]["meeting"]} for i, v in meta_file["key"].items()}
    graded = read_graded(dir_, key)
    now = datetime.now(timezone.utc)
    exp = meta_file.get("experiment", "E10")
    run_id = f"{exp}G-{split}-{now.strftime('%Y%m%dT%H%M%S%fZ')}"
    record = {"run_id": run_id, "experiment": exp, "kind": "sonnet-grades", "split": split, "grader": "sonnet",
              "cases_hash": meta_file["cases_hash"], "rows_key": meta_file["rows_key"], "rubric": RUBRIC_REL,
              "arm_blind": True, "labels_reviewed": False, **summarize_grades(key, graded), "finished_at": now.isoformat()}
    (root / run.LOCAL_DIR).mkdir(parents=True, exist_ok=True)
    (root / run.RECORD_DIR).mkdir(parents=True, exist_ok=True)
    # the one-line reasons can quote meeting text, so they stay local
    write_atomic(root / run.LOCAL_DIR / f"{run_id}.reasons.json", json.dumps({i: {**graded[i], **meta_file["key"][i]} for i in graded}, indent=1) + "\n")
    path = root / run.RECORD_DIR / f"{run_id}.json"
    write_atomic(path, json.dumps(record, indent=1) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["export", "import"])
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--dir", "--out", dest="dir", type=Path, required=True)
    ap.add_argument("--cards", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--commit", help="export: the commit whose saved rows to sample (default: HEAD)")
    ap.add_argument("--config", type=Path, help="export: config under the repo root; default configs/e10.json (E11 is configs/e11.json)")
    args = ap.parse_args(argv)
    if args.command == "export":
        print(json.dumps(export(ROOT, args.split, args.dir, n_cards=args.cards, batch_size=args.batch_size, commit=args.commit,
                                config=args.config)))
    else:
        print(fold_in(ROOT, args.split, args.dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
