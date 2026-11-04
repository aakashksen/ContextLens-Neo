import argparse
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from contextlens.data.scenarios import load_scenarios
from contextlens.eval import series
from contextlens.eval.check_cache import PassCache, code_hash, record_key
from contextlens.eval.records import IDS_ONLY_SOURCES, LABEL_SOURCE, ROOT, RUNS_REL, SCENARIO_SOURCES, SUBSETS_REL, git, split_hash
from contextlens.eval.series import JUDGED, base_metrics, base_name
from contextlens.replay.runner import Prediction
from contextlens.schemas import Scenario
from contextlens.train_guard import non_finite_epochs

REQUIRED = (
    "run_id experiment git_commit dirty dataset_hash split scenario_source config models prompt_versions n metrics "
    "intervals comparisons attempts claude_calls labels_reviewed predictions started_at finished_at"
).split()
TOLERANCE = 1e-9
K_DEFAULT = (1, 3, 5, 10)
FAKE = re.compile(r"fake|stub", re.I)
TIMESTAMP = re.compile(r"\d{8}T\d{12}Z")
SUBSET_NAME = re.compile(r"[A-Za-z0-9_-]+")
CRLF, LF = chr(13) + chr(10), chr(10)


def recompute_metrics(
    scenarios: list[Scenario], predictions: list[Prediction], ks: Iterable[int] = K_DEFAULT
) -> dict[str, float | None]:
    """Every metric an experiment may save, recomputed from the saved rows with the functions in
    metrics.py. Also gives each metric restricted to one scenario category or origin as
    `name[category=x]` / `name[origin=y]`."""
    ks = tuple(ks)
    rows = _rows(scenarios, predictions)
    out = base_metrics(rows, ks)
    for key in ("category", "origin"):
        for value in sorted({getattr(s, key) for s, _ in rows}):
            sub = base_metrics([(s, p) for s, p in rows if getattr(s, key) == value], ks)
            out.update({f"{name}[{key}={value}]": v for name, v in sub.items()})
    return out


def _rows(scenarios: list[Scenario], predictions: list[Prediction]) -> list[tuple[Scenario, Prediction]]:
    by_id = {p.scenario_id: p for p in predictions}
    return [(s, by_id[s.id]) for s in scenarios if s.id in by_id]


def recompute_intervals(
    scenarios: list[Scenario], predictions: list[Prediction], names: Iterable[str]
) -> dict[str, tuple[float, float] | None]:
    """Bootstrap interval per metric name (seed 13, 10,000 resamples, by conversation); None when
    the interval cannot be recomputed."""
    rows = _rows(scenarios, predictions)
    return {n: series.interval(rows, n) for n in names}


def recompute_comparison(
    scenarios: list[Scenario], a: list[Prediction], b: list[Prediction], metric: str
) -> dict[str, object] | None:
    """System a minus system b on one metric, as saved in a record's `comparisons`."""
    return series.comparison(_rows(scenarios, a), _rows(scenarios, b), metric)


def _same_value(fresh: float | None, saved: object) -> bool:
    if fresh is None or saved is None:
        return fresh is None and saved is None
    return isinstance(saved, (int, float)) and abs(fresh - saved) <= TOLERANCE


def _same_pair(fresh: tuple[float, float] | list[float], saved: object) -> bool:
    return (isinstance(saved, (list, tuple)) and len(saved) == len(fresh)
            and all(_same_value(f, v) for f, v in zip(fresh, saved, strict=True)))


def _strings(x: Any) -> list[str]:
    if isinstance(x, str):
        return [x]
    if isinstance(x, dict):
        return [s for v in x.values() for s in _strings(v)]
    if isinstance(x, list):
        return [s for v in x for s in _strings(v)]
    return []


def _timestamp(name: str) -> str:
    found = TIMESTAMP.findall(name)
    return found[-1] if found else name


def records_for(experiment: str, root: Path = ROOT) -> list[Path]:
    """Every record of the experiment, oldest first."""
    found = (root / RUNS_REL).glob(f"{experiment}-*Z.json")
    return sorted(found, key=lambda p: (_timestamp(p.name), p.name))


def judge_problems(config: Any) -> list[str]:
    """A record that saves a judged metric must say who judged it."""
    judge = config.get("judge") if isinstance(config, dict) else None
    if not isinstance(judge, dict):
        return ["config.judge is missing: a record with grounded_rate or unsupported_claim_rate must name "
                "its judge (model, rubric_version, same_model_and_prompt_as_writer)"]
    out = []
    for key in ("model", "rubric_version"):
        if not (isinstance(judge.get(key), str) and judge[key].strip()):
            out.append(f"config.judge.{key} is missing")
    if isinstance(judge.get("model"), str) and FAKE.search(judge["model"]):
        out.append("the judge model matches fake or stub")
    if not isinstance(judge.get("same_model_and_prompt_as_writer"), bool):
        out.append("config.judge.same_model_and_prompt_as_writer must say true or false")
    return out


def review_problems(config: Any) -> list[str]:
    """A claim on a judged metric needs a person's check of the judgements on a random sample."""
    review = config.get("judge_review") if isinstance(config, dict) else None
    if not isinstance(review, dict):
        return ["config.judge_review is missing: no recorded person review of the judgements"]
    out = []
    rate, size = review.get("agreement_rate"), review.get("sample_size")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 0 <= rate <= 1:
        out.append("config.judge_review.agreement_rate must be a number from 0 to 1")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        out.append("config.judge_review.sample_size must be a positive integer")
    if review.get("sample") != "random":
        out.append("config.judge_review.sample must be 'random'")
    return out


def _row_problems(scenarios: list[Scenario], preds: list[Prediction]) -> list[str]:
    """Rows must be all cards or all retrieval, and may cite only turns at or before as_of_seq."""
    out = []
    if len({p.card is not None for p in preds}) > 1:
        out.append("rows are mixed: some have cards and some do not")
    by_id = {s.id: s for s in scenarios}
    for p in preds:
        s = by_id[p.scenario_id]
        seq = {t.id: t.seq for t in s.conversation.turns}
        cited = list(p.ranked_turn_ids or [])
        cited += [e.turn_id for f in (p.card.findings if p.card else []) for e in f.evidence]
        bad = [t for t in cited if t not in seq or seq[t] > s.request.as_of_seq]
        if bad:
            out.append(f"{p.scenario_id} cites turns that are unknown or past as_of_seq: {bad[:3]}")
    return out


def committed_unchanged(rec: dict[str, Any], root: Path, rel: str) -> bool:
    """The file is the same at the record's git_commit as on disk, so it was fixed before the run."""
    path = root / rel
    shown = git(root, "show", f"{rec['git_commit']}:{rel}")
    return (path.is_file() and shown.returncode == 0
            and shown.stdout.replace(CRLF, LF) == path.read_text(encoding="utf-8").replace(CRLF, LF))


def subset_problems(rec: dict[str, Any], root: Path, split_ids: list[str]) -> tuple[set[str] | None, list[str]]:
    """The ids of the record's named subset, or None with the problems when the subset can't be trusted.
    The manifest must be the same at the record's git_commit as on disk, so it was fixed before the run."""
    name = rec["subset"]
    if not (isinstance(name, str) and SUBSET_NAME.fullmatch(name)):
        return None, [f"subset {name!r} is not a valid subset name"]
    rel = (SUBSETS_REL / f"{name}.json").as_posix()
    path = root / rel
    if not path.is_file():
        return None, [f"subset {name} has no manifest at {rel}"]
    text = path.read_text(encoding="utf-8")
    if not committed_unchanged(rec, root, rel):
        return None, [f"subset {name} manifest was not committed before the run (absent or different at git_commit)"]
    try:
        manifest = json.loads(text)
        ids, split = manifest["ids"], manifest["split"]
        ok = isinstance(ids, list) and all(isinstance(i, str) for i in ids)
    except (ValueError, KeyError, TypeError):
        ok = False
    if not ok:
        return None, [f"subset {name} manifest needs split and a list of ids"]
    if split != rec["split"]:
        return None, [f"subset {name} belongs to split {split}, not {rec['split']}"]
    if not ids:
        return None, [f"subset {name} has no ids"]
    if len(set(ids)) != len(ids):
        return None, [f"subset {name} repeats ids"]
    stray = sorted(set(ids) - set(split_ids))
    if stray:
        return None, [f"subset {name} has ids not in the split: {stray[:3]}"]
    return set(ids), []


def training_loss_problems(rec: dict[str, Any]) -> list[str]:
    """A record whose chosen model (config.winner) has a missing, NaN or infinite training loss in its history."""
    config, attempts = rec.get("config"), rec.get("attempts")
    winner = config.get("winner") if isinstance(config, dict) else None
    if winner is None or not isinstance(attempts, list):
        return []
    return [f"the chosen model {winner} has a non-finite training loss in epoch {e}"
            for a in attempts if isinstance(a, dict) and a.get("setting") == winner and a.get("kind") == "training"
            for e in non_finite_epochs(a.get("history") or [])]


def _ancestor_problems(rec: dict[str, Any], root: Path) -> list[str]:
    ok = git(root, "merge-base", "--is-ancestor", rec["git_commit"], "HEAD").returncode == 0
    return [] if ok else ["git_commit is not an ancestor of HEAD"]


def _header_problems(rec: dict[str, Any], root: Path, min_n: int) -> list[str]:
    problems = []
    if rec["dirty"]:
        problems.append("record was made on a dirty tree")
    if any(FAKE.search(s) for s in _strings(rec["models"])):
        problems.append("a model name matches fake or stub")
    if not rec["attempts"]:
        problems.append("attempts is empty")
    problems += training_loss_problems(rec)
    if "label_source" in rec and rec["label_source"] != LABEL_SOURCE.get(rec["scenario_source"]):  # older records have none
        problems.append(f"label_source {rec['label_source']!r} is not the one for scenario source {rec['scenario_source']!r}")
    config = rec["config"]
    if isinstance(config, dict) and "labels_reviewed" in config and config["labels_reviewed"] != rec["labels_reviewed"]:
        problems.append(f"config.labels_reviewed ({config['labels_reviewed']!r}) and labels_reviewed "
                        f"({rec['labels_reviewed']!r}) disagree; the top-level flag is the one write_run sets")
    problems += _ancestor_problems(rec, root)
    if min_n and rec["n"] < min_n:
        problems.append(f"n={rec['n']} is below --min-n {min_n}")
    return problems


def _manifest_entry(rec: dict[str, Any], root: Path) -> tuple[dict[str, Any] | None, list[str]]:
    """The record's split entry in its source's manifest, with problems finding it or trusting the manifest."""
    problems = []
    split, source = rec["split"], rec["scenario_source"]
    manifest_rel = SCENARIO_SOURCES[source][0]
    entry = None
    if (root / manifest_rel).is_file():
        manifest = json.loads((root / manifest_rel).read_text(encoding="utf-8"))
        if manifest.get("source", "scenarios") != source:
            problems.append(f"the manifest at {manifest_rel.as_posix()} is not for scenario source {source}")
        else:
            entry = manifest.get("splits", {}).get(split)
    if source != "scenarios" and not committed_unchanged(rec, root, manifest_rel.as_posix()):
        problems.append(f"the {source} manifest was not committed before the run (absent or different at git_commit)")
    return entry, problems


def _scenario_problems(rec: dict[str, Any], root: Path) -> tuple[list[str], list[Scenario]]:
    """Problems with the record's split against its committed manifest, and the scenarios it was scored on
    (empty when they cannot be trusted)."""
    scenarios: list[Scenario] = []
    split, source = rec["split"], rec["scenario_source"]
    scenario_dir = SCENARIO_SOURCES[source][1]
    local = source != "scenarios"
    entry, problems = _manifest_entry(rec, root)
    if entry is None:
        return problems + [f"no manifest entry for split {split} of source {source}"], scenarios
    if not (root / scenario_dir / f"{split}.jsonl").is_file():
        return problems + [f"no scenario file for split {split} of source {source}"], scenarios
    if rec["dataset_hash"] != entry["hash"]:
        problems.append("dataset_hash does not match the split manifest")
    if split_hash(split, root, source) != entry["hash"]:
        problems.append("scenario file no longer matches the split manifest")
    else:
        scenarios = load_scenarios(split, root / scenario_dir)
        if local and [s.id for s in scenarios] != entry.get("ids"):
            problems.append("scenario ids do not match the manifest's ids")
            scenarios = []
    expected_n, what = entry["n"], "split"
    if "subset" in rec:
        ids, bad = subset_problems(rec, root, [s.id for s in scenarios])
        problems += bad
        if ids is None:
            scenarios = []
        else:
            scenarios = [s for s in scenarios if s.id in ids]
            expected_n, what = len(ids), "subset"
    if rec["n"] != expected_n:
        problems.append(f"n={rec['n']} differs from the {what}'s {expected_n} cases")
    return problems, scenarios


def _read_predictions(pred_path: Path, tag: str) -> tuple[list[Prediction] | None, list[str]]:
    if not pred_path.is_file():
        return None, [tag + "predictions file is missing"]
    try:
        return [Prediction.model_validate_json(x)
                for x in pred_path.read_text(encoding="utf-8").splitlines() if x.strip()], []
    except ValueError as exc:
        return None, [tag + f"predictions file is unreadable: {str(exc).splitlines()[0]}"]


def _recompute_problems(
    tag: str, scenarios: list[Scenario], preds: list[Prediction], rows: list[tuple[Scenario, Prediction]],
    metrics: dict[str, Any], intervals: dict[str, Any],
) -> list[str]:
    problems = []
    wanted = {int(k) for m in metrics for k in re.findall(r"recall@(\d+)", m)}
    fresh = recompute_metrics(scenarios, preds, K_DEFAULT + tuple(sorted(wanted)))
    for metric, saved in metrics.items():
        if metric not in fresh:
            problems.append(tag + f"metric {metric} cannot be recomputed from predictions")
        elif not _same_value(fresh[metric], saved):
            problems.append(tag + f"metric {metric} does not recompute from predictions")
    for metric, saved in intervals.items():
        got = series.interval(rows, metric)
        if got is None:
            problems.append(tag + f"interval {metric} cannot be recomputed from predictions")
        elif not _same_pair(got, saved):
            problems.append(tag + f"interval {metric} does not recompute from predictions")
    return problems


def _system_problems(
    name: str, rel: str, rec: dict[str, Any], root: Path, scenarios: list[Scenario], intervals: dict[str, Any],
) -> tuple[list[str], list[tuple[Scenario, Prediction]] | None]:
    """Problems with one system's predictions file, and its rows when they are usable for later checks."""
    tag = f"system {name}: "
    source, metrics = rec["scenario_source"], rec["metrics"]
    preds, problems = _read_predictions(root / rel, tag)
    if preds is None:
        return problems, None
    if len(preds) != rec["n"]:
        problems.append(tag + f"predictions hold {len(preds)} rows but n={rec['n']}")
    if not metrics[name]:
        problems.append(tag + "metrics is empty")
    if source in IDS_ONLY_SOURCES and any(p.card is not None or p.raw_output is not None or p.drops for p in preds):
        problems.append(tag + f"rows of source {source} may hold ids and metrics only, with no card, raw output or dropped-finding text")
    if "rerank" in name.lower() and not (isinstance(rec["models"], dict) and rec["models"].get("reranker")):
        problems.append(tag + "a rerank system must record its reranker model in models.reranker")
    if not scenarios:
        return problems, None
    pred_ids = [p.scenario_id for p in preds]
    if len(set(pred_ids)) != len(pred_ids) or set(pred_ids) != {s.id for s in scenarios}:
        return problems + [tag + "prediction ids do not match the split's scenario ids exactly once each"], None
    row_problems = _row_problems(scenarios, preds)
    problems += [tag + m for m in row_problems]
    if row_problems:
        return problems, None
    rows = _rows(scenarios, preds)
    problems += _recompute_problems(tag, scenarios, preds, rows, metrics[name], intervals.get(name) or {})
    return problems, rows


def _saves_judged_metric(metrics: dict[str, Any], intervals: dict[str, Any]) -> bool:
    """An undefined (None) judged metric makes no claim, so a retrieval-only record needs no judge for it."""
    defined = [m for d in metrics.values() for m, v in d.items() if v is not None] + [m for d in intervals.values() for m in d]
    return any(base_name(m) in JUDGED for m in defined)


def _shape_problems(rec: dict[str, Any]) -> tuple[list[str], dict[str, Any], bool]:
    """Problems with the shape of predictions, metrics, intervals and comparisons; the usable intervals; and
    whether the shape is too broken to check the systems at all."""
    paths, metrics, intervals = rec["predictions"], rec["metrics"], rec["intervals"]
    if not isinstance(paths, dict) or not paths:
        return ["predictions must map each system to its predictions file"], {}, True
    if not isinstance(metrics, dict) or set(metrics) != set(paths):
        return ["metrics must hold one entry per system, matching predictions"], {}, True
    problems = []
    if not isinstance(intervals, dict) or not set(intervals) <= set(paths):
        problems.append(f"intervals name systems not in predictions: {sorted(set(intervals) - set(paths))}")
        intervals = {}
    if not isinstance(rec["comparisons"], list):
        problems.append("comparisons must be a list")
    return problems, intervals, False


def check_record(rec: dict[str, Any], root: Path, min_n: int = 0) -> list[str]:
    problems = [f"missing field {k}" for k in REQUIRED if k not in rec]
    if problems:
        return problems
    problems += _header_problems(rec, root, min_n)
    source = rec["scenario_source"]
    if source not in SCENARIO_SOURCES:
        return problems + [f"unknown scenario source {source!r}; expected one of {sorted(SCENARIO_SOURCES)}"]
    more, scenarios = _scenario_problems(rec, root)
    problems += more

    shape, intervals, fatal = _shape_problems(rec)
    problems += shape
    if fatal:
        return problems
    paths, metrics = rec["predictions"], rec["metrics"]

    usable: dict[str, list[tuple[Scenario, Prediction]]] = {}
    for name, rel in paths.items():
        more, rows = _system_problems(name, rel, rec, root, scenarios, intervals)
        problems += more
        if rows is not None:
            usable[name] = rows

    for name in intervals:
        if name in paths and name not in usable and scenarios:
            problems.append(f"system {name}: its intervals cannot be checked without usable predictions")
    if isinstance(rec["comparisons"], list):
        problems += _comparison_problems(rec["comparisons"], usable, paths) if scenarios else []

    if _saves_judged_metric(metrics, intervals):
        problems += judge_problems(rec["config"])
    return problems


def _comparison_problems(
    comps: list[Any], usable: dict[str, list[tuple[Scenario, Prediction]]], paths: dict[str, str]
) -> list[str]:
    out = []
    for i, c in enumerate(comps):
        tag = f"comparison {i}: "
        if not (isinstance(c, dict) and {"a", "b", "metric", "diff", "ci", "mcnemar_p"} <= set(c)):
            out.append(tag + "needs a, b, metric, diff, ci and mcnemar_p")
            continue
        a, b = c["a"], c["b"]
        if a == b or a not in paths or b not in paths:
            out.append(tag + f"{a} and {b} must be two different systems of this record")
            continue
        if a not in usable or b not in usable:
            out.append(tag + "cannot be checked without usable predictions for both systems")
            continue
        fresh = series.comparison(usable[a], usable[b], c["metric"])
        if fresh is None:
            out.append(tag + f"{c['metric']} cannot be recomputed for {a} and {b}")
        elif not (_same_value(fresh["diff"], c["diff"])  # type: ignore[arg-type]
                  and _same_pair(fresh["ci"], c["ci"])  # type: ignore[arg-type]
                  and _same_value(fresh["mcnemar_p"], c["mcnemar_p"])):  # type: ignore[arg-type]
            out.append(tag + f"{c['metric']} for {a} vs {b} does not recompute from predictions")
    return out


def _load_records(experiment: str, root: Path) -> list[tuple[Path, dict[str, Any]]]:
    return [(p, json.loads(p.read_text(encoding="utf-8"))) for p in records_for(experiment, root)]


def superseded_by(experiment: str, root: Path = ROOT) -> dict[str, tuple[str, str]]:
    """run_id -> (run_id of the later record that supersedes it, reason). Only an entry with a
    non-empty reason, naming an earlier record of the same split, counts."""
    recs = _load_records(experiment, root)
    by_id = {r.get("run_id"): (p, r) for p, r in recs}
    out: dict[str, tuple[str, str]] = {}
    for p, r in recs:
        for e in r.get("supersedes") or []:
            target = by_id.get(e.get("run_id")) if isinstance(e, dict) else None
            reason = e.get("reason") if isinstance(e, dict) else None
            if (target and isinstance(reason, str) and reason.strip()
                    and target[1].get("split") == r.get("split")
                    and (_timestamp(target[0].name), target[0].name) < (_timestamp(p.name), p.name)):
                out[e["run_id"]] = (r["run_id"], reason)
    return out


def _supersede_problems(recs: list[tuple[Path, dict[str, Any]]], effective: dict[str, Any]) -> list[str]:
    out = []
    for p, r in recs:
        for e in r.get("supersedes") or []:
            rid = e.get("run_id") if isinstance(e, dict) else None
            if rid not in effective or effective[rid][0] != r.get("run_id"):
                out.append(f"{p.name}: supersedes entry {e!r} needs the run_id of an earlier record of the "
                           "same experiment and split, and a non-empty reason")
    return out


def check(
    experiment: str, *, root: Path = ROOT, min_n: int = 0, split: str | None = None, cache: PassCache | None = None
) -> list[str]:
    """Problems with the experiment's records. With a `cache`, a record whose key is stored as passed is not
    rechecked, except that its git_commit must still be an ancestor of HEAD; a record that passes is added."""
    recs = _load_records(experiment, root)
    if split:
        recs = [(p, r) for p, r in recs if r.get("split") == split]
    if not recs:
        return [f"no run record for {experiment}"]
    gone = superseded_by(experiment, root)
    problems = _supersede_problems(recs, gone)
    for path, rec in recs:
        if rec.get("run_id") in gone:
            continue
        key = record_key(path, rec, root, min_n, split, cache.code) if cache and "git_commit" in rec else None
        if key and cache and cache.hit(key):
            found = _ancestor_problems(rec, root)
        else:
            found = check_record(rec, root, min_n)
            if key and cache and not found:
                cache.add(key, path.name)
        problems += [f"{path.name}: {m}" for m in found]
    if any(r.get("split") == "test" for _, r in recs):
        problems += _test_once_problems(root)
    return problems


def _test_once_problems(root: Path) -> list[str]:
    """Every test-split record of every experiment counts: one record covers all systems, and
    renaming an experiment does not give another touch. Superseded records still count."""
    found = []
    for p in (root / RUNS_REL).glob("*-*Z.json"):
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("split") == "test":
            found.append((_timestamp(p.name), p.name, rec))
    found.sort(key=lambda t: t[:2])
    return [f"{rec.get('run_id')}: more than one test record and no override_note in config"
            for _, _, rec in found[1:] if not (rec.get("config") or {}).get("override_note")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--min-n", type=int, default=0)
    ap.add_argument("--split")
    ap.add_argument("--all", action="store_true", help="recheck every record, ignoring remembered passes")
    a = ap.parse_args(argv)
    cache = PassCache(ROOT, code_hash(), read=not a.all)
    problems = check(a.experiment, min_n=a.min_n, split=a.split, cache=cache)
    cache.save()
    for old, (new, reason) in superseded_by(a.experiment).items():
        print(f"note: {old} is superseded by {new}: {reason}")
    for p in problems:
        print("FAIL", p)
    if not problems:
        print(f"{a.experiment} ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
