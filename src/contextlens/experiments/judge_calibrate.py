"""Calibration of the cheaper grader (docs/DECISIONS.md, entry 53): does it agree with Sonnet at least as well as the
local judge did, and may it take claims about 20 at a time?

    python -m contextlens.experiments.judge_calibrate --sample 300 --concurrency 8 --mode single
    python -m contextlens.experiments.judge_calibrate --sample 300 --concurrency 8 --mode batch --max-weekly-pct 90

The pool is every claim of the two JC records that hold both the local judge's grade and Sonnet's (old and new sets, the
systems `single` and `team_changed`). In their saved rows `claim_supported` is the LOCAL verdict and `cloud_supported` is
SONNET's; `load_pool` checks both against the shares the records report (`grounded_rate` and `cloud_grounded_rate`).
Both modes grade the identical seeded sample. Answers are saved as they arrive under results/partial/judge-calibrate/
(ids and verdicts only, never text), so a stopped run resumes. The result is a provisional JSON under results/calibration/
(ids, verdicts and counts only). Sonnet is a reference, not ground truth."""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contextlens.eval.metrics import cohen_kappa
from contextlens.eval.records import ROOT, head_commit, is_dirty
from contextlens.eval.stats import bootstrap_ci, paired_bootstrap_diff, percentile
from contextlens.experiments import parallel_grader as pg
from contextlens.experiments.e2_context import RUBRIC_VERSION
from contextlens.experiments.judge_check import Refused, preflight
from contextlens.llm.client import CLAUDE_CALL_CAP, CLAUDE_CALLS_PATH
from contextlens.replay.runner import Prediction

JC_RECORDS = ("JC-20261104T144503532496Z", "JC-20261104T144600470844Z")
SEED = 20261104  # fixed here, in the committed code, before any run
SAMPLE_DEFAULT = 300
RESAMPLES = 5000
RATE_POINT = 0.01  # entry 53: a grade rate that moves by a point or more keeps one claim per call
SESSIONS_DIR = Path.home() / ".claude" / "hud" / "state" / "sessions"
USAGE_FILES = 5  # the newest few session files are read for the weekly percentage
USAGE_CHECK_S = 30.0
PARTIAL_REL = Path("results/partial/judge-calibrate")
OUT_REL = Path("results/calibration")
SAMPLING_RULE = (
    "Pool: every claim in the two JC records with both a local and a Sonnet grade. Strata: set (old, new) by system "
    "(single, team_changed). Each stratum gets a share of the sample in proportion to its size (largest remainder); inside "
    "a stratum claims are drawn at random with a fixed seed. The sample is graded in a seeded random order, so a run that "
    "stops early has graded a random part of it."
)
CLEARLY_WORSE = "clearly worse"
NOT_CLEARLY_WORSE = "not clearly worse"


class Stopped(RuntimeError):
    """The run ended early on purpose (a limit); the saved rows are kept and a rerun resumes. No record is written."""

    def __init__(self, reason: str, usage_start_pct: float | None, usage_end_pct: float | None) -> None:
        super().__init__(reason)
        self.usage_start_pct, self.usage_end_pct = usage_start_pct, usage_end_pct


class UsageUnreadable(RuntimeError):
    pass


@dataclass(frozen=True)
class CalClaim:
    id: str
    set: str
    system: str
    scenario_id: str
    local: bool  # the local judge's verdict: `claim_supported` in the saved rows
    sonnet: bool  # Sonnet's verdict: `cloud_supported` in the saved rows
    item: pg.Item


@dataclass(frozen=True)
class Sample:
    claims: list[CalClaim]  # in the order they are graded
    sample_id: str
    strata: dict[str, dict[str, int]]
    seed: int
    n: int


# ---------------------------------------------------------------- pool and sample

def load_pool(root: Path = ROOT, records: tuple[str, ...] | None = None) -> list[CalClaim]:
    pool: list[CalClaim] = []
    for rid in records or JC_RECORDS:
        path = root / "results/runs" / f"{rid}.json"
        if not path.is_file():
            raise Refused(f"no run record {rid}")
        rec = json.loads(path.read_text(encoding="utf-8"))
        set_name = rec["config"]["set"]
        for system, rel in rec["predictions"].items():
            rows = [Prediction.model_validate_json(x) for x in (root / rel).read_text(encoding="utf-8").splitlines() if x.strip()]
            mine: list[CalClaim] = []
            for p in rows:
                if p.card is None or not p.card.findings:
                    continue
                n = len(p.card.findings)
                if p.claim_supported is None or p.cloud_supported is None or len(p.claim_supported) != n or len(p.cloud_supported) != n:
                    raise Refused(f"{rid} {system} {p.scenario_id}: the row does not hold a local and a Sonnet verdict per claim")
                for i, (f, loc, son) in enumerate(zip(p.card.findings, p.claim_supported, p.cloud_supported, strict=True)):
                    cid = f"{rid}/{system}/{p.scenario_id}/{i}"
                    mine.append(CalClaim(cid, set_name, system, p.scenario_id, loc, son,
                                         pg.Item(cid, f.statement, tuple((e.speaker, e.quote) for e in f.evidence))))
            for field_name, metric, getter in (("claim_supported", "grounded_rate", lambda c: c.local),
                                               ("cloud_supported", "cloud_grounded_rate", lambda c: c.sonnet)):
                share = sum(getter(c) for c in mine) / len(mine)
                if abs(share - rec["metrics"][system][metric]) > 1e-9:
                    raise Refused(f"{rid} {system}: the share of {field_name} is {share:.4f} but the record's {metric} is "
                                  f"{rec['metrics'][system][metric]:.4f}, so the field does not mean what this code assumes")
            pool += mine
    return pool


def draw_sample(pool: list[CalClaim], n: int, seed: int = SEED) -> Sample:
    """`n` claims, proportional to stratum size (largest remainder), random inside a stratum, graded in random order."""
    rng = random.Random(seed)
    strata: dict[str, list[CalClaim]] = defaultdict(list)
    for c in sorted(pool, key=lambda c: c.id):
        strata[f"{c.set}/{c.system}"].append(c)
    n = min(n, len(pool))
    quota = {k: n * len(v) // len(pool) for k, v in strata.items()}
    for k in sorted(strata, key=lambda k: (-(n * len(strata[k]) % len(pool)), k))[: n - sum(quota.values())]:
        quota[k] += 1
    chosen: list[CalClaim] = []
    for k in sorted(strata):
        chosen += rng.sample(strata[k], quota[k])
    rng.shuffle(chosen)
    sample_id = hashlib.sha256(json.dumps([seed, n, [c.id for c in chosen]]).encode()).hexdigest()[:12]
    return Sample(chosen, sample_id, {k: {"pool": len(strata[k]), "taken": quota[k]} for k in sorted(strata)}, seed, n)


# ---------------------------------------------------------------- weekly usage

def usage_percent(sessions: Path | None = None, newest: int = USAGE_FILES) -> float:
    """The highest seven-day used percentage in the newest few session files. Read only."""
    folder = SESSIONS_DIR if sessions is None else Path(sessions)
    try:
        files = sorted((p for p in folder.glob("*.json") if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True)[:newest]
    except OSError as err:
        raise UsageUnreadable(f"cannot list {folder}: {err}") from err
    seen: list[float] = []
    for p in files:
        try:
            v = json.loads(p.read_text(encoding="utf-8"))["data"]["rate_limits"]["seven_day"]["used_percentage"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if isinstance(v, int | float) and not isinstance(v, bool):
            seen.append(float(v))
    if not seen:
        raise UsageUnreadable(f"no seven-day used_percentage in the newest {newest} files of {folder}")
    return max(seen)


class UsageGuard:
    def __init__(self, limit: float, *, read: Callable[[], float] | None = None, now: Callable[[], float] = time.monotonic,
                 interval_s: float = USAGE_CHECK_S) -> None:
        self.limit, self.read, self.now, self.interval_s = limit, read or (lambda: usage_percent(SESSIONS_DIR)), now, interval_s
        self.first: float | None = None
        self.last: float | None = None
        self._at = 0.0

    def start(self) -> float:
        self.first = self.last = self.read()
        self._at = self.now()
        if self.first >= self.limit:
            raise Refused(f"weekly usage is already {self.first:g}%, at or over the limit of {self.limit:g}%")
        return self.first

    def check(self) -> str | None:
        """A reason to stop, or None. Reads the files at most once per interval."""
        if self.now() - self._at < self.interval_s:
            return None
        self._at = self.now()
        try:
            self.last = self.read()
        except UsageUnreadable as exc:
            return f"weekly usage can no longer be read ({exc})"
        if self.last >= self.limit:
            return f"weekly usage {self.last:g}% reached the limit of {self.limit:g}%"
        return None

    def final(self) -> float | None:
        try:
            self.last = self.read()
        except UsageUnreadable:
            pass
        return self.last


# ---------------------------------------------------------------- statistics

def _kappa_ci(a: list[bool], b: list[bool], groups: list[str], n: int, seed: int = 13) -> tuple[float, float] | None:
    """95% interval of Cohen's kappa, resampling whole cases. Resamples where kappa is undefined are dropped."""
    by: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        by[g].append(i)
    buckets = list(by.values())
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        idx = [i for _ in buckets for i in buckets[rng.randrange(len(buckets))]]
        k = cohen_kappa([a[i] for i in idx], [b[i] for i in idx])
        if k is not None:
            vals.append(k)
    return (percentile(vals, 2.5), percentile(vals, 97.5)) if len(vals) >= n // 2 else None


def _vs_sonnet(x: list[bool], son: list[bool], groups: list[str], resamples: int) -> dict[str, Any]:
    agree = [float(p == q) for p, q in zip(x, son, strict=True)]
    lo, hi = bootstrap_ci(agree, groups, n=resamples)
    return {
        "agreement": sum(agree) / len(agree), "agreement_ci": [lo, hi], "kappa": cohen_kappa(x, son),
        "kappa_ci": _kappa_ci(x, son, groups, resamples),
        "disagree": {"grader_unsupported_sonnet_supported": sum((not p) and q for p, q in zip(x, son, strict=True)),
                     "grader_supported_sonnet_unsupported": sum(p and not q for p, q in zip(x, son, strict=True))},
    }


def _one_group(claims: list[CalClaim], haiku: dict[str, bool], resamples: int) -> dict[str, Any]:
    graded = [c for c in claims if c.id in haiku]
    if not graded:
        return {"n": 0, "verdict": "undetermined"}
    h, loc, son = [haiku[c.id] for c in graded], [c.local for c in graded], [c.sonnet for c in graded]
    groups = [c.scenario_id for c in graded]
    h_agree, l_agree = [float(p == q) for p, q in zip(h, son, strict=True)], [float(p == q) for p, q in zip(loc, son, strict=True)]
    mean, lo, hi = paired_bootstrap_diff(h_agree, l_agree, groups, n=resamples)
    n = len(graded)
    return {
        "n": n, "haiku_vs_sonnet": _vs_sonnet(h, son, groups, resamples), "local_vs_sonnet": _vs_sonnet(loc, son, groups, resamples),
        "agreement_difference": {"what": "haiku agreement minus local agreement, each with Sonnet", "mean": mean, "lo": lo, "hi": hi},
        "share_unsupported": {"sonnet": son.count(False) / n, "haiku": h.count(False) / n, "local": loc.count(False) / n},
        "verdict": CLEARLY_WORSE if hi < 0 else NOT_CLEARLY_WORSE,
    }


def analyse(claims: list[CalClaim], haiku: dict[str, bool], resamples: int = RESAMPLES) -> dict[str, Any]:
    """Grader against Sonnet and local judge against Sonnet on the same claims (those the grader graded), overall and per
    set. 'Clearly worse' only if the interval of (grader agreement - local agreement) lies wholly below zero."""
    return {"all": _one_group(claims, haiku, resamples),
            **{s: _one_group([c for c in claims if c.set == s], haiku, resamples) for s in ("old", "new")}}


_ISSUES = {"missing", "repeated", "unparsed", "call_failed"}


def batching_report(ids: list[str], single: dict[str, dict], batch: dict[str, dict], unknown_ids: int) -> dict[str, Any]:
    """Entry 53: batches are used only if their grades match the one-per-call grades on the same claims. Any lost or
    misaligned claim, or a supported-share that differs by a point or more, keeps one claim per call."""
    lost = sorted(i for i in ids if batch.get(i, {}).get("batch_issue") in _ISSUES)
    both = [i for i in ids if single.get(i, {}).get("status") == "ok" and batch.get(i, {}).get("status") == "ok"
            and batch[i].get("via") == "batch"]
    s, b = [single[i]["cloud"] for i in both], [batch[i]["cloud"] for i in both]
    rate_diff = round(sum(b) / len(b) - sum(s) / len(s), 9) if both else None
    mismatches = sum(p != q for p, q in zip(s, b, strict=True))
    complete = all(single.get(i, {}).get("status") == "ok" and batch.get(i, {}).get("status") == "ok" for i in ids)
    if lost or unknown_ids or (rate_diff is not None and abs(rate_diff) >= RATE_POINT):
        verdict = "keep one per call"
    else:
        verdict = "batching allowed" if complete and both else "not determined"
    return {
        "rule": "entry 53: keep one claim per call on any lost or misaligned claim, or if the share graded supported "
                "changes by a point or more; otherwise batches of about 20 are allowed",
        "compared": len(both), "match_rate": (1 - mismatches / len(both)) if both else None, "mismatches": mismatches,
        "single_supported_rate": sum(s) / len(s) if s else None, "batch_supported_rate": sum(b) / len(b) if b else None,
        "rate_difference": rate_diff, "lost_or_misaligned": len(lost), "lost_or_misaligned_ids": lost,
        "unknown_ids_in_replies": unknown_ids, "complete": complete, "verdict": verdict,
    }


# ---------------------------------------------------------------- the run

@dataclass(frozen=True)
class Args:
    sample: int
    concurrency: int
    mode: str
    max_weekly_pct: float
    use_guard: bool


def _meta_path(root: Path, sample_id: str) -> Path:
    return root / PARTIAL_REL / f"{sample_id}.batch.meta.json"


def _add_meta(root: Path, sample_id: str, unknown: int, calls: int) -> int:
    path = _meta_path(root, sample_id)
    old = json.loads(path.read_text()) if path.exists() else {"unknown_ids": 0, "batch_calls": 0}
    new = {"unknown_ids": old["unknown_ids"] + unknown, "batch_calls": old["batch_calls"] + calls}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(new))
    os.replace(tmp, path)
    return new["unknown_ids"]


def _git(root: Path) -> tuple[str | None, bool | None]:
    try:
        return head_commit(root), is_dirty(root)
    except (RuntimeError, OSError):
        return None, None


def _planned(mode: str, n: int) -> tuple[int, int]:
    """Calls expected and the most this run could use: a reply that is not valid JSON costs the client a second call."""
    return (n, 2 * n) if mode == "single" else (math.ceil(n / pg.BATCH_SIZE), math.ceil(n / pg.BATCH_SIZE) + 2 * n)


def run(args: Args, *, llm, root: Path = ROOT, counter_path: Path = CLAUDE_CALLS_PATH, cap: int = CLAUDE_CALL_CAP,
        guard: UsageGuard | None, say: Callable[[str], None] = print, sleep: Callable[[float], None] = time.sleep) -> Path:
    if args.mode not in ("single", "batch"):
        raise Refused("mode must be single or batch")
    if args.use_guard and guard is None:
        raise Refused("the weekly usage guard needs a usage reader; pass none only with --no-usage-guard")
    start_pct = guard.start() if guard is not None and args.use_guard else None
    pool = load_pool(root)
    sample = draw_sample(pool, args.sample)
    items = [c.item for c in sample.claims]
    header = {"kind": "judge-calibrate-rows", "sample_id": sample.sample_id, "mode": args.mode, "model": llm.model,
              "seed": SEED, "rubric": RUBRIC_VERSION, **({"batch_size": pg.BATCH_SIZE} if args.mode == "batch" else {})}
    path = root / PARTIAL_REL / f"{sample.sample_id}.{args.mode}.rows.jsonl"
    store = pg.RowStore(path, header)
    todo = pg.pending(items, store)
    expected, worst = _planned(args.mode, len(todo))
    say(f"sample {sample.sample_id}: {len(sample.claims)} claims from a pool of {len(pool)}; {len(sample.claims) - len(todo)} already "
        f"graded in saved rows. Planned: {expected} calls to {llm.model} in mode {args.mode} (up to {worst} if replies need "
        f"retries), concurrency {args.concurrency}; shared count {_shared_count(counter_path)} of {cap}. Weekly usage "
        f"{'not guarded' if start_pct is None else f'{start_pct:g}% (stop at {args.max_weekly_pct:g}%)'}.")
    before = preflight(len(todo), counter_path, cap)
    calls_before = llm.calls_made
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()
    stop = (lambda: guard.check()) if guard is not None and args.use_guard else (lambda: None)
    if args.mode == "single":
        out = pg.grade_single(items, llm, store, concurrency=args.concurrency, should_stop=stop, sleep=sleep)
    else:
        out = pg.grade_batches(items, llm, store, concurrency=args.concurrency, should_stop=stop, sleep=sleep)
    wall = time.monotonic() - t0
    end_pct = guard.final() if guard is not None and args.use_guard else None
    unknown_total = _add_meta(root, sample.sample_id, out.unknown_ids, out.batch_calls) if args.mode == "batch" else 0
    graded_now = len(todo) - len(pg.pending(items, store))
    if out.stop_reason:
        raise Stopped(f"{out.stop_reason}; {len(store.rows)} rows kept, rerun the same command to resume", start_pct, end_pct)
    ids = [c.id for c in sample.claims]
    haiku = {i: store.rows[i]["cloud"] for i in ids if store.rows.get(i, {}).get("status") == "ok"}
    if not haiku:
        raise Refused("no claim got a verdict, so there is nothing to compare; see the saved rows")
    other = "single" if args.mode == "batch" else "batch"
    other_path = root / PARTIAL_REL / f"{sample.sample_id}.{other}.rows.jsonl"
    batching = None
    if other_path.exists():
        other_store = pg.RowStore(other_path, json.loads(other_path.read_text(encoding="utf-8").splitlines()[0]))
        single_rows, batch_rows = (store.rows, other_store.rows) if args.mode == "single" else (other_store.rows, store.rows)
        unknown = unknown_total if args.mode == "batch" else (json.loads(_meta_path(root, sample.sample_id).read_text())["unknown_ids"]
                                                              if _meta_path(root, sample.sample_id).exists() else 0)
        batching = batching_report(ids, single_rows, batch_rows, unknown)
    commit, dirty = _git(root)
    rec = {
        "kind": "judge-calibration", "id": f"JCAL-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}", "provisional": True,
        "notes": [
            "Provisional: no claim comes from this record alone.",
            "Sonnet is a reference, not ground truth. The cheaper grader is from the same family, so their errors may be "
            "correlated and agreement may flatter it.",
            "Entry 53's rule: the cheaper grader is 'clearly worse' only if the interval of its agreement with Sonnet minus "
            "the local judge's lies wholly below zero. Claims the grader did not grade are left out of every figure and listed.",
            "Intervals resample whole cases (claims of one case move together); per-set figures rest on fewer cases.",
            "In the saved rows `claim_supported` is the local verdict and `cloud_supported` is Sonnet's; checked against the "
            "records' grounded_rate and cloud_grounded_rate.",
        ],
        "mode": args.mode, "grader": {"model": llm.model, "rubric_version": RUBRIC_VERSION,
                                      "prompt": "the local judge's rubric text, e2-v1" if args.mode == "single" else
                                                "the e2-v1 rubric text with a reply of one verdict per item id"},
        "git_commit": commit, "dirty": dirty, "source_records": list(JC_RECORDS),
        "source_runs": ["E3T-20261104T131120203662Z", "E3T-20261104T131417960789Z"],
        "sampling": {"rule": SAMPLING_RULE, "seed": SEED, "n": len(sample.claims), "n_requested": args.sample,
                     "sample_id": sample.sample_id, "pool": len(pool), "strata": sample.strata},
        "run": {
            "started_at": started, "concurrency": args.concurrency, "wall_seconds": wall, "claims_graded_this_run": graded_now,
            "claims_per_minute": graded_now / max(wall, 1e-9) * 60, "planned_calls": expected, "planned_calls_worst": worst,
            "calls_made": llm.calls_made - calls_before, "shared_counter_before": before,
            "shared_counter_after": _shared_count(counter_path), "failures": out.failed_rows,
            "rate_limit_events": out.rate_limit_events, "batch_calls": out.batch_calls,
            "usage_start_pct": start_pct, "usage_end_pct": end_pct, "max_weekly_pct": args.max_weekly_pct if args.use_guard else None,
            "note": "calls_made counts this process's calls to the client; the shared counter also moves for other work",
        },
        "ungraded": [i for i in ids if i not in haiku],
        "comparison": analyse(sample.claims, haiku),
        "batching": batching,
        "rows": [{"id": c.id, "set": c.set, "system": c.system, "local": c.local, "sonnet": c.sonnet, "grader": haiku.get(c.id),
                  "via": store.rows.get(c.id, {}).get("via")} for c in sample.claims],
    }
    target = root / OUT_REL / f"{rec['id']}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    return target


def _shared_count(path: Path) -> int:
    from contextlens.llm.client import _load_calls

    return _load_calls(path)["count"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Calibrate the cheaper grader against Sonnet (docs/DECISIONS.md, entry 53)")
    ap.add_argument("--sample", type=int, default=SAMPLE_DEFAULT)
    ap.add_argument("--concurrency", type=int, default=pg.DEFAULT_CONCURRENCY)
    ap.add_argument("--mode", choices=("single", "batch"), required=True)
    ap.add_argument("--max-weekly-pct", type=float, default=90.0)
    ap.add_argument("--no-usage-guard", action="store_true", help="run without reading weekly usage (needed if it cannot be read)")
    a = ap.parse_args(argv)
    args = Args(a.sample, a.concurrency, a.mode, a.max_weekly_pct, not a.no_usage_guard)
    guard = UsageGuard(args.max_weekly_pct) if args.use_guard else None
    try:
        client = pg.CountedClaude()
        path = run(args, llm=client, guard=guard)
    except UsageUnreadable as exc:
        print(f"refused: weekly usage cannot be read ({exc}); pass --no-usage-guard to run without the guard", file=sys.stderr)
        return 1
    except Refused as exc:
        print("refused:", exc, file=sys.stderr)
        return 1
    except Stopped as exc:
        print(f"stopped: {exc}. Weekly usage at start {exc.usage_start_pct}%, at end {exc.usage_end_pct}%.", file=sys.stderr)
        return 2
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
