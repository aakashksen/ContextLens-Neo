"""Choose the operating point of the trained-then-prompted system on dev, from saved predictions only.

No model is run: the trained model's probabilities come from `rescore` and the prompted checker's verdicts from T5.2's
records, joined by case id. A threshold on P(contradiction) is swept for two systems: the trained model alone, and
the cascade (flagged cases are alerts only if the prompted checker also says contradiction). Threshold 0 for the cascade
is the prompted checker alone.

Rule, fixed here before any sweep result was looked at: among points whose catch rate is at least CATCH_FLOOR on every
source, take the lowest false-alert rate (mean over the sources, so the small scenario set counts as much as DECODE).
Ties go to the higher catch rate, then to the trained model alone (cheaper), then to the higher threshold.

Why the catch floor is 0.5: the design's named failure is false alarms that make users stop trusting the checker, so
false alerts are what is minimised. But a checker that never alerts has no false alarms and is worth nothing, so the floor
keeps it useful: below half, it misses more real contradictions than it finds and cannot be relied on as a check. A
higher floor would push the choice toward the many-false-alarms end the design warns against. The number comes from
that reasoning, not from a result; the only results seen before fixing it were T5.2's headline scores, not any sweep.

Dev is selection data here, so the chosen point's scores are not held out, and whether the dev labels are reviewed is the records' labels_reviewed flag, not stated here.
The test splits are never read.

Run: python -m contextlens.consistency.select_cascade --data-root <folder with datasets/raw/>
"""
import argparse
import json
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from contextlens.agents.cascade import is_flagged
from contextlens.consistency.rescore import OUT_REL, RECORDS, SPLIT, load_dev, read_rows
from contextlens.eval.records import ROOT, SCENARIO_SOURCES
from contextlens.eval.series import predicted_label
from contextlens.replay.runner import Prediction
from contextlens.schemas import Scenario

CATCH_FLOOR = 0.5
THRESHOLDS = tuple(round(i * 0.05, 2) for i in range(20))  # 0.0 to 0.95
SYSTEMS = ("trained", "cascade")
SOURCES = ("decode", "scenarios")
BASE_RATES = (0.05, 0.20)
SEED = 13
HAND_READ_N = 30
CONFIG_REL = Path("configs/cascade.json")
PROMPTED_MODEL = "qwen3.5:4b-q4_K_M"
PRECISION_NOTE = ("calculated, not measured: precision = c*p / (c*p + f*(1-p)) from the measured catch rate c and "
                  "false-alert rate f at the selected point, for a stated share p of statements that really are contradictions")


@dataclass(frozen=True)
class Case:
    source: str
    id: str
    contradiction: bool  # the gold label
    p_contradiction: float  # the trained model's
    prompted_alert: bool  # the prompted checker said contradiction
    trained_ms: float | None = None
    prompted_ms: float | None = None


@dataclass(frozen=True)
class Point:
    system: str
    threshold: float
    catch: dict[str, tuple[int, int]]  # per source: real contradictions alerted, real contradictions
    far: dict[str, tuple[int, int]]  # per source: other cases alerted, other cases


def alert(case: Case, system: str, threshold: float) -> bool:
    return is_flagged(case.p_contradiction, threshold) and (system == "trained" or case.prompted_alert)


def _rate(kn: tuple[int, int]) -> float:
    return kn[0] / kn[1]


def mean_catch(p: Point) -> float:
    return statistics.mean(_rate(v) for v in p.catch.values())


def mean_far(p: Point) -> float:
    return statistics.mean(_rate(v) for v in p.far.values())


def sweep(cases: list[Case], thresholds: tuple[float, ...] = THRESHOLDS) -> list[Point]:
    out = []
    for system in SYSTEMS:
        for t in thresholds:
            catch, far = {}, {}
            for src in sorted({c.source for c in cases}):
                pos = [c for c in cases if c.source == src and c.contradiction]
                neg = [c for c in cases if c.source == src and not c.contradiction]
                if not pos or not neg:
                    raise ValueError(f"{src} needs both real contradictions and other cases to score a sweep")
                catch[src] = (sum(alert(c, system, t) for c in pos), len(pos))
                far[src] = (sum(alert(c, system, t) for c in neg), len(neg))
            out.append(Point(system, t, catch, far))
    return out


def select(points: list[Point], floor: float = CATCH_FLOOR) -> Point | None:
    eligible = [p for p in points if all(_rate(v) >= floor for v in p.catch.values())]
    return min(eligible, key=lambda p: (mean_far(p), -mean_catch(p), p.system != "trained", -p.threshold), default=None)


def alert_precision(catch: float, far: float, base_rate: float) -> float | None:
    denominator = catch * base_rate + far * (1 - base_rate)
    return catch * base_rate / denominator if denominator else None


def build_cases(dev: dict[str, list[Scenario]], rescored: dict[str, list[dict]], prompted: dict[str, list[Prediction]],
                trained: dict[str, list[Prediction]] | None = None) -> list[Case]:
    """Join the sources of a case by id. Dev, rescored and prompted ids must be the same list; unlabelled scenarios are not scored."""
    out = []
    for src, scs in dev.items():
        ids = [s.id for s in scs]
        if [r["id"] for r in rescored[src]] != ids or [p.scenario_id for p in prompted[src]] != ids:
            raise ValueError(f"{src}: the dev, rescored and prompted ids do not line up")
        t_ms = {p.scenario_id: p.stage_latency_ms.get("total") for p in (trained or {}).get(src, [])}
        for s, r, p in zip(scs, rescored[src], prompted[src], strict=True):
            if s.gold_label is None:
                continue
            out.append(Case(src, s.id, s.gold_label == "contradiction", r["probs"]["contradiction"],
                            predicted_label(p) == "contradiction", t_ms.get(s.id), p.stage_latency_ms.get("total")))
    return out


def sample_false_alarms(cases: list[Case], point: Point, n: int = HAND_READ_N, seed: int = SEED) -> list[Case]:
    wrong = sorted((c for c in cases if not c.contradiction and alert(c, point.system, point.threshold)),
                   key=lambda c: (c.source, c.id))
    return random.Random(seed).sample(wrong, min(n, len(wrong)))


def write_config(path: Path, point: Point, provenance: dict) -> None:
    cfg = {
        "system": point.system, "flag_threshold": point.threshold, "catch_floor": CATCH_FLOOR,
        "selection_rule": "lowest mean false-alert rate over the sources among points with catch rate >= catch_floor on every "
                          "source; ties to higher catch, then the trained model alone, then the higher threshold",
        "selection_data": "dev (selection data, not held out); the labels' review status is in each record's labels_reviewed",
        "provisional": True, "thresholds_swept": list(THRESHOLDS),
        "stage_two": {"model": PROMPTED_MODEL, "prompt": "consistency v1"}, **provenance,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")


def _kn(kn: tuple[int, int]) -> str:
    return f"{_rate(kn):.3f} ({kn[0]}/{kn[1]})"


def _composed_ms(cases: list[Case], point: Point) -> str:
    ms = sorted(c.trained_ms + (c.prompted_ms if point.system == "cascade" and is_flagged(c.p_contradiction, point.threshold) else 0.0)
                for c in cases if c.trained_ms is not None and c.prompted_ms is not None)
    if not ms:
        return "not available"
    return f"mean {statistics.mean(ms):.0f}, p50 {ms[len(ms) // 2]:.0f}, p95 {ms[int(0.95 * (len(ms) - 1))]:.0f}"


def report(point: Point, points: list[Point], cases: list[Case] | None = None) -> str:
    lines = ["# Operating point on dev (dev is selection data; label status is the records' labels_reviewed flag)", "",
             f"Selected: system {point.system}, flag threshold {point.threshold}, catch floor {CATCH_FLOOR}.", "",
             "## Measured on dev (rate, with counts)", ""]
    for src in sorted(point.catch):
        lines.append(f"- {src}: catch rate {_kn(point.catch[src])}; false-alert rate {_kn(point.far[src])}")
    lines += ["", f"## Alert precision ({PRECISION_NOTE})", ""]
    for src in sorted(point.catch):
        for rate in BASE_RATES:
            prec = alert_precision(_rate(point.catch[src]), _rate(point.far[src]), rate)
            lines.append(f"- {src}, {rate:.0%} of statements are contradictions: precision "
                         + (f"{prec:.3f}" if prec is not None else "undefined") + " (calculated, not measured)")
    if cases:
        lines += ["", "## Latency per case, ms, composed from T5.2's saved per-case stage times (not a live cascade run)", ""]
        for src in sorted(point.catch):
            lines.append(f"- {src}: {_composed_ms([c for c in cases if c.source == src], point)}; "
                         "tokens: not recorded in T5.2's records")
    lines += ["", "## Sweep: catch / false-alert per source, eligible = catch floor met on every source", ""]
    for p in points:
        ok = all(_rate(v) >= CATCH_FLOOR for v in p.catch.values())
        cells = "; ".join(f"{s} catch {_kn(p.catch[s])} false {_kn(p.far[s])}" for s in sorted(p.catch))
        lines.append(f"- {p.system} t={p.threshold}: {'eligible' if ok else 'below floor'}; {cells}")
    return "\n".join(lines) + "\n"


def load_inputs(root: Path, data_root: Path) -> tuple[dict[str, list[Scenario]], list[Case]]:
    dev, rescored, prompted, trained = {}, {}, {}, {}
    for src in SOURCES:
        manifest_rel, dir_rel = SCENARIO_SOURCES[src]
        dev[src] = load_dev(root / manifest_rel, data_root / dir_rel / f"{SPLIT}.jsonl")
        rescored[src] = read_rows(root / OUT_REL / f"rescored-{src}-{SPLIT}.jsonl")
        for store, kind in ((prompted, "prompted"), (trained, "trained")):
            path = root / "results/runs" / f"{RECORDS[src]}.{kind}.predictions.jsonl"
            store[src] = [Prediction.model_validate_json(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return dev, build_cases(dev, rescored, prompted, trained)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=ROOT, help="folder holding datasets/raw/ (git-ignored)")
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args()
    _, cases = load_inputs(args.root, args.data_root)
    points = sweep(cases)
    point = select(points)
    if point is None:
        print(f"no point meets the catch floor of {CATCH_FLOOR} on every source; nothing written", file=sys.stderr)
        return 2
    record = json.loads((args.root / "results/runs" / f"{RECORDS['scenarios']}.json").read_text(encoding="utf-8"))
    write_config(args.root / CONFIG_REL, point, {
        "trained_checkpoint": "models/e4/lr3e-05_ep3", "checkpoint_sha256": record["config"]["checkpoint_sha256"],
        "from_records": RECORDS, "cases": {s: sum(c.source == s for c in cases) for s in SOURCES}})
    text = report(point, points, cases)
    out = args.root / OUT_REL
    out.mkdir(parents=True, exist_ok=True)
    (out / "selection-report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
