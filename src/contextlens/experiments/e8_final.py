"""E8: the final test. One run, on the test split, of three systems and the detector, with three commands.

    freeze  writes configs/e8_freeze.json from choices made on dev (no test data is read). Commit it.
    write   (GPU) runs everything on test and saves each finished row as it exists. Resumable at the same commit.
    judge   (no GPU) grades every claim with Haiku through `claude -p`, one claim per call, in parallel, then writes the
            one E8 record. Sonnet grades a seeded 300-claim sample (frozen, not a flag); `sonnet-extra` adds more apart from the record.

    python -m contextlens.experiments.e8_final freeze --adapter-base DIR --adapter DIR --checker DIR
    python -m contextlens.experiments.e8_final write --run-conditions "<what else was running>" [--override-note TEXT]
    python -m contextlens.experiments.e8_final judge --from-commit SHA [--override-note TEXT]

The three card systems run on our scenarios' test cases, prompt v3 for all (docs/DECISIONS.md entry 58): `team_shipping_changed` (the
product setting `CONTEXTLENS_ASSISTANT=team_shipping_changed`, entry 57) as primary, `single`, and the same team with the trained
models (the T6.2 adapter as the writer, the T5.2 checkpoint as the consistency checker). Cases whose conversation is LongMemEval,
and QMSum, are searched only (the frozen retrieval config) and no answer is written. The detector runs on DECODE test (the
trained model and TF-IDF only) and on our scenarios (the trained model and the baseline that won dev). Each source is reported
apart in `config.benchmarks`. Simpler systems stay as dev results.

The test split is touched once. `write` refuses unless the freeze file is committed and unchanged, the tree is clean, and no
earlier test record or rows of another commit exist (an `--override-note` allows a repeat and goes in the record).
"""

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contextlens.agents.consistency_check import ConsistencyChecker
from contextlens.agents.graph import TeamAssistant
from contextlens.agents.tools import Tools
from contextlens.eval import series
from contextlens.eval.check_run import recompute_metrics
from contextlens.eval.records import (ROOT, RUNS_REL, SCENARIO_SOURCES, git, head_commit, is_dirty, label_text, load_verified,
                                      write_atomic, write_run)
from contextlens.eval.stats import mcnemar_exact, paired_bootstrap_diff
from contextlens.experiments import e1_retrieval as e1
from contextlens.experiments import e2_context as e2
from contextlens.experiments import e3_team as e3
from contextlens.experiments import e4_consistency as e4
from contextlens.experiments import e6_judge as j6
from contextlens.experiments import e6_prompts as e6
from contextlens.experiments import parallel_grader as pg
from contextlens.experiments.judge_calibrate import UsageGuard, UsageUnreadable
from contextlens.experiments.judge_check import Refused
from contextlens.experiments.team_attempt import pair_verdict
from contextlens.llm.client import CLAUDE_CALL_CAP, CLAUDE_CALLS_PATH, LLMClient
from contextlens.llm.ollama_ctl import OllamaCtl
from contextlens.memory.extract import MemoryStore
from contextlens.memory.store import InMemoryTranscriptStore
from contextlens.replay.runner import Prediction
from contextlens.retrieval.hybrid import Retriever
from contextlens.retrieval.query import query_for
from contextlens.schemas import Scenario

EXPERIMENT, SPLIT = "E8", "test"
FREEZE_REL = Path("configs/e8_freeze.json")
ROWS_REL = Path("results/partial/e8-rows.jsonl")  # git-ignored; one line per finished row
JUDGE_DIR_REL = Path("results/partial/e8-judge")
EXTRA_PREFIX = "E8X"  # public-benchmark and detector rows: results/runs/E8X-<stamp>Z.<part>-<system>.predictions.jsonl (a name is_dirty allows)
PRIMARY, SINGLE, TRAINED_TEAM = "team_shipping_changed", "single", "team_trained"
SYSTEMS = (PRIMARY, SINGLE, TRAINED_TEAM)
PROMPT = "v3"
SUBSET_NAME = "e8-cards"  # the ids of the test cases that get written answers (check_run's subset manifest)
SUBSET_REL = Path("datasets/splits/subsets") / f"{SUBSET_NAME}.json"
LME = "longmemeval"
SEARCH_SYSTEM = "hybrid_t1"
MAX_WEEKLY_PCT = 95.0  # the person, 2026-11-05
SONNET_SAMPLE = 300  # the coordinator: Haiku grades every claim, Sonnet a seeded sample of this size
EXTRA_DIR_REL = Path("results/e8-sonnet-extra")
SAMPLE_SEED = 13
DECISION_ENTRIES = (49, 51, 56, 57, 58)
CAL_NOTE = j6.CALIBRATION_NOTE


# ---------------------------------------------------------------- freeze

def sha256_path(path: Path) -> str:
    """A file's hash, or for a directory the hash of its files' relative names and bytes in sorted order."""
    h = hashlib.sha256()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for f in files:
        h.update((f.relative_to(path).as_posix() if path.is_dir() else f.name).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def _need(path: Path, what: str) -> Path:
    if not path.exists():
        raise Refused(f"{what} not found at {path}; the trained models must exist before E8 is frozen")
    return path


def write_cards_subset(root: Path) -> Path:
    """The ids-only manifest of the test cases that get cards: every case whose conversation is not LongMemEval. It reads
    only `id` and `conversation.source` from each line of the test file, never the text."""
    ids = []
    for line in (root / SCENARIO_SOURCES["scenarios"][1] / f"{SPLIT}.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            if d["conversation"]["source"] != LME:
                ids.append(d["id"])
    path = root / SUBSET_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, json.dumps({"name": SUBSET_NAME, "split": SPLIT, "ids": ids}, indent=1) + "\n")
    return path


def build_freeze(root: Path, *, adapter_base: Path, adapter: Path, checker: Path, checker_threshold: float,
                 baselines: dict[str, str] | None = None) -> dict[str, Any]:
    """The choices made on dev. Reads only the committed manifests (hashes), never a test file."""
    _need(adapter_base, "the adapter's base model")
    _need(adapter, "the T6.2 adapter")
    _need(checker, "the T5.2 consistency checkpoint")
    test_hashes = {src: json.loads((root / SCENARIO_SOURCES[src][0]).read_text(encoding="utf-8"))["splits"][SPLIT]["hash"]
                   for src in ("scenarios", "qmsum", "decode")}
    retrieval = root / e2.CONFIG_REL
    subset = write_cards_subset(root)
    return {
        "kind": "e8-freeze", "experiment": EXPERIMENT, "chosen_on": "dev", "decision_entries": list(DECISION_ENTRIES),
        "systems": {"primary": PRIMARY, "others": [SINGLE, TRAINED_TEAM]},
        "assistant_setting": {"env": "CONTEXTLENS_ASSISTANT", "value": PRIMARY},
        "prompt_version": PROMPT, "writer_model": e2.WRITER_MODEL, "embedder_model": e2.EMBED_MODEL, "deadline_s": e2.DEADLINE_S,
        "retrieval": {"path": e2.CONFIG_REL.as_posix(), "sha256": sha256_path(retrieval)},
        "trained": {"adapter": {"base_model_path": adapter_base.as_posix(), "adapter_path": adapter.as_posix(),
                                "sha256": sha256_path(adapter)},
                    "checker": {"path": checker.as_posix(), "sha256": sha256_path(checker), "threshold": checker_threshold}},
        "detectors": baselines or {"scenarios": "prompted", "decode": "tfidf"},
        "search_only": ["qmsum", LME],
        "judging": {"haiku": pg.HAIKU_MODEL, "sonnet": e2.CLOUD_MODEL, "sonnet_sample": SONNET_SAMPLE, "sample_seed": SAMPLE_SEED,
                    "max_weekly_pct": MAX_WEEKLY_PCT, "rubric": e2.RUBRIC_VERSION},
        "test_hashes": test_hashes,
        "cards_subset": {"name": SUBSET_NAME, "path": SUBSET_REL.as_posix(), "sha256": sha256_path(subset),
                         "reads": "only `id` and `conversation.source` of each line of the test file; no text",
                         "committed": "in the same commit as this config file; committed_freeze refuses otherwise"},
    }


def write_freeze(root: Path, freeze: dict[str, Any]) -> Path:
    path = root / FREEZE_REL
    write_atomic(path, json.dumps(freeze, indent=1) + "\n")
    return path


def committed_freeze(root: Path) -> dict[str, Any]:
    """The freeze config, only if it is committed and the working copy matches the commit."""
    rel = FREEZE_REL.as_posix()
    if not (root / FREEZE_REL).is_file():
        raise Refused(f"{rel} does not exist; run `freeze` and commit it before the test is touched")
    freeze = json.loads((root / FREEZE_REL).read_text(encoding="utf-8"))
    if freeze.get("kind") != "e8-freeze":
        raise Refused(f"{rel} is not an E8 freeze file")
    paths = [rel, *([freeze["cards_subset"]["path"]] if "cards_subset" in freeze else [])]
    for r in paths:
        if git(root, "ls-files", "--error-unmatch", r).returncode != 0:
            raise Refused(f"{r} is not committed; commit it before the test is touched")
        if git(root, "diff", "--quiet", "HEAD", "--", r).returncode != 0:
            raise Refused(f"{r} differs from the committed one; commit the change before the test is touched")
    return freeze


def check_inputs(root: Path, freeze: dict[str, Any]) -> None:
    """The trained models and the retrieval config are what was frozen, or the run is refused."""
    t = freeze["trained"]
    for what, p, want in (("adapter", t["adapter"]["adapter_path"], t["adapter"]["sha256"]),
                          ("consistency checkpoint", t["checker"]["path"], t["checker"]["sha256"])):
        path = Path(p) if Path(p).is_absolute() else root / p
        _need(path, f"the {what}")
        if sha256_path(path) != want:
            raise Refused(f"the {what} at {path} is not the one that was frozen (hash differs)")
    base = Path(t["adapter"]["base_model_path"])
    _need(base if base.is_absolute() else root / base, "the adapter's base model")
    if sha256_path(root / freeze["retrieval"]["path"]) != freeze["retrieval"]["sha256"]:
        raise Refused("the retrieval config is not the one that was frozen")
    if sha256_path(root / freeze["cards_subset"]["path"]) != freeze["cards_subset"]["sha256"]:
        raise Refused("the card-case manifest is not the one that was frozen")
    for src, want in freeze["test_hashes"].items():
        entry = json.loads((root / SCENARIO_SOURCES[src][0]).read_text(encoding="utf-8"))["splits"][SPLIT]["hash"]
        if entry != want:
            raise Refused(f"the {src} test manifest changed since the freeze")


def existing_test_records(root: Path) -> list[str]:
    """Run ids of every test-split record of any experiment (what check_run counts as a touch)."""
    out = []
    for p in sorted((root / RUNS_REL).glob("*-*Z.json")):
        if json.loads(p.read_text(encoding="utf-8")).get("split") == SPLIT:
            out.append(p.stem)
    return out


def other_commit_rows(root: Path, commit: str) -> list[str]:
    path = root / ROWS_REL
    if not path.exists():
        return []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            c = json.loads(line)["commit"]
        except (ValueError, KeyError):
            continue
        if c != commit:
            seen.add(c[:12])
    return sorted(seen)


def require_first_test_run(root: Path, commit: str, override_note: str | None) -> None:
    """Refuse a second test run unless the person's note says why. Rows saved at this same commit are a resume, not a second run."""
    again = [f"test record {r}" for r in existing_test_records(root)] + [f"saved rows of commit {c}" for c in other_commit_rows(root, commit)]
    if again and not (override_note or "").strip():
        raise Refused(f"the test split was already run ({', '.join(again)}); a second run needs --override-note saying why")


def require_ready(root: Path, override_note: str | None) -> dict[str, Any]:
    if is_dirty(root):
        raise Refused("the working tree has uncommitted changes; commit first")
    freeze = committed_freeze(root)
    require_first_test_run(root, head_commit(root), override_note)
    check_inputs(root, freeze)
    return freeze


# ---------------------------------------------------------------- saved rows

def _saved(root: Path, commit: str, part: str, system: str, ids: set[str]) -> dict[str, Prediction]:
    return e2._saved_rows(root / ROWS_REL, commit, part, system, ids, EXPERIMENT)[0]


def run_part(root: Path, commit: str, part: str, system: str, scenarios: list[Scenario], make: Callable[[Scenario], Any],
             say: Callable[[str], None]) -> list[Prediction]:
    """Every case of `scenarios` on `system`; a case with a saved row at this commit is never run again."""
    have = _saved(root, commit, part, system, {s.id for s in scenarios})

    def save(row: Prediction) -> None:
        e2._append_durable(root / ROWS_REL, json.dumps({"commit": commit, "subset": part, "winner": system, "model": EXPERIMENT,
                                                        "calls": 0, "row": json.loads(row.model_dump_json())}))

    say(f"{part}/{system}: {len(have)} of {len(scenarios)} rows already saved")
    return e2.run_system(system, scenarios, make, e2.STREAK_LOCAL, say=say, have=have, on_row=save)


def run_rows(root: Path, commit: str, part: str, system: str, scenarios: list[Scenario], make_row: Callable[[Scenario], Prediction],
             say: Callable[[str], None]) -> list[Prediction]:
    """Like run_part for rows that are not cards (ranked turns, a bare label): one row per case, saved the moment it exists."""
    have = _saved(root, commit, part, system, {s.id for s in scenarios})
    rows = []
    for sc in scenarios:
        row = have.get(sc.id)
        if row is None:
            row = make_row(sc)
            e2._append_durable(root / ROWS_REL, json.dumps({"commit": commit, "subset": part, "winner": system, "model": EXPERIMENT,
                                                            "calls": 0, "row": json.loads(row.model_dump_json())}))
        rows.append(row)
    say(f"{part}/{system}: {len(rows)} rows ({len(have)} were already saved)")
    return rows


def split_cards(scenarios: list[Scenario]) -> tuple[list[Scenario], list[Scenario]]:
    """(cases that get written answers, LongMemEval cases that are searched only)."""
    return ([s for s in scenarios if s.conversation.source != LME], [s for s in scenarios if s.conversation.source == LME])


def search_rows(scenarios: list[Scenario], embedder: Any, config: dict[str, Any], say: Callable[[str], None]) -> list[Prediction]:
    """Ranked turns from the frozen retrieval config (one-turn chunks, no reranker); no card is written."""
    if config.get("rerank") or config.get("chunk_turns", 1) != 1:
        raise Refused("the frozen retrieval config needs a one-turn chunk and no reranker")
    by_conv: dict[str, list[Scenario]] = defaultdict(list)
    for sc in scenarios:
        by_conv[sc.conversation.id].append(sc)
    rows: dict[str, Prediction] = {}
    for group in by_conv.values():
        units, members = e1.chunks(group[0].conversation.turns, 1)
        store = InMemoryTranscriptStore()
        for u in units:
            store.add(u)
        retriever = Retriever(store, embedder)
        for sc in group:
            query, last = query_for(sc)
            if last >= 0:
                retriever.search(sc.conversation.id, query, last, k=1, mode="dense")
            failed: dict[str, str] = {}
            row = e1._press(retriever, None, config["mode"], sc, members, query, last, SEARCH_SYSTEM, failed, say)
            if row is None:
                raise e2.RunBroken(f"search failed on {sc.id}: {failed.get(SEARCH_SYSTEM)}")
            rows[sc.id] = row
    return [rows[s.id] for s in scenarios]


def build_team_trained(sc: Scenario, hf_writer: LLMClient, checker: ConsistencyChecker, embedder: Any, memory: MemoryStore) -> Any:
    """The shipping team (checker, memory_gap_abstains on, prompt v3) with the adapter as every role's model."""
    retriever = e6._retriever(sc, embedder)
    tools = Tools(retriever.store, retriever, memory, sc.request.as_of_seq, sc.conversation.id, checker)
    return TeamAssistant({"memory": hf_writer, "consistency": hf_writer, "assistance": hf_writer}, tools, PROMPT,
                         deadline_s=e2.DEADLINE_S, memory_gap_abstains=True)


# ---------------------------------------------------------------- the write command (GPU)

def run_write(*, root: Path = ROOT, writer: LLMClient, embedder: Any, ctl: OllamaCtl, make_hf_writer: Callable[[dict[str, Any]], LLMClient],
              make_checker: Callable[[str, dict[str, Any]], ConsistencyChecker], run_conditions: str, override_note: str | None = None,
              say: Callable[[str], None] = print) -> dict[str, Any]:
    """Run the test once and save every row. Returns counts. Judging is a separate command."""
    if not run_conditions.strip():
        raise e2.RunBroken("run_conditions is required: say what else was running while latency was measured")
    freeze = require_ready(root, override_note)
    commit = head_commit(root)
    cfg = e2.load_retrieval_config(root / freeze["retrieval"]["path"])
    loaded = {src: load_verified(src, SPLIT, root) for src in ("scenarios", "qmsum", "decode")}
    cards, lme = split_cards(loaded["scenarios"][0])
    names = e2.Models()
    mine = [(names.writer, False), (names.embedder, True)]
    counts: dict[str, int] = {}
    try:
        have = {n: _saved(root, commit, "cards", n, {s.id for s in cards}) for n in SYSTEMS}
        todo = [s for s in cards if any(s.id not in have[n] for n in SYSTEMS)]
        say(f"cards: {len(cards)} cases, building memory for {len(todo)}")
        memories: dict[str, MemoryStore] = {}
        for sc in todo:
            memories[sc.id], _ = e2.build_memory(sc, writer)
        for system in (PRIMARY, SINGLE):
            counts[system] = len(run_part(root, commit, "cards", system, cards, lambda sc, system=system: e6._build_assistant(
                system, PROMPT, sc, writer, e6._retriever(sc, embedder), memories.get(sc.id, MemoryStore()), cfg), say))
        ctl.unload(names.writer)  # the adapter needs the GPU the writer held
        hf = make_hf_writer(freeze["trained"]["adapter"])
        trained = make_checker("trained", freeze["trained"]["checker"])
        counts[TRAINED_TEAM] = len(run_part(root, commit, "cards", TRAINED_TEAM, cards, lambda sc: build_team_trained(
            sc, hf, trained, embedder, memories.get(sc.id, MemoryStore())), say))
        for source, scs in (("qmsum", loaded["qmsum"][0]), (LME, lme)):
            counts[f"search:{source}"] = len(run_rows(
                root, commit, f"search:{source}", SEARCH_SYSTEM, scs, lambda sc: search_rows([sc], embedder, cfg, say)[0], say))
        for source, scs in (("scenarios", cards), ("decode", loaded["decode"][0])):
            predict = e4.predict_scenarios if source == "scenarios" else e4.predict_decode
            for name in ("trained", freeze["detectors"][source]):
                checker = trained if name == "trained" else make_checker(name, freeze["trained"]["checker"])
                counts[f"detector:{source}:{name}"] = len(run_rows(
                    root, commit, f"detector:{source}", name, scs, lambda sc, checker=checker, predict=predict: predict(checker, [sc])[0], say))
        write_atomic(root / ROWS_REL.with_name("e8-write.json"), json.dumps({"commit": commit, "run_conditions": run_conditions,
                                                                           "override_note": override_note,
                                                                           "finished_at": datetime.now(timezone.utc).isoformat()}))
    finally:
        e2._unload_models(ctl, mine, say)
    return counts


# ---------------------------------------------------------------- judging and the record (no GPU)

def claim_id(system: str, scenario_id: str, i: int) -> str:
    return f"e8/{system}/{scenario_id}/{i}"


def items_of(rows: dict[str, list[Prediction]]) -> list[pg.Item]:
    return [pg.Item(claim_id(n, p.scenario_id, i), f.statement, tuple((e.speaker, e.quote) for e in f.evidence))
            for n, ps in rows.items() for p in ps if p.card is not None for i, f in enumerate(p.card.findings)]


def sonnet_order(ids: list[str], seed: int = SAMPLE_SEED) -> list[str]:
    """A seeded random order of the claims. The first SONNET_SAMPLE are the original sample; a later extra step takes the next ones."""
    order = sorted(ids)
    random.Random(seed).shuffle(order)
    return order


def sonnet_sample(ids: list[str], n: int = SONNET_SAMPLE, seed: int = SAMPLE_SEED) -> list[str]:
    return sonnet_order(ids, seed)[:n]


def sonnet_extra(ids: list[str], seed: int, done: int, more: int) -> list[str]:
    """The `more` claims after the first `done` of the seeded order: never one already graded."""
    return sonnet_order(ids, seed)[done:done + more]


def agreement(haiku: dict[str, bool], sonnet: dict[str, bool]) -> dict[str, Any]:
    """Haiku against Sonnet on the claims both graded: agreement, which way the disagreements lean, Cohen's kappa."""
    ids = sorted(set(haiku) & set(sonnet))
    n = len(ids)
    if not n:
        return {"claims": 0}
    both = sum(haiku[i] and sonnet[i] for i in ids)
    neither = sum(not haiku[i] and not sonnet[i] for i in ids)
    h_yes_s_no = sum(haiku[i] and not sonnet[i] for i in ids)
    h_no_s_yes = n - both - neither - h_yes_s_no
    p_o = (both + neither) / n
    p_e = ((both + h_yes_s_no) * (both + h_no_s_yes) + (neither + h_no_s_yes) * (neither + h_yes_s_no)) / (n * n)
    return {"claims": n, "agreement": p_o, "haiku_yes_sonnet_no": h_yes_s_no, "haiku_no_sonnet_yes": h_no_s_yes,
            "kappa": None if p_e == 1 else (p_o - p_e) / (1 - p_e)}


def with_claims(rows: dict[str, list[Prediction]], verdicts: dict[str, bool], field: str) -> dict[str, list[Prediction]]:
    """Rows where `field` holds one verdict per finding; a card with a finding that has no verdict is left as it was."""
    out = {}
    for n, ps in rows.items():
        out[n] = [p.model_copy(update={field: [verdicts[claim_id(n, p.scenario_id, i)] for i in range(len(p.card.findings))]})
                  if p.card is not None and p.card.findings
                  and all(claim_id(n, p.scenario_id, i) in verdicts for i in range(len(p.card.findings))) else p
                  for p in ps]
    return out


def all_saved(path: Path, commit: str, part: str, system: str, ids: set[str]) -> dict[str, Prediction]:
    """Every saved row of the commit, part and system, broken ones included (a broken card is counted, as an uninterrupted run would)."""
    out: dict[str, Prediction] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
            row = Prediction.model_validate(e["row"])
            if (e["commit"], e["subset"], e["winner"]) == (commit, part, system) and row.scenario_id in ids:
                out[row.scenario_id] = row
        except (ValueError, KeyError, TypeError):
            continue
    return out


def load_part(root: Path, commit: str, part: str, system: str, scs: list[Scenario]) -> list[Prediction]:
    have = all_saved(root / ROWS_REL, commit, part, system, {s.id for s in scs})
    missing = [s.id for s in scs if s.id not in have]
    if missing:
        raise Refused(f"{part}/{system}: {len(missing)} of {len(scs)} rows are not saved at {commit[:12]}; run `write` first")
    return [have[s.id] for s in scs]


def false_alert_compare(scs: list[Scenario], rows_a: list[Prediction], rows_b: list[Prediction], a: str, b: str) -> dict[str, Any] | None:
    """False alerts (a non-contradiction case predicted as contradiction) of a against b on the same cases: the difference,
    a bootstrap interval resampling conversations, and exact McNemar. Better or worse only when both agree."""
    ra, rb = {p.scenario_id: p for p in rows_a}, {p.scenario_id: p for p in rows_b}
    keep = [s for s in scs if s.gold_label and s.gold_label != "contradiction" and series.predicted_label(ra[s.id])
            and series.predicted_label(rb[s.id])]
    if not keep:
        return None
    ya = [series.predicted_label(ra[s.id]) == "contradiction" for s in keep]
    yb = [series.predicted_label(rb[s.id]) == "contradiction" for s in keep]
    diff, lo, hi = paired_bootstrap_diff([float(x) for x in ya], [float(x) for x in yb], [s.conversation.id for s in keep],
                                         n=series.RESAMPLES, seed=series.SEED)
    p = mcnemar_exact(ya, yb)
    return {"a": a, "b": b, "metric": "false_alert_rate", "cases": len(keep), "diff": diff, "ci": [lo, hi], "mcnemar_p": p,
            f"{a}_only": sum(x and not y for x, y in zip(ya, yb, strict=True)),
            f"{b}_only": sum(y and not x for x, y in zip(ya, yb, strict=True)),
            "verdict": pair_verdict(diff, lo, hi, p, True, higher_is_better=False)}


def detector_block(source: str, scs: list[Scenario], rows: dict[str, list[Prediction]], baseline: str) -> dict[str, Any]:
    metrics = {n: e4._saved(recompute_metrics(scs, p), source) for n, p in rows.items()}
    return {"cases": len(scs), "systems": list(rows), "baseline": baseline, "metrics": metrics,
            "false_alert_comparison": false_alert_compare(scs, rows["trained"], rows[baseline], "trained", baseline),
            "note": e4.DECODE_CAVEAT if source == "decode" else "four-label macro-F1 on our scenarios; the baseline is the dev winner"}


def search_block(scs: list[Scenario], rows: list[Prediction]) -> dict[str, Any]:
    fresh = recompute_metrics(scs, rows, e1.KS)
    return {"cases": len(scs), "system": SEARCH_SYSTEM, "metrics": {k: v for k, v in fresh.items() if series.base_name(k) in e1.SAVED},
            "note": "search only: no written answer exists for these cases"}


def primary_vs_single(comparisons: list[dict[str, Any]]) -> dict[str, str]:
    """The written verdict of each compared metric, the primary against `single`: better or worse only when the interval
    and, for a yes/no metric, McNemar agrees."""
    out = {}
    for c in comparisons:
        if (c["a"], c["b"]) == (PRIMARY, SINGLE) and c["metric"] in e3.COMPARE and "[" not in c["metric"]:
            out[c["metric"]] = pair_verdict(c["diff"], *c["ci"], c["mcnemar_p"], c["metric"] in e3.BINARY_METRICS,
                                            c["metric"] in e3.HIGHER_IS_BETTER)
    return out


def run_judge(*, root: Path = ROOT, from_commit: str, haiku: Any, sonnet: Any, guard: UsageGuard, override_note: str | None = None,
              counter_path: Path = CLAUDE_CALLS_PATH, cap: int = CLAUDE_CALL_CAP, concurrency: int = pg.DEFAULT_CONCURRENCY,
              say: Callable[[str], None] = print, sleep: Callable[[float], None] = time.sleep) -> Path:
    started = datetime.now(timezone.utc).isoformat()
    if is_dirty(root):
        raise Refused("the working tree has uncommitted changes; commit first, or the record cannot be written")
    freeze = committed_freeze(root)
    commit = j6.resolve_commit(root, from_commit)
    require_first_test_run(root, commit, override_note)
    changed = j6.card_code_changed(root, commit)
    if changed:
        raise Refused(f"the prompts or card code changed between {commit[:12]} and HEAD: {changed[:8]}")
    loaded = {src: load_verified(src, SPLIT, root) for src in ("scenarios", "qmsum", "decode")}
    cards, lme = split_cards(loaded["scenarios"][0])
    rows = {n: load_part(root, commit, "cards", n, cards) for n in SYSTEMS}
    extra_rows: dict[tuple[str, str], list[Prediction]] = {
        ("search:qmsum", SEARCH_SYSTEM): load_part(root, commit, "search:qmsum", SEARCH_SYSTEM, loaded["qmsum"][0]),
        (f"search:{LME}", SEARCH_SYSTEM): load_part(root, commit, f"search:{LME}", SEARCH_SYSTEM, lme)}
    detectors = {"scenarios": cards, "decode": loaded["decode"][0]}
    for source, scs in detectors.items():
        for n in ("trained", freeze["detectors"][source]):
            extra_rows[(f"detector:{source}", n)] = load_part(root, commit, f"detector:{source}", n, scs)

    if guard.limit != freeze["judging"]["max_weekly_pct"]:
        raise Refused(f"the weekly stop is {guard.limit:g}% but the freeze says {freeze['judging']['max_weekly_pct']:g}%; it is not a choice")
    start_pct = guard.start()
    header = {"kind": "e8-judge-rows", "commit": commit, "rubric": e2.RUBRIC_VERSION}
    stem = commit[:12]
    items = items_of(rows)
    if not items:
        raise Refused("no card made a claim, so there is nothing to judge")
    h_verdicts = j6.grade(items, haiku, root / JUDGE_DIR_REL / f"{stem}.haiku.rows.jsonl", {**header, "model": haiku.model}, guard,
                          counter_path=counter_path, cap=cap, concurrency=concurrency, say=say, sleep=sleep)
    chosen = set(sonnet_sample([i.id for i in items], freeze["judging"]["sonnet_sample"], freeze["judging"]["sample_seed"]))
    s_items = [i for i in items if i.id in chosen]
    s_verdicts = j6.grade(s_items, sonnet, root / JUDGE_DIR_REL / f"{stem}.sonnet.rows.jsonl", {**header, "model": sonnet.model}, guard,
                          counter_path=counter_path, cap=cap, concurrency=concurrency, say=say, sleep=sleep)
    end_pct = guard.final()

    judged = with_claims(rows, h_verdicts, "claim_supported")
    labels = label_text(root, "scenarios")
    metrics, intervals, comparisons = e3.card_stats(cards, judged)
    sonnet_view: dict[str, Any] = {
        "sample": freeze["judging"]["sonnet_sample"], "claims_graded": len(s_verdicts), "of_claims": len(items),
        "agreement_with_haiku": agreement(h_verdicts, s_verdicts),
        "note": ("Haiku graded every claim; Sonnet graded a seeded random sample "
                 f"(seed {freeze['judging']['sample_seed']}), so Sonnet's figure is agreement with Haiku on the sample, not a second score. "
                 "More Sonnet claims may be graded later by the separate `sonnet-extra` step, which keeps this figure and reports the extra apart.")}
    detectors_out = {s: detector_block(s, scs, {n: extra_rows[(f"detector:{s}", n)] for n in ("trained", freeze["detectors"][s])},
                                       freeze["detectors"][s]) for s, scs in detectors.items()}
    benchmarks = {"qmsum": search_block(loaded["qmsum"][0], extra_rows[("search:qmsum", SEARCH_SYSTEM)]),
                  LME: search_block(lme, extra_rows[(f"search:{LME}", SEARCH_SYSTEM)]),
                  "decode": detectors_out["decode"], "consistency_scenarios": detectors_out["scenarios"]}
    extra_files = {}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    (root / RUNS_REL).mkdir(parents=True, exist_ok=True)
    for (part, system), ps in extra_rows.items():
        path = root / RUNS_REL / f"{EXTRA_PREFIX}-{stamp}.{part.replace(':', '-')}-{system}.predictions.jsonl"
        write_atomic(path, "".join(p.model_dump_json() + "\n" for p in ps))
        extra_files[f"{part}/{system}"] = {"path": path.relative_to(root).as_posix(), "sha256": sha256_path(path)}

    n_claims = len(items)
    config: dict[str, Any] = {
        "experiment": EXPERIMENT, "freeze": {"path": FREEZE_REL.as_posix(), "sha256": sha256_path(root / FREEZE_REL), **freeze},
        "generating_commit": commit, "systems": list(SYSTEMS), "primary": PRIMARY, "prompt_note": f"prompt {PROMPT} for every system",
        "cards_subset": freeze["cards_subset"], "cards_cases": len(cards), "search_only_cases": {LME: len(lme), "qmsum": len(loaded["qmsum"][0])},
        "written_score": {n: e3.score(m) for n, m in metrics.items()}, "primary_vs_single": primary_vs_single(comparisons),
        "judge": {"model": haiku.model, "rubric_version": e2.RUBRIC_VERSION, "same_model_and_prompt_as_writer": False, "via": "claude -p, one claim per call, in parallel", "claims": n_claims,
                  "share_supported": sum(h_verdicts.values()) / n_claims, "calibration": "failed the written test (entry 56)"},
        "judge_note": CAL_NOTE, "sonnet": {"model": sonnet.model, **sonnet_view},
        "benchmarks": benchmarks, "extra_rows": extra_files,
        "usage": {"start_pct": start_pct, "end_pct": end_pct, "max_weekly_pct": MAX_WEEKLY_PCT},
        "notes": [CAL_NOTE, "answer_when_answerable counts cards that did not abstain, not cards that answered correctly",
                  "Public benchmarks are reported separately under config.benchmarks; their rows are in config.extra_rows and are "
                  "not recomputed by check_run.", labels,
                  "The trained-model team is T6.2's adapter, which may lose to the prompted writer; its result stands as measured."],
        "claude_calls": n_claims + len(s_verdicts), "started_at": started, "run_conditions": "judging only, no GPU"}
    if override_note:
        config["override_note"] = override_note
    names = e2.Models()
    attempts = [{"setting": n, "kind": "card system", "ok": True, "error": None,
                 "health": e2.check_health(n, rows[n])} for n in SYSTEMS]
    return write_run(EXPERIMENT, SPLIT, config, {"writer": names.writer, "judge": haiku.model, "cloud_grader": sonnet.model,
                                                 "embedder": names.embedder}, {"recall": PROMPT}, metrics, intervals, attempts, judged,
                     loaded["scenarios"][1], comparisons=comparisons, scenario_source="scenarios", subset=SUBSET_NAME, root=root)


def extra_report(haiku: dict[str, bool], original: dict[str, bool], extra: dict[str, bool]) -> dict[str, Any]:
    """The original sample's figure unchanged, the extra claims' figure, and the two together, each against Haiku."""
    return {"original": agreement(haiku, original), "extra": agreement(haiku, extra), "combined": agreement(haiku, {**original, **extra}),
            "note": "`original` is the figure in the E8 record and is not changed; `extra` covers only the further claims graded after "
                    "the weekly reset; `combined` is for reading only. This addendum does not replace the record's figure."}


def saved_verdicts(path: Path) -> dict[str, bool]:
    """Claim id to verdict from a judge rows file (the last row of a claim wins)."""
    out: dict[str, bool] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                out[row["claim_id"]] = bool(row["cloud"])
            except (ValueError, KeyError, TypeError):
                continue
    return out


def run_sonnet_extra(*, root: Path = ROOT, from_commit: str, sonnet: Any, more: int, guard: UsageGuard,
                     counter_path: Path = CLAUDE_CALLS_PATH, cap: int = CLAUDE_CALL_CAP, concurrency: int = pg.DEFAULT_CONCURRENCY,
                     say: Callable[[str], None] = print, sleep: Callable[[float], None] = time.sleep) -> Path:
    """Grade the next `more` claims of the frozen seeded order with Sonnet, after the E8 record exists. The record is not touched:
    an addendum file holds the original figure, the extra's, and both together."""
    freeze = committed_freeze(root)
    if guard.limit != freeze["judging"]["max_weekly_pct"]:
        raise Refused("the weekly stop is not the frozen one")
    commit = j6.resolve_commit(root, from_commit)
    records = existing_test_records(root)
    if not records:
        raise Refused("there is no E8 record yet; run `judge` first")
    cards, _ = split_cards(load_verified("scenarios", SPLIT, root)[0])
    items = items_of({n: load_part(root, commit, "cards", n, cards) for n in SYSTEMS})
    seed, done = freeze["judging"]["sample_seed"], freeze["judging"]["sonnet_sample"]
    chosen = set(sonnet_extra([i.id for i in items], seed, done, more))
    if not chosen:
        raise Refused(f"no more claims: all {len(items)} claims are already in the original sample of {done}")
    stem = commit[:12]
    header = {"kind": "e8-judge-rows", "commit": commit, "rubric": e2.RUBRIC_VERSION, "model": sonnet.model}
    guard.start()
    verdicts = j6.grade([i for i in items if i.id in chosen], sonnet, root / EXTRA_DIR_REL / f"{stem}.sonnet-extra.rows.jsonl", header,
                        guard, counter_path=counter_path, cap=cap, concurrency=concurrency, say=say, sleep=sleep)
    out = {"record": records[-1], "generating_commit": commit, "extra_requested": more, "extra_graded": len(verdicts), "seed": seed,
           "weekly_usage_pct": guard.final(),
           **extra_report(saved_verdicts(root / JUDGE_DIR_REL / f"{stem}.haiku.rows.jsonl"),
                          saved_verdicts(root / JUDGE_DIR_REL / f"{stem}.sonnet.rows.jsonl"), verdicts)}
    path = root / EXTRA_DIR_REL / f"{stem}.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    write_atomic(path, json.dumps(out, indent=1) + "\n")
    return path


# ---------------------------------------------------------------- command line

def _real_hf_writer(adapter: dict[str, Any]) -> LLMClient:
    from contextlens.llm.hf import HFClient  # T6.2's client; absent until that task lands

    return HFClient(adapter["base_model_path"], adapter["adapter_path"])


def _real_checker(name: str, checker: dict[str, Any]) -> ConsistencyChecker:
    from contextlens.agents.consistency_check import PromptedChecker
    from contextlens.consistency import model as cm
    from contextlens.consistency.baselines import TfidfChecker
    from contextlens.data.scenarios import load_scenarios
    from contextlens.llm.client import OllamaClient

    if name == "trained":
        return cm.TrainedChecker.from_checkpoint(Path(checker["path"]), threshold=checker["threshold"], strip_instructions=False)
    if name == "prompted":
        return PromptedChecker(OllamaClient(e2.WRITER_MODEL))
    if name == "tfidf":
        dev = load_scenarios("dev", ROOT / SCENARIO_SOURCES["scenarios"][1])
        return TfidfChecker().fit(e4.training_data(ROOT / e4.RAW_DIR, ROOT, dev).pairs)
    raise Refused(f"no detector named {name!r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze", help="write configs/e8_freeze.json (then commit it)")
    f.add_argument("--adapter-base", type=Path, required=True)
    f.add_argument("--adapter", type=Path, required=True)
    f.add_argument("--checker", type=Path, required=True)
    f.add_argument("--checker-threshold", type=float, default=0.5)
    w = sub.add_parser("write", help="GPU: run the test and save rows")
    w.add_argument("--run-conditions", required=True)
    w.add_argument("--override-note")
    jd = sub.add_parser("judge", help="no GPU: grade the saved claims and write the record")
    jd.add_argument("--from-commit", required=True, help="the commit that wrote the rows")
    jd.add_argument("--override-note")
    jd.add_argument("--concurrency", type=int, default=pg.DEFAULT_CONCURRENCY)
    ex = sub.add_parser("sonnet-extra", help="no GPU: grade more Sonnet claims after the E8 record, apart from it (additive)")
    ex.add_argument("--from-commit", required=True)
    ex.add_argument("--more", type=int, required=True, help="how many further claims")
    ex.add_argument("--concurrency", type=int, default=pg.DEFAULT_CONCURRENCY)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "freeze":
            path = write_freeze(ROOT, build_freeze(ROOT, adapter_base=args.adapter_base, adapter=args.adapter, checker=args.checker,
                                                   checker_threshold=args.checker_threshold))
            print(f"wrote {path.as_posix()}; commit it before `write`")
            return 0
        if args.cmd == "write":
            from contextlens.llm.client import Embedder, OllamaClient

            counts = run_write(writer=OllamaClient(e2.WRITER_MODEL), embedder=Embedder(e2.EMBED_MODEL), ctl=OllamaCtl(),
                               make_hf_writer=_real_hf_writer, make_checker=_real_checker, run_conditions=args.run_conditions,
                               override_note=args.override_note)
            print(json.dumps(counts, indent=1))
            return 0
        if args.cmd == "sonnet-extra":
            out = run_sonnet_extra(from_commit=args.from_commit, sonnet=pg.CountedClaude(model=e2.CLOUD_MODEL), more=args.more,
                                   guard=UsageGuard(MAX_WEEKLY_PCT), concurrency=args.concurrency)
            print(out.as_posix())
            return 0
        rec = run_judge(from_commit=args.from_commit, haiku=pg.CountedClaude(), sonnet=pg.CountedClaude(model=e2.CLOUD_MODEL),
                        guard=UsageGuard(MAX_WEEKLY_PCT), override_note=args.override_note, concurrency=args.concurrency)
        print(rec.as_posix())
        return 0
    except UsageUnreadable as exc:
        print(f"refused: weekly usage cannot be read ({exc})", file=sys.stderr)
    except (Refused, e2.RunBroken, j6.JudgeStopped) as exc:
        print(f"E8 {args.cmd} stopped, nothing more written: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
