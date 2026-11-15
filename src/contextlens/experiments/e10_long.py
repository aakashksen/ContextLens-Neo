"""E10: do agent teams help on long meetings? Runs the arms of configs/e10.json on the frozen long-meeting cases and
scores every card in code (contextlens.eval.e10_metrics). No judge and no API: the grader is a separate step (e10_grade).

Every arm is built by the product's own factory, `production.build_services(assistant=<name>)`, so an arm exists here only
if `production.ASSISTANTS` lists it; a missing arm stops the run before any model call, with the name.

  .venv/Scripts/python -m contextlens.experiments.e10_long run --split dev            # all arms of the config
  .venv/Scripts/python -m contextlens.experiments.e10_long run --split test --run-conditions "<what else was running>"
  .venv/Scripts/python -m contextlens.experiments.e10_long run --split dev --smoke --arms single,team_shipping_changed
  .venv/Scripts/python -m contextlens.experiments.e10_long report --split dev        # record from saved rows, no model

Rows are saved the moment they exist (git-ignored results/partial/e10-rows.jsonl) and a rerun skips what is saved, so a
stopped run resumes. The record (ids, booleans and numbers, no meeting text) goes to results/e10/; the cards stay local.
"""

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contextlens.api import production
from contextlens.data import e10_cases
from contextlens.eval import e10_metrics as M
from contextlens.eval.records import ROOT, head_commit, is_dirty, write_atomic
from contextlens.llm.client import LLMClient, OllamaClient
from contextlens.memory.extract import MemoryExtractor, MemoryStore
from contextlens.memory.store import InMemoryTranscriptStore
from contextlens.replay.runner import Prediction, replay
from contextlens.retrieval.query import query_for
from contextlens.schemas import MemoryItem, Scenario

EXPERIMENT = "E10"
CONFIG_REL = Path("configs/e10.json")
ROWS_REL = Path("results/partial/e10-rows.jsonl")
MEMORY_REL = Path("results/partial/e10-memory.jsonl")
RECORD_DIR = Path("results/e10")
SMOKE_DIR = Path("results/local/e10-smoke")
LOCAL_DIR = Path("results/local/e10")
SECONDARY = ("cite_all", "precision", "catch_contradiction", "catch_update", "flag_both_rate", "false_alert_rate",
             "conflict_label_accuracy", "abstain_when_insufficient", "answer_when_answerable", "twin_answerable_cite_any",
             "deadline_cut_rate", "tokens", "latency_ms", "control_label_accuracy", "control_abstain", "control_answer",
             "control_mode_accuracy")
SMOKE_PER_TYPE = {"qmsum": 3, "conflict": 3, "twin": 2, "control": 2}


class ArmMissing(RuntimeError):
    """An arm the config names is not registered in production.ASSISTANTS."""


def load_config(root: Path = ROOT) -> dict[str, Any]:
    return json.loads((root / CONFIG_REL).read_text(encoding="utf-8"))


def config_sha(cfg: dict[str, Any]) -> str:
    keep = {k: cfg[k] for k in ("arms", "prompt_version", "deadline_s", "writer_model", "embedder_model", "memory")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()


def check_arms(arms: list[str]) -> None:
    missing = [a for a in arms if a not in production.ASSISTANTS]
    if missing:
        raise ArmMissing(f"arm(s) {missing} are not in production.ASSISTANTS {production.ASSISTANTS}; "
                         "merge the branch that registers them (feat/team-routed) before running E10")


def check_pins(cfg: dict[str, Any], root: Path) -> None:
    """The run uses the frozen model, prompt and deadline, so a drifted constant cannot slip into the record."""
    now = {"prompt_version": production.PROMPT_VERSION, "deadline_s": production.DEADLINE_S, "writer_model": production.MODEL,
           "embedder_model": production.EMBED_MODEL}
    off = {k: (cfg[k], v) for k, v in now.items() if cfg[k] != v}
    if off:
        raise RuntimeError(f"configs/e10.json and production disagree (config, production): {off}")
    if e10_cases.manifest_sha(root) != cfg["cases"]["manifest_sha256"]:
        raise RuntimeError("datasets/splits/e10-scenarios.json is not the manifest that configs/e10.json froze")


# ------------------------------------------------------------------ durable rows

def _append(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cut = False
    if path.is_file() and path.stat().st_size:
        with open(path, "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            cut = fh.read(1) != b"\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(("\n" if cut else "") + line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def saved_rows(root: Path, key: str, split: str) -> dict[str, dict[str, Prediction]]:
    """arm -> case id -> row, for this config key and split. A line cut short by a crash is skipped; a row that errored is rerun."""
    out: dict[str, dict[str, Prediction]] = {}
    path = root / ROWS_REL
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
            if e["key"] != key or e["split"] != split:
                continue
            row = Prediction.model_validate(e["row"])
        except (ValueError, KeyError, TypeError):
            continue
        if not row.errored:
            out.setdefault(e["arm"], {})[row.scenario_id] = row
    return out


# ------------------------------------------------------------------ memory (the notes a running product would have)

def _store(sc: Scenario) -> InMemoryTranscriptStore:
    store = InMemoryTranscriptStore()
    for t in sorted(sc.conversation.turns, key=lambda t: t.seq):
        if t.seq <= sc.request.as_of_seq:
            store.add(t)
    return store


def memory_key(sc: Scenario, chunk: int, model: str) -> str:
    h = hashlib.sha256(f"{model}|{chunk}|{sc.request.as_of_seq}".encode())
    for t in sc.conversation.turns:
        h.update(f"\x1f{t.speaker}\x1e{t.text}".encode())
    return h.hexdigest()


def build_memory(sc: Scenario, llm: LLMClient, chunk: int, root: Path) -> tuple[list[MemoryItem], dict[str, Any]]:
    """Memory built the way the live stream builds it: the extractor reads the turns in chunks, each chunk seeing what is already
    known. Cached on disk by the transcript's words (QMSum questions on one meeting share it); a failed chunk is skipped and counted.
    Building is outside every card's timing, as in the product, where it runs in the background."""
    key = memory_key(sc, chunk, production.MODEL)
    path, cid = root / MEMORY_REL, sc.conversation.id
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("key") == key:
                items = [MemoryItem.model_validate(i) for i in json.loads(e["items"].replace("\x00CID\x00", cid))]
                return items, {"cached": True, "ms": 0.0, "failed_chunks": e["failed_chunks"], "chunks": e["chunks"]}
    turns = _store(sc).turns(cid, sc.request.as_of_seq)
    extractor, items, failed, t0 = MemoryExtractor(llm), [], 0, time.perf_counter()
    chunks = [turns[i:i + chunk] for i in range(0, len(turns), chunk)]
    for part in chunks:
        try:
            new = extractor.update(part, items)
        except Exception as exc:
            failed += 1
            print(f"memory chunk failed for {sc.id}: {type(exc).__name__}", file=sys.stderr)
            continue
        items.extend(new)
    blob = json.dumps([i.model_dump(mode="json") for i in items]).replace(cid, "\x00CID\x00")
    _append(path, json.dumps({"key": key, "items": blob, "failed_chunks": failed, "chunks": len(chunks)}))
    return items, {"cached": False, "ms": (time.perf_counter() - t0) * 1000, "failed_chunks": failed, "chunks": len(chunks)}


# ------------------------------------------------------------------ one card

def run_case(sc: Scenario, arm: str, writer: LLMClient, embedder: Any, items: list[MemoryItem], deadline_s: float) -> Prediction:
    """One press of `arm`, built by the product's factory over this case's transcript, with the memory a running product would hold.
    Indexing the turns happens before the timing starts."""
    prod = production.build_services(
        store=_store(sc), embedder=embedder, llms={"memory": writer, "consistency": writer, "assistance": writer},
        background_llm=None, speech=None, assistant=arm, deadline_s=deadline_s)
    try:
        prod.services.memory.add(items)
        query, last = query_for(sc)
        if last >= 0:
            prod.services.retriever.search(sc.conversation.id, query, last, k=1, mode="dense")
        inner = prod.traced.inner
        return replay([sc], lambda: inner)[0]
    finally:
        pool = getattr(prod.traced, "_pool", None)
        if pool is not None:
            pool.shutdown(wait=False)


def pick_smoke(scs: list[Scenario], meta: dict[str, dict[str, Any]]) -> list[Scenario]:
    """The first few of each type, in file order; conflict cases are taken as a whole triplet's three."""
    left, out = dict(SMOKE_PER_TYPE), []
    for s in scs:
        t = meta[s.id]["type"]
        if left[t] > 0:
            out.append(s)
            left[t] -= 1
    return out


# ------------------------------------------------------------------ run and report

def run_key(root: Path, cfg: dict[str, Any], commit: str | None = None) -> str:
    """Rows are reused only at the same config and the same commit, so changed agent code never mixes with older rows."""
    return f"{config_sha(cfg)}:{(commit or head_commit(root))[:12]}"


def run(root: Path, split: str, arms: list[str], *, smoke: bool, limit: int | None, run_conditions: str, say=print) -> dict[str, Any]:
    cfg = load_config(root)
    check_arms(arms)
    check_pins(cfg, root)
    scs, meta, digest = e10_cases.load_cases(split, root)
    if smoke:
        scs = pick_smoke(scs, meta)
    if limit:
        scs = scs[:limit]
    key, rsplit = run_key(root, cfg), f"smoke-{split}" if smoke else split
    have = saved_rows(root, key, rsplit)
    from contextlens.experiments.e5_cards import EmbedCache
    from contextlens.llm.client import Embedder

    embedder = EmbedCache(Embedder(base_url=production.OLLAMA_URL, model=production.EMBED_MODEL, keep_alive=production.KEEP_ALIVE))
    writer = OllamaClient(production.MODEL, production.OLLAMA_URL, keep_alive=production.KEEP_ALIVE)
    chunk, timing = cfg["memory"]["chunk_turns"], {"memory_ms": 0.0, "memory_failed_chunks": 0, "memory_cases_built": 0, "cards_ms": 0.0}
    t_start = time.perf_counter()
    done = Counter()
    try:
        for i, sc in enumerate(scs, 1):
            todo = [a for a in arms if sc.id not in have.get(a, {})]
            if not todo:
                continue
            items, info = build_memory(sc, writer, chunk, root)
            timing["memory_ms"] += info["ms"]
            timing["memory_failed_chunks"] += info["failed_chunks"]
            timing["memory_cases_built"] += not info["cached"]
            for arm in todo:
                t0 = time.perf_counter()
                row = run_case(sc, arm, writer, embedder, items, cfg["deadline_s"])
                timing["cards_ms"] += (time.perf_counter() - t0) * 1000
                _append(root / ROWS_REL, json.dumps({"key": key, "split": rsplit, "arm": arm, "cases_hash": digest,
                                                     "row": json.loads(row.model_dump_json())}))
                if not row.errored:
                    have.setdefault(arm, {})[sc.id] = row
                done[arm] += 1
            if i % 5 == 0 or i == len(scs):
                say(f"  {i}/{len(scs)} cases, {sum(done.values())} cards run this session")
    finally:
        if os.environ.get("E10_KEEP_LOADED") != "1":
            production.unload_local_models()
    timing["wall_s"] = time.perf_counter() - t_start
    timing["cards_run"] = dict(done)
    path = write_report(root, split, arms, scs, meta, digest, key, cfg, smoke=smoke, run_conditions=run_conditions, timing=timing)
    say(f"wrote {path}")
    return {"record": path, "timing": timing}


def tables_from_rows(scs: list[Scenario], meta: dict[str, dict[str, Any]], rows: dict[str, dict[str, Prediction]],
                     arms: list[str]) -> dict[str, list[dict[str, Any]]]:
    groups = {s.id: meta[s.id]["meeting"] for s in scs}
    return {a: M.score_arm(scs, list(rows.get(a, {}).values()), groups) for a in arms if rows.get(a)}


def compare_all(tables: dict[str, list[dict[str, Any]]], resamples: int = 5000) -> list[dict[str, Any]]:
    out = []
    for a, b in itertools.combinations(tables, 2):
        for m in (*M.PRIMARIES, *SECONDARY):
            c = M.compare(tables[a], tables[b], m, resamples)
            if c:
                out.append({"a": a, "b": b, **c})
    return out


def write_report(root: Path, split: str, arms: list[str], scs: list[Scenario], meta: dict[str, dict[str, Any]], digest: str, key: str,
                 cfg: dict[str, Any], *, smoke: bool, run_conditions: str, timing: dict[str, Any] | None = None,
                 resamples: int = 5000) -> Path:
    rows = saved_rows(root, key, f"smoke-{split}" if smoke else split)
    ids = {s.id for s in scs}
    rows = {a: {i: p for i, p in r.items() if i in ids} for a, r in rows.items()}
    tables = tables_from_rows(scs, meta, rows, arms)
    now = datetime.now(timezone.utc)
    run_id = f"{EXPERIMENT}-{'S' if smoke else ''}{split}-{now.strftime('%Y%m%dT%H%M%S%fZ')}"
    local = root / (SMOKE_DIR if smoke else LOCAL_DIR)
    for a, r in rows.items():  # the cards, with meeting text: local only
        write_atomic(local / f"{run_id}.{a}.predictions.jsonl", "".join(p.model_dump_json() + "\n" for p in r.values()))
    types = Counter(meta[s.id]["type"] for s in scs)
    record = {
        "run_id": run_id, "experiment": EXPERIMENT, "split": split, "smoke": smoke,
        "git_commit": head_commit(root), "dirty": is_dirty(root), "rows_key": key, "config_sha256": config_sha(cfg), "cases_hash": digest,
        "scenario_source": "e10", "labels_reviewed": False, "label_source": "e10_templated",
        "label_note": "Conflict and twin statements are invented from templates and no person has read a sample of them yet; "
                      "qmsum_long labels are the publisher's. Claims on conflict and twin metrics wait for that read.",
        "arms": arms, "n_cases": len(scs), "cases_by_type": dict(types), "run_conditions": run_conditions,
        "models": {"writer": production.MODEL, "embedder": production.EMBED_MODEL}, "prompt_version": production.PROMPT_VERSION,
        "deadline_s": production.DEADLINE_S, "timing": timing or {}, "claude_calls": 0,
        "n_cards": {a: len(t) for a, t in tables.items()},
        "metrics": {a: M.summarize(t, 2000) for a, t in tables.items()},
        "comparisons": compare_all(tables, resamples),
        "claim_verdict": M.claim_verdict(tables), "per_case": tables,
        "started_at": now.isoformat(),
    }
    path = root / (SMOKE_DIR if smoke else RECORD_DIR) / f"{run_id}.json"
    write_atomic(path, json.dumps(record, indent=1) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "report"])
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--arms", help="comma list; default: the arms of configs/e10.json")
    ap.add_argument("--smoke", action="store_true", help="a few cases of each type; the record goes to git-ignored results/local")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--commit", help="report: the commit whose saved rows to use (default: HEAD)")
    ap.add_argument("--run-conditions", default="", help="what else was running while latency was measured")
    args = ap.parse_args(argv)
    cfg = load_config(ROOT)
    arms = args.arms.split(",") if args.arms else cfg["arms"]
    if not set(arms) <= set(cfg["arms"]):
        ap.error(f"arms must come from configs/e10.json: {cfg['arms']}")
    if args.command == "run":
        if not args.run_conditions.strip() and not args.smoke:
            ap.error("--run-conditions is required: say what else was running while latency was measured")
        run(ROOT, args.split, arms, smoke=args.smoke, limit=args.limit, run_conditions=args.run_conditions)
    else:
        check_pins(cfg, ROOT)
        scs, meta, digest = e10_cases.load_cases(args.split, ROOT)
        if args.smoke:
            scs = pick_smoke(scs, meta)
        print(write_report(ROOT, args.split, arms, scs, meta, digest, run_key(ROOT, cfg, args.commit), cfg, smoke=args.smoke,
                           run_conditions=args.run_conditions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
