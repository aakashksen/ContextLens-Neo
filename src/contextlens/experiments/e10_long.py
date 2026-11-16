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
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contextlens.api import production
from contextlens.data import e10_cases
from contextlens.eval import e10_metrics as M
from contextlens.eval.records import ROOT, head_commit, is_dirty, write_atomic
from contextlens.llm.client import LLMClient, LLMError, OllamaClient
from contextlens.memory.extract import MemoryExtractor
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


def load_config(root: Path = ROOT, rel: Path | None = None) -> dict[str, Any]:
    return json.loads((root / (rel or CONFIG_REL)).read_text(encoding="utf-8"))


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


def memory_key(sc: Scenario, chunk: int, model: str, num_ctx: int | None = None) -> str:
    """Meeting words plus everything that shapes the notes (model, chunking, context size, extraction prompt and token cap).
    Not the commit and not the case id, so memory built now survives later commits and is shared by questions on one meeting."""
    from contextlens.memory import extract as X

    prompt = hashlib.sha256(f"{X.SYSTEM}|{X.EXTRACT_MAX_TOKENS}".encode()).hexdigest()[:16]
    h = hashlib.sha256(f"{model}|{chunk}|{num_ctx}|{prompt}|{sc.request.as_of_seq}".encode())
    for t in sc.conversation.turns:
        h.update(f"\x1f{t.speaker}\x1e{t.text}".encode())
    return h.hexdigest()


def _entry(root: Path, key: str) -> dict[str, Any] | None:
    """The newest saved entry for this key (a redo appends a newer one), or None."""
    path, found = root / MEMORY_REL, None
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if f'"key": "{key}"' in line[:120]:
                try:
                    found = json.loads(line)
                except ValueError:
                    continue
    return found


def _cached(root: Path, key: str) -> bool:
    """A finished entry: every chunk produced notes or was read successfully. An entry with failed chunks is not finished."""
    e = _entry(root, key)
    return bool(e) and e["failed_chunks"] == 0


class Patient:
    """The memory writer, for extraction. Three differences from the plain client:
    - a call gets at least `timeout_s` (the extractor asks for 60 s; a call abandoned on our side keeps running inside Ollama and
      queues behind the next ones, so a short timeout on a busy GPU turns one slow chunk into a pile of failures);
    - no second generation after bad JSON: the client's schema retry repeats the same long request, which is how a truncated reply
      cost 2 x 130 s and still failed;
    - a reply cut off at the token cap keeps its complete items instead of being thrown away (`salvage_items`).
    `last` describes the latest call (completion tokens, whether the cap was hit, how many items were kept) for the progress log."""

    def __init__(self, inner: LLMClient, timeout_s: float) -> None:
        self.inner, self.timeout_s, self.last = inner, timeout_s, {}

    def complete(self, messages: list[dict], **kw: Any) -> Any:
        kw["timeout_s"] = max(kw.get("timeout_s") or 0.0, self.timeout_s)
        schema = kw.get("schema")
        raw = getattr(self.inner, "_raw", None)
        if schema is None or raw is None:
            return self.inner.complete(messages, **kw)
        cap = kw.get("max_tokens", 512)
        res = raw(messages, schema, cap, kw.get("temperature", 0.0), kw["timeout_s"])
        self.last = {"completion_tokens": res.completion_tokens, "hit_cap": res.completion_tokens >= cap, "salvaged": False}
        try:
            res.parsed = schema.model_validate_json(res.text.strip())
            return res
        except ValueError:
            kept = salvage_items(res.text, schema)
        if kept is None:
            raise LLMError(f"invalid model reply ({res.completion_tokens} tokens, cap {cap}, no complete item)", results=[res])
        self.last.update(salvaged=True, kept=len(kept.items))
        res.parsed = kept
        return res


def salvage_items(text: str, schema: Any) -> Any:
    """The complete items of a reply of the form {"items": [ {...}, {...}, <cut off>: each object that parses and validates, in
    order, up to the first one that does not. None when there is none."""
    start = text.find("[", text.find('"items"'))
    if start < 0:
        return None
    dec, pos, kept = json.JSONDecoder(), start + 1, []
    item_type = schema.model_fields["items"].annotation.__args__[0]
    while pos < len(text):
        while pos < len(text) and text[pos] in " \n\r\t,":
            pos += 1
        if pos >= len(text) or text[pos] == "]":
            break
        try:
            obj, pos = dec.raw_decode(text, pos)
            kept.append(item_type.model_validate(obj))
        except ValueError:
            break
    return schema(items=kept) if kept else None


def split_chunk(part: list[Any], max_chars: int) -> list[list[Any]]:
    """A chunk of turns cut into runs of at most `max_chars` of text (a single longer turn stays alone), in order. The extractor's
    reply is about as long as its input, and the reply is capped, so a chunk of very long turns must be read in pieces."""
    out, cur, size = [], [], 0
    for t in part:
        n = len(t.text) + len(t.speaker) + len(t.id) + 8
        if cur and size + n > max_chars:
            out.append(cur)
            cur, size = [], 0
        cur.append(t)
        size += n
    return [*out, cur] if cur else out


def _why(exc: Exception, llm: Any) -> str:
    """One line for the log: the exception type, and the model call's numbers. No transcript or model text."""
    last = getattr(llm, "last", {}) or {}
    msg = str(exc)
    safe = msg[:120] if msg.startswith(("request to ", "invalid model reply", "too many")) else ""
    bits = [type(exc).__name__, safe, f"completion_tokens={last.get('completion_tokens')}", f"hit_cap={last.get('hit_cap')}"]
    return " ".join(b for b in bits if b)


def build_memory(sc: Scenario, llm: LLMClient, chunk: int, root: Path, num_ctx: int | None = None,
                 on_chunk: Callable[[int, int, float], None] | None = None, attempts: int = 2,
                 max_chunk_chars: int | None = None, on_fail: Callable[[str], None] | None = None) -> tuple[list[MemoryItem], dict[str, Any]]:
    """Memory built the way the live stream builds it: the extractor reads the turns in chunks of `chunk` turns, each piece seeing
    the notes of the text before it; a chunk whose text is longer than `max_chunk_chars` is read in pieces (split_chunk). Cached on
    disk by the transcript's words (QMSum questions on one meeting share it). A chunk that still fails after `attempts` tries is
    recorded as failed and the entry is NOT finished: the next call redoes only the failed chunks, dropping their partial notes, and
    keeps the rest. Building is outside every card's timing, as in the product, where it runs in the background."""
    key = memory_key(sc, chunk, production.MODEL, num_ctx)
    path, cid = root / MEMORY_REL, sc.conversation.id
    turns = _store(sc).turns(cid, sc.request.as_of_seq)
    chunks = [turns[i:i + chunk] for i in range(0, len(turns), chunk)]
    entry = _entry(root, key)
    items: list[MemoryItem] = []
    if entry:
        items = [MemoryItem.model_validate(i) for i in json.loads(entry["items"].replace("\x00CID\x00", cid))]
        if entry["failed_chunks"] == 0:
            return items, {"cached": True, "ms": 0.0, "failed_chunks": 0, "chunks": entry["chunks"], "redone": 0}
        have = {i.as_of_seq for i in items}
        todo = entry.get("failed") or [n for n, part in enumerate(chunks) if part[-1].seq not in have]  # old entries: chunks with no notes
    else:
        todo = list(range(len(chunks)))
    extractor, failed, t0, kept_partial = MemoryExtractor(llm), [], time.perf_counter(), 0
    for n in todo:
        part, t1 = chunks[n], time.perf_counter()
        lo, hi = part[0].seq, part[-1].seq
        items = [i for i in items if not lo <= i.as_of_seq <= hi]  # a redo starts the chunk afresh
        ok = True
        for piece in (split_chunk(part, max_chunk_chars) if max_chunk_chars else [part]):
            before = [i for i in items if i.as_of_seq < piece[0].seq]
            for attempt in range(attempts):
                try:
                    items.extend(extractor.update(piece, before))
                    kept_partial += bool(getattr(llm, "last", {}).get("salvaged"))
                    break
                except Exception as exc:
                    line = f"{sc.id} chunk {n + 1}/{len(chunks)} (try {attempt + 1}/{attempts}, {len(piece)} turns, " \
                           f"{sum(len(t.text) for t in piece)} chars): {_why(exc, llm)}"
                    print("memory chunk failed:", line, file=sys.stderr)
                    if on_fail:
                        on_fail(line)
            else:
                ok = False
                break
        if not ok:
            failed.append(n)
        if on_chunk:
            on_chunk(n + 1, len(chunks), time.perf_counter() - t1)
    items.sort(key=lambda i: i.as_of_seq)
    blob = json.dumps([i.model_dump(mode="json") for i in items]).replace(cid, "\x00CID\x00")
    _append(path, json.dumps({"key": key, "items": blob, "failed_chunks": len(failed), "failed": failed, "chunks": len(chunks)}))
    return items, {"cached": False, "ms": (time.perf_counter() - t0) * 1000, "failed_chunks": len(failed), "chunks": len(chunks),
                   "redone": len(todo) if entry else 0, "truncated_replies_kept": kept_partial}


def memory_client(cfg: dict[str, Any]) -> LLMClient:
    """The writer with the context size the memory prompts need (the default of 4096 tokens cuts the biggest QMSum chunks) and a
    long per-call timeout (see Patient)."""
    m = cfg["memory"]
    return Patient(OllamaClient(production.MODEL, production.OLLAMA_URL, keep_alive=production.KEEP_ALIVE, num_ctx=m.get("num_ctx")),
                   m.get("call_timeout_s", 900.0))


def prebuild(root: Path, splits: list[str], say: Callable[[str], None] = print) -> dict[str, Any]:
    """Memory only, no cards: build and cache the notes of every case of the splits, in order. Progress goes to
    results/local/e10/prebuild-<splits>.log (chunks done of total, seconds per chunk)."""
    cfg = load_config(root)
    check_pins(cfg, root)
    chunk, num_ctx, llm = cfg["memory"]["chunk_turns"], cfg["memory"].get("num_ctx"), memory_client(cfg)
    log_path = root / LOCAL_DIR / f"prebuild-{'-'.join(splits)}.log"

    def log(text: str) -> None:
        _append(log_path, f"{datetime.now().strftime('%H:%M:%S')} {text}")
        say(text)

    work, seen = [], set()
    for sp in splits:
        scs, _, _ = e10_cases.load_cases(sp, root)
        for sc in scs:
            k = memory_key(sc, chunk, production.MODEL, num_ctx)
            if k not in seen:
                seen.add(k)
                work.append((sp, sc, k))
    todo = [(sp, sc) for sp, sc, k in work if not _cached(root, k)]
    def chunks_left(sc: Scenario, k: str) -> int:
        e = _entry(root, k)
        n = -(-len(_store(sc).turns(sc.conversation.id, sc.request.as_of_seq)) // chunk)
        return e["failed_chunks"] if e else n

    total = sum(chunks_left(sc, memory_key(sc, chunk, production.MODEL, num_ctx)) for _, sc in todo)
    log(f"prebuild {splits}: {len(work)} distinct transcripts, {len(work) - len(todo)} cached, {len(todo)} to build, "
        f"{total} chunks, chunk_turns={chunk}, num_ctx={num_ctx}")
    done, secs, t0 = 0, 0.0, time.perf_counter()
    try:
        for i, (sp, sc) in enumerate(todo, 1):
            def on_chunk(n: int, of: int, s: float, sc=sc, sp=sp, i=i) -> None:
                nonlocal done, secs
                done, secs = done + 1, secs + s
                log(f"{sp} {sc.id} ({i}/{len(todo)}) chunk {n}/{of} {s:.1f}s | total {done}/{total} chunks, "
                    f"{secs / done:.1f} s/chunk, eta {(total - done) * secs / done / 60:.0f} min")

            _, info = build_memory(sc, llm, chunk, root, num_ctx, on_chunk, max_chunk_chars=cfg['memory'].get('max_chunk_chars'), on_fail=lambda m: log('FAILED ' + m))
            if info["failed_chunks"]:
                log(f"WARNING {sc.id}: {info['failed_chunks']} chunk(s) still failed; the entry stays unfinished and is redone next time")
    finally:
        if os.environ.get("E10_KEEP_LOADED") != "1":
            production.unload_local_models()
    log(f"prebuild done: {done} chunks in {(time.perf_counter() - t0) / 60:.1f} min")
    return {"chunks": done, "seconds": time.perf_counter() - t0}


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


def warm_up(cfg: dict[str, Any], arms: list[str], sc: Scenario | list[Scenario], writer: LLMClient, embedder: Any, say: Callable[[str], None]) -> None:
    """Before any timed press: point the cascade at the trained checker, then press every arm once on the first case and throw the
    cards away, so the models and the checker are loaded. Refuses to run an arm that needs the checker if it fell back to the
    prompted one (it costs an extra model call, so the timing would not be the product's)."""
    directory = Path(cfg["checker_dir"])
    os.environ.setdefault("CONTEXTLENS_CHECKER_DIR", str(directory))
    cases = sc if isinstance(sc, list) else [sc]
    if any(a.startswith("team_rag") for a in arms):
        production.preload_rag()
    for arm in arms:
        for one in cases:  # two presses: a first press on a fresh model can still run past the deadline (E11 smoke, 2026-11-16)
            t0 = time.perf_counter()
            run_case(one, arm, writer, embedder, [], cfg["deadline_s"])
            say(f"warm-up {arm} on {one.id}: {time.perf_counter() - t0:.1f} s (card discarded)")
    needs = [a for a in arms if a in ("team_routed", "single_plus")]
    if needs and production._TRAINED.get("first") is None:
        raise RuntimeError(f"{needs} need the trained checker and it did not load from {os.environ['CONTEXTLENS_CHECKER_DIR']}")
    rag = [a for a in arms if a.startswith("team_rag")]
    if rag and not production._TRAINED.get("pairing"):
        raise RuntimeError(f"{rag} need the pairwise conflict finder and its models did not load")


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


def run(root: Path, split: str, arms: list[str], *, smoke: bool, limit: int | None, run_conditions: str, say=print,
        config: Path | None = None) -> dict[str, Any]:
    cfg = load_config(root, config)
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
    writer = OllamaClient(production.MODEL, production.OLLAMA_URL, keep_alive=production.KEEP_ALIVE,
                          num_ctx=(cfg.get("card") or {}).get("num_ctx"))  # one setting for every arm
    mem_llm, num_ctx = memory_client(cfg), cfg["memory"].get("num_ctx")
    chunk, timing = cfg["memory"]["chunk_turns"], {"memory_ms": 0.0, "memory_failed_chunks": 0, "memory_cases_built": 0, "cards_ms": 0.0}
    t_start = time.perf_counter()
    done = Counter()
    log_path = root / LOCAL_DIR / f"run-{rsplit}.log"

    def say(text: str) -> None:
        _append(log_path, f"{datetime.now().strftime('%H:%M:%S')} {text}")
        print(text)

    try:
        warm_up(cfg, arms, scs[:2], writer, embedder, say)
        say(f"{rsplit}: {len(scs)} cases, arms {arms}, {sum(len(have.get(a, {})) for a in arms)} rows already saved")
        for i, sc in enumerate(scs, 1):
            todo = [a for a in arms if sc.id not in have.get(a, {})]
            if not todo:
                continue
            if cfg["memory"].get("cached_only") and not _cached(root, memory_key(sc, chunk, production.MODEL, num_ctx)):
                items, info = [], {"cached": True, "ms": 0.0, "failed_chunks": 0, "absent": True}  # no notes: every arm runs without them
                timing["memory_absent_cases"] = timing.get("memory_absent_cases", 0) + 1
            else:
                items, info = build_memory(sc, mem_llm, chunk, root, num_ctx, max_chunk_chars=cfg['memory'].get('max_chunk_chars'))
            if info["failed_chunks"]:
                raise RuntimeError(f"memory for {sc.id} is incomplete ({info['failed_chunks']} chunk(s) failed); run prebuild again "
                                   "rather than pressing on a meeting with missing notes")
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
            el = time.perf_counter() - t_start
            todo_left = sum(1 for s in scs[i:] for a in arms if s.id not in have.get(a, {}))
            run_n = max(1, sum(done.values()))
            say(f"{i}/{len(scs)} cases | {sum(done.values())} cards this session, {el / run_n:.1f} s/card, "
                f"eta {todo_left * el / run_n / 60:.0f} min | {sc.id} cut={sum(1 for a in arms if have.get(a, {}).get(sc.id) and M.deadline_cut(have[a][sc.id]))}")
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
    run_id = f"{cfg.get('experiment', EXPERIMENT)}-{'S' if smoke else ''}{split}-{now.strftime('%Y%m%dT%H%M%S%fZ')}"
    local = root / (SMOKE_DIR if smoke else LOCAL_DIR)
    local.mkdir(parents=True, exist_ok=True)
    (root / (SMOKE_DIR if smoke else RECORD_DIR)).mkdir(parents=True, exist_ok=True)
    for a, r in rows.items():  # the cards, with meeting text: local only
        write_atomic(local / f"{run_id}.{a}.predictions.jsonl", "".join(p.model_dump_json() + "\n" for p in r.values()))
    types = Counter(meta[s.id]["type"] for s in scs)
    record = {
        "run_id": run_id, "experiment": cfg.get("experiment", EXPERIMENT), "split": split, "smoke": smoke,
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
    ap.add_argument("command", choices=["run", "report", "prebuild"])
    ap.add_argument("--split", required=True, help="dev or test; prebuild also takes a list in order, e.g. test,dev")
    ap.add_argument("--arms", help="comma list; default: the arms of configs/e10.json")
    ap.add_argument("--smoke", action="store_true", help="a few cases of each type; the record goes to git-ignored results/local")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--commit", help="report: the commit whose saved rows to use (default: HEAD)")
    ap.add_argument("--config", type=Path, help="config under the repo root; default configs/e10.json (E11 is configs/e11.json)")
    ap.add_argument("--run-conditions", default="", help="what else was running while latency was measured")
    args = ap.parse_args(argv)
    cfg = load_config(ROOT, args.config)
    arms = args.arms.split(",") if args.arms else cfg["arms"]
    if not set(arms) <= set(cfg["arms"]):
        ap.error(f"arms must come from the config: {cfg['arms']}")
    if args.command == "prebuild":
        prebuild(ROOT, args.split.split(","))
        return 0
    if args.split not in ("dev", "test"):
        ap.error("--split must be dev or test here")
    if args.command == "run":
        if not args.run_conditions.strip() and not args.smoke:
            ap.error("--run-conditions is required: say what else was running while latency was measured")
        run(ROOT, args.split, arms, smoke=args.smoke, limit=args.limit, run_conditions=args.run_conditions, config=args.config)
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
