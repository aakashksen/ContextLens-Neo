"""The cheaper grader: many `claude -p` calls at once, with the local judge's rubric and verdict parsing.

One claim per call (`grade_single`) asks through `Judge` from e2_context, the code that asked the local judge, so the
rubric text (`e2-v1`) and the parsing are the same. Answers are saved as they arrive (ids and verdicts only, never
text), so a stopped run resumes. A rate or usage limit is waited out and reported, and is never a verdict; a claim
with no valid verdict stays ungraded and is listed."""

import json
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from contextlens.experiments.e2_context import _RUBRIC, Judge, JudgeFailed
from contextlens.experiments.judge_check import _fsync_append, open_rows
from contextlens.llm.client import CLAUDE_CALL_CAP, CLAUDE_CALLS_PATH, ClaudeCodeClient, LLMError, LLMResult
from contextlens.llm.untrusted import escape

HAIKU_MODEL = "claude-haiku-5-5"
DEFAULT_CONCURRENCY = 8
BACKOFF_S = (30.0, 120.0, 300.0)  # waits after the 1st, 2nd and 3rd limit hit on one unit; a 4th ends the run
MAX_FAILURES_IN_A_ROW = 5

_LIMIT = re.compile(r"rate.?limit|usage limit|hit your limit|limit reached|\b429\b|overloaded|too many requests|quota", re.I)
_FROM_CLAUDE = re.compile(r"claude (exited|reported an error|call failed)")


def is_limit(message: str) -> bool:
    """True for a failure of the `claude` call that says a rate or usage limit was hit. A reply that merely failed
    validation is never one, even if the words appear in it."""
    return bool(_FROM_CLAUDE.search(message)) and "schema validation" not in message and bool(_LIMIT.search(message))


def is_cap(message: str) -> bool:
    return "call cap of" in message


@dataclass(frozen=True)
class Item:
    id: str
    statement: str
    evidence: tuple[tuple[str, str], ...]  # (speaker, quote)


class CountedClaude(ClaudeCodeClient):
    """`claude -p` with the cheaper model, counting this process's own calls (the shared counter also moves for others)."""

    def __init__(self, model: str = HAIKU_MODEL, calls_path: Path = CLAUDE_CALLS_PATH, cap: int = CLAUDE_CALL_CAP):
        super().__init__(model=model, calls_path=calls_path, cap=cap)
        self.calls_made = 0
        self._count_lock = threading.Lock()

    def _raw(self, messages, schema, max_tokens, temperature, timeout_s) -> LLMResult:
        with self._count_lock:
            self.calls_made += 1
        return super()._raw(messages, schema, max_tokens, temperature, timeout_s)


class RowStore:
    """The saved rows of one run. The last row per claim wins; a `batch_issue` once seen is kept on later rows."""

    def __init__(self, path: Path, header: dict[str, Any]) -> None:
        self.path = path
        self.rows, self.asks = open_rows(path, header)
        self._lock = threading.Lock()

    def save(self, row: dict[str, Any]) -> None:
        with self._lock:
            old = self.rows.get(row["claim_id"], {})
            if "batch_issue" in old and "batch_issue" not in row:
                row = {**row, "batch_issue": old["batch_issue"]}
            _fsync_append(self.path, json.dumps(row, sort_keys=True))
            self.rows[row["claim_id"]] = row
            self.asks += 1


@dataclass
class Outcome:
    stop_reason: str | None = None
    rate_limit_events: int = 0
    failed_rows: int = 0
    batch_calls: int = 0
    unknown_ids: int = 0  # ids in batch replies that belong to no claim of that batch


class RateLimited(Exception):
    pass


class CapReached(Exception):
    pass


def pending(items: list[Item], store: RowStore) -> list[Item]:
    return [i for i in items if store.rows.get(i.id, {}).get("status") != "ok"]


def ungraded(items: list[Item], store: RowStore) -> list[str]:
    return [i.id for i in items if store.rows.get(i.id, {}).get("status") != "ok"]


@dataclass
class _Unit:
    kind: str
    items: list[Item]


@dataclass
class _Pool:
    queue: deque[_Unit]
    store: RowStore
    handlers: dict[str, Callable[[list[Item]], tuple[list[dict[str, Any]], list[_Unit]]]]
    should_stop: Callable[[], str | None]
    sleep: Callable[[float], None]
    backoff: tuple[float, ...]
    out: Outcome = field(default_factory=Outcome)
    lock: threading.Lock = field(default_factory=threading.Lock)
    pause_until: float = 0.0
    in_a_row: int = 0

    def stop(self, reason: str) -> None:
        with self.lock:
            self.out.stop_reason = self.out.stop_reason or reason

    def next_unit(self) -> _Unit | None:
        with self.lock:
            if self.out.stop_reason or not self.queue:
                return None
        reason = self.should_stop()
        if reason:
            self.stop(reason)
            return None
        with self.lock:
            return self.queue.popleft() if self.queue and not self.out.stop_reason else None

    def wait_gate(self) -> None:
        with self.lock:
            remaining = self.pause_until - time.monotonic()
        if remaining > 0:
            self.sleep(remaining)

    def process(self, unit: _Unit) -> None:
        hits = 0
        while True:
            self.wait_gate()
            try:
                rows, follow = self.handlers[unit.kind](unit.items)
                break
            except RateLimited as exc:
                with self.lock:
                    self.out.rate_limit_events += 1
                    if hits < len(self.backoff):
                        self.pause_until = max(self.pause_until, time.monotonic() + self.backoff[hits])
                hits += 1
                if hits > len(self.backoff):
                    rows = [{"claim_id": i.id, "status": "rate_limited", "error": str(exc)[:200]} for i in unit.items]
                    follow = []
                    self.stop(f"rate limit still hit after {len(self.backoff)} waits ({str(exc)[:120]}); rows kept, rerun to resume")
                    break
            except CapReached as exc:
                rows = [{"claim_id": i.id, "status": "capped", "error": str(exc)[:200]} for i in unit.items]
                follow = []
                self.stop(f"Claude call cap reached; only the person can raise it ({str(exc)[:100]})")
                break
        for row in rows:
            self.store.save(row)
        with self.lock:
            self.queue.extendleft(reversed(follow))
            for row in rows:
                if row["status"] == "failed":
                    self.out.failed_rows += 1
                    self.in_a_row += 1
                elif row["status"] == "ok":
                    self.in_a_row = 0
            streak = self.in_a_row
        if streak >= MAX_FAILURES_IN_A_ROW:
            self.stop(f"{streak} claims failed in a row; rows kept, rerun to resume")

    def worker(self) -> None:
        while (unit := self.next_unit()) is not None:
            self.process(unit)


def run_pool(
    units: list[_Unit], store: RowStore, handlers: dict[str, Callable], *, concurrency: int,
    should_stop: Callable[[], str | None], sleep: Callable[[float], None], backoff: tuple[float, ...],
) -> Outcome:
    pool = _Pool(deque(units), store, handlers, should_stop, sleep, backoff)
    threads = [threading.Thread(target=pool.worker, daemon=True) for _ in range(max(1, concurrency))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return pool.out


def _classify(exc: JudgeFailed) -> None:
    msg = str(exc)
    if is_cap(msg):
        raise CapReached(msg)
    if is_limit(msg):
        raise RateLimited(msg)


def _single_handler(judge: Judge, via: str) -> Callable[[list[Item]], tuple[list[dict[str, Any]], list[_Unit]]]:
    def one(unit_items: list[Item]) -> tuple[list[dict[str, Any]], list[_Unit]]:
        (item,) = unit_items
        try:
            verdict = judge.ask(item.statement, list(item.evidence))
        except JudgeFailed as exc:
            _classify(exc)
            return [{"claim_id": item.id, "status": "failed", "error": str(exc)[:200]}], []
        return [{"claim_id": item.id, "status": "ok", "cloud": verdict, "via": via}], []

    return one


def grade_single(
    items: list[Item], llm, store: RowStore, *, concurrency: int = DEFAULT_CONCURRENCY,
    should_stop: Callable[[], str | None] = lambda: None, sleep: Callable[[float], None] = time.sleep,
    backoff: tuple[float, ...] = BACKOFF_S,
) -> Outcome:
    """Ask once per claim that has no verdict yet, `concurrency` at a time."""
    units = [_Unit("single", [i]) for i in pending(items, store)]
    return run_pool(units, store, {"single": _single_handler(Judge(llm, tries=1), "single")}, concurrency=concurrency,
                    should_stop=should_stop, sleep=sleep, backoff=backoff)


# ---------------------------------------------------------------- batches

BATCH_SIZE = 20
BATCH_TIMEOUT_S = 180.0
BATCH_RUBRIC = (
    _RUBRIC.split("Reply with")[0].rstrip()
    + "\nYou get several items at once. Each is an <item id=...> holding its own <claim> and <evidence>. Judge every item "
    "on its own, by the rule above, and do not let one item affect another.\n"
    'Reply with one JSON object: {"verdicts": [{"id": "c1", "supported": true}, {"id": "c2", "supported": false}]} '
    "with exactly one entry for each item id, each id once, and nothing else."
)


def batch_messages(items: list[Item]) -> tuple[list[dict], dict[str, str]]:
    """The prompt for one batch, and the map from the short ids used in it to the claim ids."""
    ids = {f"c{n}": item.id for n, item in enumerate(items, 1)}
    blocks = []
    for short, item in zip(ids, items, strict=True):
        lines = "\n".join(f"{escape(sp)}: {escape(q)}" for sp, q in item.evidence)
        blocks.append(f'<item id="{short}">\n<claim>{escape(item.statement)}</claim>\n<evidence>\n{lines}\n</evidence>\n</item>')
    return [{"role": "system", "content": BATCH_RUBRIC}, {"role": "user", "content": "\n".join(blocks)}], ids


@dataclass
class BatchParse:
    verdicts: dict[str, bool] = field(default_factory=dict)
    issues: dict[str, str] = field(default_factory=dict)  # short id -> missing | repeated | unparsed
    unknown: list[str] = field(default_factory=list)
    valid: bool = True


def parse_batch(text: str, ids: list[str]) -> BatchParse:
    """Strict: a short id is graded only if it appears exactly once with a true or false. A reply that is not the
    expected shape grades nothing."""
    body = text.strip()
    if body.startswith("```"):
        body = body.strip("`").removeprefix("json").strip()
    try:
        entries = json.loads(body)["verdicts"]
        if not isinstance(entries, list) or not all(
            isinstance(e, dict) and isinstance(e.get("id"), str) and isinstance(e.get("supported"), bool) for e in entries
        ):
            raise ValueError("shape")
    except (ValueError, KeyError, TypeError):
        return BatchParse(issues=dict.fromkeys(ids, "unparsed"), valid=False)
    seen: dict[str, list[bool]] = {}
    for e in entries:
        seen.setdefault(e["id"], []).append(e["supported"])
    out = BatchParse(unknown=sorted(k for k in seen if k not in ids))
    for i in ids:
        got = seen.get(i, [])
        if len(got) == 1:
            out.verdicts[i] = got[0]
        else:
            out.issues[i] = "missing" if not got else "repeated"
    return out


def grade_batches(
    items: list[Item], llm, store: RowStore, *, batch_size: int = BATCH_SIZE, retry_single: bool = True,
    concurrency: int = DEFAULT_CONCURRENCY, should_stop: Callable[[], str | None] = lambda: None,
    sleep: Callable[[float], None] = time.sleep, backoff: tuple[float, ...] = BACKOFF_S,
) -> Outcome:
    """About `batch_size` claims per call. A claim whose id is missing or repeated, or whose batch reply is unusable, is
    asked again alone when `retry_single`, and otherwise stays ungraded; nothing is guessed. The issue stays on its row."""
    counts = {"calls": 0, "unknown": 0}
    count_lock = threading.Lock()

    def batch(unit_items: list[Item]) -> tuple[list[dict[str, Any]], list[_Unit]]:
        messages, ids = batch_messages(unit_items)
        by_claim = {item.id: item for item in unit_items}
        try:
            with count_lock:
                counts["calls"] += 1
            reply = llm.complete(messages, max_tokens=40 * len(unit_items), timeout_s=BATCH_TIMEOUT_S).text
            parsed = parse_batch(reply, list(ids))
        except LLMError as exc:
            if is_cap(str(exc)):
                raise CapReached(str(exc)) from exc
            if is_limit(str(exc)):
                raise RateLimited(str(exc)) from exc
            parsed = BatchParse(issues=dict.fromkeys(ids, "call_failed"), valid=False)
        with count_lock:
            counts["unknown"] += len(parsed.unknown)
        rows: list[dict[str, Any]] = []
        follow: list[_Unit] = []
        for short, claim_id in ids.items():
            if short in parsed.verdicts:
                rows.append({"claim_id": claim_id, "status": "ok", "cloud": parsed.verdicts[short], "via": "batch"})
                continue
            rows.append({"claim_id": claim_id, "status": "batch_issue" if retry_single else "ungraded",
                         "batch_issue": parsed.issues[short]})
            if retry_single:
                follow.append(_Unit("single", [by_claim[claim_id]]))
        return rows, follow

    todo = pending(items, store)
    units = [_Unit("batch", todo[n:n + batch_size]) for n in range(0, len(todo), batch_size)]
    out = run_pool(units, store, {"batch": batch, "single": _single_handler(Judge(llm, tries=1), "retry")},
                   concurrency=concurrency, should_stop=should_stop, sleep=sleep, backoff=backoff)
    out.batch_calls, out.unknown_ids = counts["calls"], counts["unknown"]
    return out
