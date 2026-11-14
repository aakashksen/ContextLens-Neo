"""The cheaper grader, one claim per call, in parallel. Fakes only: no `claude` process is started."""
import json
import subprocess
import threading
import time
from pathlib import Path

import pytest

from contextlens.experiments import parallel_grader as pg
from contextlens.experiments.judge_check import open_rows
from contextlens.llm.client import ClaudeCodeClient, FakeLLM, LLMError, LLMResult

SECRET = "the budget is 4242 dollars"
HEADER = {"kind": "test", "sample_id": "s1"}


def _items(n: int) -> list[pg.Item]:
    return [pg.Item(f"c{i}", f"claim {i} {SECRET}", (("Maya", f"quote {i}"),)) for i in range(n)]


class Fake(FakeLLM):
    """Supported exactly when the claim number is even. `script` maps call number to an error text."""

    def __init__(self, script: dict[int, str] | None = None, delay: float = 0.0):
        super().__init__([])
        self.script, self.delay, self.n, self.inflight, self.peak = script or {}, delay, 0, 0, 0
        self.lock = threading.Lock()

    def _raw(self, messages, schema, max_tokens, temperature, timeout_s) -> LLMResult:
        with self.lock:
            self.n += 1
            k = self.n
            self.inflight += 1
            self.peak = max(self.peak, self.inflight)
        try:
            time.sleep(self.delay)
            if k in self.script:
                raise LLMError(self.script[k])
            num = int(messages[-1]["content"].split("<claim>claim ", 1)[1].split(" ", 1)[0])
            return LLMResult(text=json.dumps({"supported": num % 2 == 0}), model="fake")
        finally:
            with self.lock:
                self.inflight -= 1


def _store(tmp_path: Path) -> pg.RowStore:
    return pg.RowStore(tmp_path / "rows.jsonl", HEADER)


def test_grades_in_parallel_and_saves_ids_and_verdicts_only(tmp_path: Path) -> None:
    llm = Fake(delay=0.05)
    store = _store(tmp_path)
    out = pg.grade_single(_items(16), llm, store, concurrency=4)
    assert out.stop_reason is None
    assert {k: r["cloud"] for k, r in store.rows.items()} == {f"c{i}": i % 2 == 0 for i in range(16)}
    assert 2 <= llm.peak <= 4
    text = (tmp_path / "rows.jsonl").read_text(encoding="utf-8")
    assert SECRET not in text and "quote" not in text


def test_resume_grades_only_what_is_not_done(tmp_path: Path) -> None:
    store = _store(tmp_path)
    pg.grade_single(_items(6)[:3], Fake(), store, concurrency=2)
    rows, _ = open_rows(tmp_path / "rows.jsonl", HEADER)
    assert len(rows) == 3
    llm = Fake()
    store2 = _store(tmp_path)
    pg.grade_single(_items(6), llm, store2, concurrency=2)
    assert llm.n == 3 and len(store2.rows) == 6


def test_rate_limit_backs_off_then_succeeds_and_is_never_a_verdict(tmp_path: Path) -> None:
    sleeps: list[float] = []
    llm = Fake({1: "claude reported an error: You've hit your limit, resets 3pm"})
    store = _store(tmp_path)
    out = pg.grade_single(_items(1), llm, store, concurrency=1, sleep=sleeps.append, backoff=(30.0, 60.0))
    assert len(sleeps) == 1 and 29.0 <= sleeps[0] <= 30.0 and out.rate_limit_events == 1 and out.stop_reason is None
    assert store.rows["c0"]["cloud"] is True


def test_persistent_rate_limit_stops_the_run_with_no_verdicts(tmp_path: Path) -> None:
    llm = Fake({k: "claude exited 1: 429 rate limit exceeded" for k in range(1, 50)})
    store = _store(tmp_path)
    out = pg.grade_single(_items(10), llm, store, concurrency=1, sleep=lambda s: None, backoff=(1.0, 2.0))
    assert out.stop_reason and "rate limit" in out.stop_reason
    assert llm.n == 3  # the first claim, tried again after each backoff; nothing after the stop
    assert all("cloud" not in r for r in store.rows.values())
    assert [r["status"] for r in store.rows.values()] == ["rate_limited"]


def test_usage_limit_text_is_a_limit_not_a_failure(tmp_path: Path) -> None:
    assert pg.is_limit("claude reported an error: Claude AI usage limit reached")
    assert pg.is_limit("claude exited 1: Overloaded")
    assert not pg.is_limit("output failed schema validation twice: limit was not a boolean")
    assert not pg.is_limit("claude call failed: timed out")


def test_a_claim_without_a_valid_verdict_stays_ungraded_and_is_listed(tmp_path: Path) -> None:
    llm = Fake({2: "claude exited 1: boom"})
    store = _store(tmp_path)
    out = pg.grade_single(_items(5), llm, store, concurrency=1)
    bad = [k for k, r in store.rows.items() if r["status"] != "ok"]
    assert len(bad) == 1 and out.stop_reason is None
    assert pg.ungraded(_items(5), store) == bad
    assert "cloud" not in store.rows[bad[0]]


def test_five_failures_in_a_row_stop_the_run(tmp_path: Path) -> None:
    llm = Fake({k: "claude exited 1: boom" for k in range(1, 50)})
    out = pg.grade_single(_items(20), llm, _store(tmp_path), concurrency=1)
    assert out.stop_reason and "failed in a row" in out.stop_reason and llm.n == 5


def test_stop_check_ends_the_run_cleanly_with_rows_kept(tmp_path: Path) -> None:
    store = _store(tmp_path)
    seen = iter([None, None, "weekly usage 91% reached the limit of 90%"])
    out = pg.grade_single(_items(10), Fake(), store, concurrency=1, should_stop=lambda: next(seen, "stop"))
    assert out.stop_reason == "weekly usage 91% reached the limit of 90%"
    assert len(store.rows) == 2 and all(r["status"] == "ok" for r in store.rows.values())


def _fake_claude(monkeypatch: pytest.MonkeyPatch, seen: list[list[str]]) -> None:
    monkeypatch.setattr("shutil.which", lambda name: "claude")

    def run(cmd, **kw):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"result": '{"supported": true}', "usage": {}}), "")

    monkeypatch.setattr(subprocess, "run", run)


def test_the_grader_asks_for_the_explicit_model_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []
    _fake_claude(monkeypatch, seen)
    client = pg.CountedClaude(calls_path=tmp_path / "calls.json")
    pg.grade_single(_items(1), client, _store(tmp_path), concurrency=1)
    assert pg.HAIKU_MODEL == "claude-haiku-5-5" and client.model == pg.HAIKU_MODEL
    assert seen[0][seen[0].index("--model") + 1] == "claude-haiku-5-5"
    assert client.calls_made == 1


def test_the_shared_counter_never_passes_the_cap_under_concurrency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_claude(monkeypatch, [])
    path = tmp_path / "calls.json"
    client = ClaudeCodeClient(model=pg.HAIKU_MODEL, calls_path=path, cap=25)
    errors: list[str] = []

    def one() -> None:
        try:
            client.complete([{"role": "user", "content": "x"}])
        except LLMError as exc:
            errors.append(str(exc))

    threads = [threading.Thread(target=one) for _ in range(60)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert json.loads(path.read_text())["count"] == 25
    assert len(errors) == 35 and all("cap" in e for e in errors)


def test_a_cap_error_stops_the_run(tmp_path: Path) -> None:
    llm = Fake({k: "Claude Code call cap of 5000 reached; only the person can raise it" for k in range(1, 9)})
    out = pg.grade_single(_items(5), llm, _store(tmp_path), concurrency=1)
    assert out.stop_reason and "cap" in out.stop_reason and llm.n == 1


def test_a_row_saved_after_a_crash_cut_the_last_line_is_not_lost(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(HEADER) + "\n" + '{"claim_id": "c0", "status": "ok", "cloud": true}\n{"claim_id": "c1", "sta',
                    encoding="utf-8")
    _store(tmp_path).save({"claim_id": "c2", "status": "ok", "cloud": False, "via": "single"})
    rows, _ = open_rows(path, HEADER)
    assert sorted(rows) == ["c0", "c2"]
