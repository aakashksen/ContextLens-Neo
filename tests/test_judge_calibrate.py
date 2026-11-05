"""Calibration of the cheaper grader against Sonnet. FakeLLM only; nothing is called for real."""
import json
import os
import random
import threading
from pathlib import Path

import pytest

from contextlens.eval.records import ROOT
from contextlens.experiments import judge_calibrate as cal
from contextlens.experiments import parallel_grader as pg
from contextlens.llm.client import FakeLLM, LLMError, LLMResult
from contextlens.replay.runner import Prediction
from contextlens.schemas import Card, EvidenceRef, Finding

JC = ("JC-20990101T000000000001Z", "JC-20990101T000000000002Z")
SECRET = "raw words from a transcript"
REAL = ("JC-20261104T144503532496Z", "JC-20261104T144600470844Z")


@pytest.fixture(autouse=True)
def _fake_record_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cal, "JC_RECORDS", JC)


def _finding(system: str, sid: str, k: int) -> Finding:
    return Finding(agent="baseline", kind="recall", statement=f"claim {system}.{sid}.{k} {SECRET}", confidence=0.9,
                   evidence=[EvidenceRef(turn_id=f"{sid}:t0", quote=f"quote {SECRET}", speaker="Maya", start_ms=0)])


def _verdicts(set_name: str, system: str, sid: str, k: int) -> tuple[bool, bool]:
    """(local, sonnet): the local judge says yes to everything; Sonnet says no to every third claim."""
    h = sum(map(ord, f"{set_name}{system}{sid}")) + k
    return True, h % 3 != 0


def _make_repo(root: Path, sets: tuple[str, str] = ("old", "new"), cases: int = 12) -> None:
    runs = root / "results/runs"
    runs.mkdir(parents=True)
    for rid, set_name in zip(JC, sets, strict=True):
        preds, metrics = {}, {}
        for system in ("single", "team_changed"):
            rows, loc, cl, n = [], 0, 0, 0
            for c in range(cases):
                sid = f"{set_name}{c:02d}"
                ks = range(1 + c % 3)
                v = [_verdicts(set_name, system, sid, k) for k in ks]
                rows.append(Prediction(scenario_id=sid, card=Card(mode="recall", status="answered", findings=[_finding(system, sid, k) for k in ks]),
                                       claim_supported=[a for a, _ in v], cloud_supported=[b for _, b in v]))
                n += len(v)
                loc += sum(a for a, _ in v)
                cl += sum(b for _, b in v)
            rel = f"results/runs/{rid}.{system}.predictions.jsonl"
            (root / rel).write_text("".join(r.model_dump_json() + "\n" for r in rows), encoding="utf-8")
            preds[system] = rel
            metrics[system] = {"grounded_rate": loc / n, "cloud_grounded_rate": cl / n}
        rec = {"run_id": rid, "experiment": "JC", "split": "dev", "config": {"mode": "all_claims", "set": set_name}, "metrics": metrics,
               "predictions": preds}
        (runs / f"{rid}.json").write_text(json.dumps(rec), encoding="utf-8")


def test_the_real_records_hold_the_local_verdict_in_claim_supported_and_sonnet_in_cloud_supported() -> None:
    pool = cal.load_pool(ROOT, REAL)  # also checks both shares against the recorded grounded_rate and cloud_grounded_rate
    assert len(pool) == 574
    by = {}
    for c in pool:
        by.setdefault((c.set, c.system), []).append(c)
    assert {k: len(v) for k, v in by.items()} == {("old", "single"): 127, ("old", "team_changed"): 215,
                                                  ("new", "single"): 85, ("new", "team_changed"): 147}
    share = lambda cs, f: round(sum(f(c) for c in cs) / len(cs), 4)  # noqa: E731
    assert share(by[("old", "single")], lambda c: c.sonnet) == 0.874 and share(by[("old", "single")], lambda c: c.local) == 0.9764


def test_a_mismatch_between_the_field_and_the_recorded_rate_refuses(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    path = tmp_path / f"results/runs/{JC[0]}.json"
    rec = json.loads(path.read_text())
    rec["metrics"]["single"]["cloud_grounded_rate"] = 0.123
    path.write_text(json.dumps(rec))
    with pytest.raises(cal.Refused, match="cloud_grounded_rate"):
        cal.load_pool(tmp_path)


def test_sample_is_seeded_stratified_and_the_same_every_time(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    pool = cal.load_pool(tmp_path)
    a, b = cal.draw_sample(pool, 30), cal.draw_sample(pool, 30)
    assert [c.id for c in a.claims] == [c.id for c in b.claims] and a.sample_id == b.sample_id and len(a.claims) == 30
    assert {c.set for c in a.claims} == {"old", "new"} and {c.system for c in a.claims} == {"single", "team_changed"}
    assert sum(s["taken"] for s in a.strata.values()) == 30 and set(a.strata) == {f"{x}/{y}" for x in ("old", "new") for y in ("single", "team_changed")}
    assert cal.draw_sample(pool, 30, seed=1).sample_id != a.sample_id
    assert len(cal.draw_sample(pool, 10_000).claims) == len(pool)
    assert len({c.id for c in a.claims}) == 30


def test_usage_reads_the_highest_value_of_the_newest_files_and_does_not_touch_them(tmp_path: Path) -> None:
    def put(name: str, pct: float | None, age: int) -> None:
        d = {"data": {"rate_limits": {"seven_day": {"used_percentage": pct}}}} if pct is not None else {"data": {}}
        p = tmp_path / name
        p.write_text(json.dumps(d))
        os.utime(p, (1_000_000 - age, 1_000_000 - age))

    put("a.json", 70, 5)
    put("b.json", 85, 4)
    put("c.json", None, 3)
    put("old.json", 99, 100)
    (tmp_path / "x.json.123").write_text("{}")
    (tmp_path / "bad.json").write_text("{not json")
    os.utime(tmp_path / "bad.json", (1_000_000 - 1, 1_000_000 - 1))
    before = sorted((p.name, p.stat().st_mtime, p.read_bytes()) for p in tmp_path.iterdir())
    assert cal.usage_percent(tmp_path, newest=4) == 85
    assert cal.usage_percent(tmp_path, newest=6) == 99
    assert sorted((p.name, p.stat().st_mtime, p.read_bytes()) for p in tmp_path.iterdir()) == before


def test_unreadable_usage_refuses(tmp_path: Path) -> None:
    with pytest.raises(cal.UsageUnreadable):
        cal.usage_percent(tmp_path / "missing")
    (tmp_path / "a.json").write_text('{"data": {}}')
    with pytest.raises(cal.UsageUnreadable):
        cal.usage_percent(tmp_path)


def test_guard_stops_at_the_limit_and_checks_only_now_and_then(tmp_path: Path) -> None:
    values = iter([40.0, 41.0, 89.9, 90.0])
    clock = [0.0]
    guard = cal.UsageGuard(90.0, read=lambda: next(values), now=lambda: clock[0], interval_s=30.0)
    assert guard.start() == 40.0 and guard.check() is None
    clock[0] = 10
    assert guard.check() is None  # not read again yet
    clock[0] = 31
    assert guard.check() is None and guard.last == 41.0
    clock[0] = 62
    assert guard.check() is None and guard.last == 89.9
    clock[0] = 93
    assert "90" in (guard.check() or "") and guard.last == 90.0


def test_guard_refuses_to_start_at_or_over_the_limit() -> None:
    with pytest.raises(cal.Refused, match="already"):
        cal.UsageGuard(90.0, read=lambda: 91.0, now=lambda: 0.0).start()


def _agree(a: list[bool], b: list[bool]) -> float:
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def _claims(n: int, local_acc: float, haiku_acc: float, seed: int = 3) -> tuple[list[cal.CalClaim], dict[str, bool]]:
    rng = random.Random(seed)
    claims, haiku = [], {}
    for i in range(n):
        s = rng.random() < 0.7
        loc = s if rng.random() < local_acc else not s
        h = s if rng.random() < haiku_acc else not s
        cid = f"r/single/case{i // 2:03d}/{i % 2}"
        claims.append(cal.CalClaim(cid, "old" if i < n // 2 else "new", "single", f"case{i // 2:03d}", loc, s, pg.Item(cid, "x", ())))
        haiku[cid] = h
    return claims, haiku


def test_analysis_matches_hand_counts_and_calls_a_much_worse_grader_clearly_worse() -> None:
    claims = [cal.CalClaim(f"r/s/c{i}/0", "old", "s", f"c{i}", loc, son, pg.Item(f"r/s/c{i}/0", "x", ()))
              for i, (loc, son) in enumerate([(True, True), (True, False), (False, False), (True, True)])]
    haiku = {"r/s/c0/0": True, "r/s/c1/0": False, "r/s/c2/0": True, "r/s/c3/0": True}
    a = cal.analyse(claims, haiku, resamples=200)["all"]
    assert a["n"] == 4 and a["haiku_vs_sonnet"]["agreement"] == 0.75 and a["local_vs_sonnet"]["agreement"] == 0.75
    assert a["haiku_vs_sonnet"]["disagree"] == {"grader_unsupported_sonnet_supported": 0, "grader_supported_sonnet_unsupported": 1}
    assert a["local_vs_sonnet"]["disagree"] == {"grader_unsupported_sonnet_supported": 0, "grader_supported_sonnet_unsupported": 1}
    assert a["share_unsupported"] == {"sonnet": 0.5, "haiku": 0.25, "local": 0.25}
    assert a["haiku_vs_sonnet"]["kappa"] == pytest.approx(0.5)
    good, h_good = _claims(300, 0.7, 0.95)
    bad, h_bad = _claims(300, 0.95, 0.65)
    assert cal.analyse(good, h_good, resamples=300)["all"]["verdict"] == "not clearly worse"
    worse = cal.analyse(bad, h_bad, resamples=300)
    assert worse["all"]["verdict"] == "clearly worse" and worse["all"]["agreement_difference"]["hi"] < 0
    assert set(worse) == {"all", "old", "new"} and worse["old"]["n"] == 150


def test_analysis_leaves_out_claims_the_grader_did_not_grade() -> None:
    claims, haiku = _claims(40, 0.8, 0.9)
    gone = next(iter(haiku))
    del haiku[gone]
    out = cal.analyse(claims, haiku, resamples=100)
    assert out["all"]["n"] == 39


def _rows(path: Path, spec: dict[str, tuple[bool, str | None]]) -> pg.RowStore:
    store = pg.RowStore(path, {"kind": "t"})
    for cid, (v, via) in spec.items():
        store.save({"claim_id": cid, "status": "ok", "cloud": v, "via": via or "batch"})
    return store


def test_batching_rule_keeps_one_per_call_on_any_loss_or_a_point_of_rate_change(tmp_path: Path) -> None:
    ids = [f"c{i}" for i in range(200)]
    single = _rows(tmp_path / "s.jsonl", {i: (k % 2 == 0, "single") for k, i in enumerate(ids)})
    same = _rows(tmp_path / "b.jsonl", {i: (k % 2 == 0, "batch") for k, i in enumerate(ids)})
    ok = cal.batching_report(ids, single.rows, same.rows, unknown_ids=0)
    assert ok["match_rate"] == 1.0 and ok["verdict"] == "batching allowed" and ok["lost_or_misaligned"] == 0
    swapped = {i: (k % 2 == 0 if k >= 2 else k % 2 != 0, "batch") for k, i in enumerate(ids)}  # two flips of 200: 1 point... 
    two = cal.batching_report(ids, single.rows, _rows(tmp_path / "c.jsonl", swapped).rows, unknown_ids=0)
    assert two["mismatches"] == 2 and two["rate_difference"] == pytest.approx(0.0) and two["verdict"] == "batching allowed"
    shifted = {i: (k % 2 == 0 or k < 4, "batch") for k, i in enumerate(ids)}  # two claims flip to supported: +1 point
    moved = cal.batching_report(ids, single.rows, _rows(tmp_path / "d.jsonl", shifted).rows, unknown_ids=0)
    assert moved["rate_difference"] == pytest.approx(0.01) and moved["verdict"] == "keep one per call"
    lost = _rows(tmp_path / "e.jsonl", {i: (k % 2 == 0, "batch") for k, i in enumerate(ids)})
    lost.save({"claim_id": "c5", "status": "ok", "cloud": False, "via": "retry", "batch_issue": "missing"})
    r = cal.batching_report(ids, single.rows, lost.rows, unknown_ids=0)
    assert r["lost_or_misaligned"] == 1 and r["verdict"] == "keep one per call" and r["compared"] == 199
    assert cal.batching_report(ids, single.rows, same.rows, unknown_ids=1)["verdict"] == "keep one per call"
    partial = cal.batching_report(ids, single.rows, {}, unknown_ids=0)
    assert partial["verdict"] == "not determined"


class Haiku(FakeLLM):
    """Agrees with Sonnet except on claims whose key ends in .2; understands single and batch prompts."""

    def __init__(self, truth: dict[str, bool], limit_after: int | None = None):
        super().__init__([])
        self.truth, self.n, self.lock, self.limit_after = truth, 0, threading.Lock(), limit_after

    @property
    def calls_made(self) -> int:
        return self.n

    def _raw(self, messages, schema, max_tokens, temperature, timeout_s) -> LLMResult:
        with self.lock:
            self.n += 1
            k = self.n
        if self.limit_after is not None and k > self.limit_after:
            raise LLMError("claude exited 1: Claude usage limit reached")
        user = messages[-1]["content"]
        import re

        def verdict(statement: str) -> bool:
            key = statement.split(" ")[1]
            return self.truth[key] != key.endswith(".2")

        if "<item " in user:
            ids = re.findall(r'<item id="([^"]+)">', user)
            stmts = re.findall(r"<claim>(.*?)</claim>", user)
            return LLMResult(text=json.dumps({"verdicts": [{"id": i, "supported": verdict(s)} for i, s in zip(ids, stmts, strict=True)]}), model="fake")
        stmt = user.split("<claim>", 1)[1].split("</claim>", 1)[0]
        return LLMResult(text=json.dumps({"supported": verdict(stmt)}), model="fake")


def _truth(pool: list[cal.CalClaim]) -> dict[str, bool]:
    return {c.item.statement.split(" ")[1]: c.sonnet for c in pool}


def _args(tmp_path: Path, **kw):
    base = dict(sample=40, concurrency=3, mode="single", max_weekly_pct=90.0, use_guard=True)
    return cal.Args(**{**base, **kw})


def _run(tmp_path: Path, llm, **kw) -> Path:
    guard = kw.pop("guard", cal.UsageGuard(90.0, read=lambda: 50.0, now=lambda: 0.0))
    return cal.run(_args(tmp_path, **kw), llm=llm, root=tmp_path, counter_path=tmp_path / "calls.json", cap=5000,
                   guard=guard, say=lambda s: None, sleep=lambda s: None)


def test_a_full_single_run_writes_a_record_without_text_and_with_the_required_parts(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    pool = cal.load_pool(tmp_path)
    said: list[str] = []
    llm = Haiku(_truth(pool))
    guard = cal.UsageGuard(90.0, read=lambda: 50.0, now=lambda: 0.0)
    out = cal.run(_args(tmp_path), llm=llm, root=tmp_path, counter_path=tmp_path / "calls.json", cap=5000, guard=guard,
                  say=said.append, sleep=lambda s: None)
    assert any("40" in s and "call" in s for s in said[:3])  # the planned count comes first
    text = out.read_text(encoding="utf-8")
    assert SECRET not in text and "quote" not in text
    rec = json.loads(text)
    assert out.parent == tmp_path / "results/calibration" and rec["provisional"] is True
    assert any("no claim comes from this record alone" in n.lower() for n in rec["notes"])
    assert any("reference" in n and "ground truth" in n for n in rec["notes"])
    assert rec["grader"]["model"] == llm.model
    assert rec["sampling"]["seed"] == cal.SEED and rec["sampling"]["n"] == 40 and "strata" in rec["sampling"]
    assert set(rec["comparison"]) == {"all", "old", "new"} and rec["comparison"]["all"]["n"] == 40
    run = rec["run"]
    assert run["concurrency"] == 3 and run["calls_made"] >= 40 and run["failures"] == 0 and run["usage_start_pct"] == 50.0
    assert run["usage_end_pct"] == 50.0 and run["claims_per_minute"] > 0 and run["wall_seconds"] >= 0
    assert rec["ungraded"] == [] and rec["batching"] is None
    assert len(rec["rows"]) == 40 and set(rec["rows"][0]) >= {"id", "set", "system", "local", "sonnet", "grader"}


def test_the_same_sample_is_graded_in_both_modes_and_the_batching_verdict_appears(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    truth = _truth(cal.load_pool(tmp_path))
    single = json.loads(_run(tmp_path, Haiku(truth)).read_text())
    batch_llm = Haiku(truth)
    batch = json.loads(_run(tmp_path, batch_llm, mode="batch").read_text())
    assert [r["id"] for r in single["rows"]] == [r["id"] for r in batch["rows"]]
    assert single["sampling"]["sample_id"] == batch["sampling"]["sample_id"]
    assert batch["batching"]["match_rate"] == 1.0 and batch["batching"]["verdict"] == "batching allowed"
    assert batch["run"]["calls_made"] == 2  # 40 claims in two calls of 20


def test_a_usage_limit_in_the_middle_keeps_rows_writes_no_record_and_resumes(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    truth = _truth(cal.load_pool(tmp_path))
    with pytest.raises(cal.Stopped) as stop:
        _run(tmp_path, Haiku(truth, limit_after=10), concurrency=1)
    assert "rate limit" in str(stop.value) and not (tmp_path / "results/calibration").exists()
    rows_file = next((tmp_path / "results/partial/judge-calibrate").glob("*.single.rows.jsonl"))
    kept = rows_file.read_text(encoding="utf-8")
    assert SECRET not in kept and kept.count('"status": "ok"') == 10
    llm = Haiku(truth)
    rec = json.loads(_run(tmp_path, llm).read_text())
    assert llm.n == 30 and rec["comparison"]["all"]["n"] == 40


def test_the_weekly_guard_stops_cleanly_and_reports_percentages(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    truth = _truth(cal.load_pool(tmp_path))
    reads = iter([80.0, 80.0, 91.0])
    guard = cal.UsageGuard(90.0, read=lambda: next(reads, 91.0), now=iter(range(0, 10_000, 60)).__next__, interval_s=30.0)
    with pytest.raises(cal.Stopped) as stop:
        _run(tmp_path, Haiku(truth), guard=guard, concurrency=1)
    assert "91" in str(stop.value) and "90" in str(stop.value) and stop.value.usage_start_pct == 80.0 and stop.value.usage_end_pct == 91.0
    assert not (tmp_path / "results/calibration").exists()


def test_ungraded_claims_are_listed_and_never_counted(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    truth = _truth(cal.load_pool(tmp_path))

    class Flaky(Haiku):
        def _raw(self, messages, schema, max_tokens, temperature, timeout_s):
            if "claim single.old01.0 " in messages[-1]["content"]:
                raise LLMError("claude exited 1: boom")
            return super()._raw(messages, schema, max_tokens, temperature, timeout_s)

    rec = json.loads(_run(tmp_path, Flaky(truth), sample=10_000, concurrency=1).read_text())
    assert len(rec["ungraded"]) == 1 and rec["ungraded"][0].endswith("/old01/0") and rec["run"]["failures"] == 1
    assert rec["comparison"]["all"]["n"] == len(rec["rows"]) - 1


def test_the_run_refuses_over_the_call_cap_before_any_call(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    (tmp_path / "calls.json").write_text(json.dumps({"count": 4990, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}))
    llm = Haiku({})
    with pytest.raises(cal.Refused, match="cap"):
        _run(tmp_path, llm)
    assert llm.n == 0


def test_main_refuses_without_readable_usage_unless_told_not_to(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(cal, "SESSIONS_DIR", tmp_path / "none")
    assert cal.main(["--sample", "5", "--mode", "single"]) == 1
    assert "usage" in capsys.readouterr().err
