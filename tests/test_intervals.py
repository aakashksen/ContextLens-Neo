"""Intervals and paired comparisons for every metric check_run can recompute, through check_run."""
import random
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score

from contextlens.eval import series
from contextlens.eval.check_run import check, recompute_metrics
from contextlens.eval.records import split_hash, write_run, write_scenarios_manifest
from contextlens.replay.runner import Prediction
from tests.test_records import LABELS, _git, _gold as gold4, _pred, _predl, _scenario

N = 40
JUDGE = {"model": "qwen-real", "rubric_version": "r1", "same_model_and_prompt_as_writer": False}
CONFIG = {"labels_reviewed": True, "judge": JUDGE}
INTERVAL_METRICS = ["macro_f1", "false_alert_rate", "abstain_when_insufficient", "answer_when_answerable",
                    "speaker_accuracy", "grounded_rate", "unsupported_claim_rate",
                    "latency_p50_ms", "latency_p95_ms"]


@pytest.fixture(scope="module", autouse=True)
def _remember_bootstraps():
    """Each test computes an interval or comparison and then has check_run compute it again, once or twice, on the
    same rows. The bootstrap is deterministic (seed 13), so remember each answer by the full content of its rows."""
    real_interval, real_comparison = series.interval, series.comparison
    seen: dict[tuple, object] = {}

    def key(*row_sets, name: str) -> tuple:
        return (name, *(tuple((s.model_dump_json(), p.model_dump_json()) for s, p in rows) for rows in row_sets))

    def interval(rows, name):
        k = key(rows, name=name)
        if k not in seen:
            seen[k] = real_interval(rows, name)
        return seen[k]

    def comparison(a_rows, b_rows, name):
        k = key(a_rows, b_rows, name=name)
        if k not in seen:
            seen[k] = real_comparison(a_rows, b_rows, name)
        return seen[k]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(series, "interval", interval)
        mp.setattr(series, "comparison", comparison)
        yield


def _gold() -> list[tuple[str, str]]:
    return [(f"d{i + 1}", LABELS[i % 4]) for i in range(N)]


def _system(wrong_every: int, speaker_wrong_every: int, unsupported_every: int, slow: int) -> list[Prediction]:
    out = []
    for i, (sid, gold) in enumerate(_gold()):
        label = gold if i % wrong_every else "update"
        p = _pred(sid, None if label == "insufficient" else label, ms=10 + (i * slow) % 97)
        if p.card and p.card.findings:
            ev = p.card.findings[0].evidence[0].model_copy(
                update={"speaker": "B" if i % speaker_wrong_every == 0 else "A"})
            f = p.card.findings[0].model_copy(update={"evidence": [ev]})
            p = p.model_copy(update={"card": p.card.model_copy(update={"findings": [f]}),
                                     "claim_supported": [i % unsupported_every != 0]})
        out.append(p)
    return out


def _sys_a() -> list[Prediction]:
    return _system(wrong_every=5, speaker_wrong_every=7, unsupported_every=6, slow=3)


def _sys_b() -> list[Prediction]:
    return _system(wrong_every=3, speaker_wrong_every=4, unsupported_every=3, slow=5)


def _make_repo(root: Path, gold: list[tuple[str, str]]) -> Path:
    d = root / "datasets/scenarios"
    d.mkdir(parents=True)
    (d / "dev.jsonl").write_text("".join(_scenario(i, g).model_dump_json() + "\n" for i, g in gold),
                                 encoding="utf-8")
    write_scenarios_manifest(root)
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    return root


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return _make_repo(tmp_path, _gold())


def _rows(preds: list[Prediction], gold: list[tuple[str, str]] | None = None):
    by_id = {p.scenario_id: p for p in preds}
    return [(_scenario(i, g), by_id[i]) for i, g in (gold or _gold())]


def _save(repo: Path, intervals=None, comparisons=None) -> None:
    scenarios = [s for s, _ in _rows(_sys_a())]
    systems = {"a": _sys_a(), "b": _sys_b()}
    write_run("E4", "dev", CONFIG, {"llm": "qwen-real"}, {"p": "v1"},
              {n: recompute_metrics(scenarios, p) for n, p in systems.items()},
              intervals or {}, [{"setting": "x"}], systems, split_hash("dev", repo),
              comparisons=comparisons, root=repo)


def _problems(repo: Path) -> list[str]:
    return check("E4", root=repo)


def _f1(gold: list[str], pred: list[str]) -> float:
    return float(f1_score(gold, pred, labels=LABELS, average="macro", zero_division=0))


def _independent_macro_f1_ci(preds: list[Prediction]) -> tuple[float, float]:
    """sklearn macro-F1 on whole-conversation resamples drawn with random.Random(13), numpy percentiles."""
    gold = [g for _, g in _gold()]
    pred = [series.label_of(p.card) for p in preds]  # type: ignore[arg-type]
    rng = random.Random(13)
    vals = []
    for _ in range(10_000):
        idx = [rng.randrange(N) for _ in range(N)]  # each conversation holds one case
        vals.append(_f1([gold[i] for i in idx], [pred[i] for i in idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def test_macro_f1_point_value_by_hand() -> None:
    # gold: contradiction, consistent, update, insufficient; predicted: contradiction, contradiction, update,
    # insufficient. F1 per label: contradiction 2/3, consistent 0, update 1, insufficient 1.
    scenarios = [_scenario(i, g) for i, g in gold4("dev")]
    preds = [_pred(s, g) for s, g in _predl("dev")]
    assert recompute_metrics(scenarios, preds)["macro_f1"] == pytest.approx((2 / 3 + 0 + 1 + 1) / 4)


def test_macro_f1_interval_equals_an_independent_bootstrap(repo: Path) -> None:
    lo, hi = _independent_macro_f1_ci(_sys_a())
    _save(repo, intervals={"a": {"macro_f1": (lo, hi)}})
    assert _problems(repo) == []
    _save(repo, intervals={"a": {"macro_f1": (lo, hi + 0.001)}})
    assert any("macro_f1" in p and "does not recompute" in p for p in _problems(repo))


def test_every_metric_has_a_checked_interval(repo: Path) -> None:
    got = {m: series.interval(_rows(_sys_a()), m) for m in INTERVAL_METRICS}
    assert all(v is not None for v in got.values())
    _save(repo, intervals={"a": got})  # type: ignore[arg-type]
    assert _problems(repo) == []
    _save(repo, intervals={"a": {m: (v[0], v[1] + 0.01) for m, v in got.items()}})  # type: ignore[index]
    problems = _problems(repo)
    for m in INTERVAL_METRICS:
        assert any(f"interval {m} does not recompute" in p for p in problems), m


def test_comparison_on_macro_f1_matches_an_independent_difference(repo: Path) -> None:
    gold = [g for _, g in _gold()]
    expected = (_f1(gold, [series.label_of(p.card) for p in _sys_a()])  # type: ignore[arg-type]
                - _f1(gold, [series.label_of(p.card) for p in _sys_b()]))  # type: ignore[arg-type]
    c = series.comparison(_rows(_sys_a()), _rows(_sys_b()), "macro_f1")
    assert c is not None and c["diff"] == pytest.approx(expected)
    assert c["mcnemar_p"] is None
    _save(repo, comparisons=[{"a": "a", "b": "b", **c}])
    assert _problems(repo) == []


def test_every_metric_has_a_checked_comparison_and_no_mcnemar(repo: Path) -> None:
    comps = {m: series.comparison(_rows(_sys_a()), _rows(_sys_b()), m) for m in INTERVAL_METRICS}
    paired_yes_no = ("answer_when_answerable", "abstain_when_insufficient")  # per-case outcomes: McNemar applies (T4.2)
    assert all(c is not None and (c["mcnemar_p"] is not None) == (m in paired_yes_no) for m, c in comps.items())

    def save(**change: object) -> list[str]:
        _save(repo, comparisons=[{"a": "a", "b": "b", **c, **change} for c in comps.values()])  # type: ignore[dict-item]
        return _problems(repo)

    assert save() == []
    problems = save(mcnemar_p=0.5)  # a p-value on a metric that is not correct/incorrect
    for m in INTERVAL_METRICS:
        assert any("comparison" in p and m in p and "does not recompute" in p for p in problems), m


def test_a_resample_where_the_metric_is_undefined_gives_no_interval(tmp_path: Path) -> None:
    """One contradiction and two consistent conversations: one resample in 27 holds no negatives,
    so false_alert_rate is undefined there. Dropping those resamples would pull the interval toward
    the defined cases, so the metric gets no interval and no paired comparison."""
    gold = [("d1", "contradiction"), ("d2", "consistent"), ("d3", "consistent")]
    preds = [_pred(i, "consistent") for i, _ in gold]
    rows = _rows(preds, gold)
    assert recompute_metrics([s for s, _ in rows], preds)["false_alert_rate"] == 0.0
    assert series.interval(rows, "false_alert_rate") is None
    assert series.comparison(rows, rows, "false_alert_rate") is None

    repo = _make_repo(tmp_path, gold)
    write_run("E4", "dev", CONFIG, {"llm": "qwen-real"}, {"p": "v1"},
              {"a": recompute_metrics([s for s, _ in rows], preds)}, {"a": {"false_alert_rate": (0.0, 0.0)}},
              [{"setting": "x"}], {"a": preds}, split_hash("dev", repo), root=repo)
    assert any("false_alert_rate" in p and "cannot be recomputed" in p for p in _problems(repo))


def test_a_metric_undefined_on_the_whole_run_gets_no_interval() -> None:
    rows = _rows([_pred(i, "update") for i, _ in _gold()])
    assert series.interval([(s, p.model_copy(update={"card": None, "ranked_turn_ids": []})) for s, p in rows],
                           "macro_f1") is None
