"""E6 runner: prompt versions on dev, one factor each. No model is called for real."""
import json
import shutil
from pathlib import Path

import pytest

from contextlens.experiments import e6_prompts as e6
from contextlens.prompts.registry import get_prompt


def test_each_version_differs_by_a_declared_factor() -> None:
    versions = e6.declared_versions()
    assert [v.version for v in versions] == list(e6.VERSIONS)
    candidates = versions[1:]
    assert [v.factor for v in candidates] == list(e6.FACTORS)  # every factor once, none twice
    base = versions[0].text
    for v in candidates:
        assert get_prompt("recall", v.version).text == v.text  # selectable through the registry
        assert v.text != base
        added, removed = e6.line_diff(base, v.text)
        assert (list(v.added), list(v.removed)) == (added, removed)  # exactly what the version declares
    assert len({v.text_sha256 for v in versions}) == len(versions)


def _copy_prompts(tmp: Path) -> Path:
    shutil.copytree(e6.PROMPTS_DIR, tmp / "prompts")
    return tmp / "prompts"


def test_a_version_with_an_undeclared_change_or_a_repeated_factor_is_refused(tmp_path: Path, monkeypatch) -> None:
    # the registry reads its own directory, so point it at the copy
    from contextlens.prompts import registry

    d = _copy_prompts(tmp_path)
    monkeypatch.setattr(registry, "_DIR", d)
    assert [v.version for v in e6.declared_versions(d)] == list(e6.VERSIONS)
    p = d / "v3" / "recall.md"
    p.write_text(p.read_text(encoding="utf-8") + "\nBe polite.\n", encoding="utf-8")  # a second change, not declared
    with pytest.raises(ValueError, match="v3: the declared change"):
        e6.declared_versions(d)
    shutil.copy(e6.PROMPTS_DIR / "v3" / "recall.md", p)
    meta = d / "v6" / "factor.json"
    data = json.loads(meta.read_text(encoding="utf-8"))
    meta.write_text(json.dumps({**data, "factor": "few_shot"}), encoding="utf-8")
    with pytest.raises(ValueError, match="already tried"):
        e6.declared_versions(d)
    meta.write_text(json.dumps({**data, "factor": "tone"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not one of"):
        e6.declared_versions(d)
    meta.write_text(json.dumps(data), encoding="utf-8")
    shutil.copy(d / "v1" / "recall.md", d / "v6" / "recall.md")  # same text as the baseline
    with pytest.raises(ValueError, match="declared change"):
        e6.declared_versions(d)


def _m(g, a, r, lat=100.0):
    return {"grounded_rate": g, "abstain_when_insufficient": a, "answer_when_answerable": r, "latency_p50_ms": lat}


def _c(v, metric, lo, hi):
    return {"a": v, "b": "v1", "metric": metric, "diff": (lo + hi) / 2, "ci": [lo, hi], "mcnemar_p": None}


def test_selection_keeps_the_baseline_without_a_measurable_gain() -> None:
    metrics = {"v1": _m(0.8, 0.5, 0.7), "v2": _m(0.9, 0.6, 0.7)}
    out = e6.pick_best(metrics, [_c("v2", "grounded_rate", -0.02, 0.2)], ("v1", "v2"))  # interval includes zero
    assert out["selected"] == "v1" and out["qualifies"]["v2"].startswith("no: no measurable gain")


def test_selection_refuses_a_gain_bought_with_a_measurable_loss() -> None:
    metrics = {"v1": _m(0.8, 0.5, 0.7), "v2": _m(0.95, 0.6, 0.3)}
    comps = [_c("v2", "grounded_rate", 0.05, 0.2), _c("v2", "answer_when_answerable", -0.6, -0.2)]
    out = e6.pick_best(metrics, comps, ("v1", "v2"))
    assert out["selected"] == "v1" and "answer_when_answerable" in out["qualifies"]["v2"]


def test_selection_uses_the_weaker_abstention_side_and_breaks_ties_by_latency_then_order() -> None:
    assert e6.score(_m(0.8, 0.9, 0.5)) == pytest.approx(0.65)  # the lower side (0.5), not the mean of the sides
    assert e6.score(_m(0.8, None, 0.5)) is None
    metrics = {"v1": _m(0.8, 0.5, 0.7), "v2": _m(0.9, 0.7, 0.7, 90.0), "v3": _m(0.9, 0.7, 0.7, 80.0),
               "v4": _m(0.9, 0.7, 0.7, 80.0)}
    comps = [_c(v, "grounded_rate", 0.01, 0.2) for v in ("v2", "v3", "v4")]
    out = e6.pick_best(metrics, comps, ("v1", "v2", "v3", "v4"))
    assert out["selected"] == "v3"  # equal score; v3 and v4 are faster than v2, and v3 comes first
    metrics["v4"] = _m(0.95, 0.7, 0.7)
    assert e6.pick_best(metrics, comps, ("v1", "v2", "v3", "v4"))["selected"] == "v4"


def test_the_selection_rule_text_names_what_it_decides_on() -> None:
    for word in ("abstain_when_insufficient", "answer_when_answerable", "grounded_rate", "weaker abstention side",
                 "no measurable improvement"):
        assert word in e6.SELECTION_RULE
