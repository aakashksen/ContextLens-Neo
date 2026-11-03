"""E6 runner: prompt versions on dev, one factor each. No model is called for real."""
import json
import shutil
from pathlib import Path

import pytest

import tests.test_e2 as t2
from contextlens.eval.check_run import check
from contextlens.experiments import e6_prompts as e6
from contextlens.llm.client import LLMResult
from contextlens.llm.ollama_ctl import OllamaCtl
from contextlens.prompts.registry import get_prompt
from tests.test_e1 import HashEmbedder
from tests.test_e2 import NAMES, RETRIEVAL, Scripted, Wire, _cases, _git


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


# ---------------------------------------------------------------- the runner, with scripted models

repo = t2.repo  # the same dev repository, built the same way

CARD_CALLS = "You answer a question about an ongoing conversation"  # every version's system message starts with this


def _cards(llm: Scripted) -> list[list[dict]]:
    return [m for m in llm.calls if m[0]["content"].startswith(CARD_CALLS)]


class PerVersion(Scripted):
    """The v4 text makes every card abstain, so v4 differs from the others on the same cases."""

    def reply(self, name: str, user: str) -> dict:
        if name == "Card" and "Abstain only when no turn" in self.system:
            return {"mode": "recall", "status": "abstained", "findings": [], "missing": ["not said"]}
        return super().reply(name, user)

    def _raw(self, messages, schema, max_tokens, temperature, timeout_s):
        self.system = messages[0]["content"]
        return super()._raw(messages, schema, max_tokens, temperature, timeout_s)


def _go(repo: Path, *, writer=None, judge=None, wire=None, **kw):
    return e6.run_e6(root=repo, writer=writer or Scripted(), judge_llm=judge or Scripted("test-judge:1"),
                     embedder=HashEmbedder(), ctl=OllamaCtl(request=wire or Wire()), names=NAMES,
                     retrieval_config=RETRIEVAL, gpu_mb=lambda: None, say=lambda text: None, **kw)


def test_record_lists_every_version(repo: Path) -> None:
    writer = PerVersion()
    result = _go(repo, writer=writer)
    assert check("E6", root=repo, min_n=len(_cases()), split="dev") == []
    rec = json.loads(result["record"].read_text(encoding="utf-8"))
    versions = e6.declared_versions()
    assert sorted(rec["predictions"]) == sorted(v.version for v in versions) and rec["n"] == len(_cases())
    assert [a["setting"] for a in rec["attempts"]] == list(e6.VERSIONS) and all(a["ok"] for a in rec["attempts"])
    assert rec["prompt_versions"] == {v.version: v.text_sha256 for v in versions}  # hashes, so a label is not shared
    cfg = rec["config"]
    assert [(v["version"], v["factor"]) for v in cfg["versions"]] == [(v.version, v.factor) for v in versions]
    assert len({v["system_message_sha256"] for v in cfg["versions"]}) == len(versions)
    assert cfg["selection_rule"] == e6.SELECTION_RULE and cfg["selection"]["selected"] == result["selected"]
    assert {(c["a"], c["b"]) for c in rec["comparisons"]} == {(v, "v1") for v in e6.VERSIONS[1:]}
    assert rec["metrics"]["v4"]["answer_when_answerable"] == 0.0  # the version's text reached the model
    assert rec["metrics"]["v1"]["answer_when_answerable"] > 0.5
    assert set(cfg["modes"]) == set(e6.VERSIONS) and set(cfg["injection_findings"]) == set(e6.VERSIONS)
    assert rec["claude_calls"] == 0 and rec["config"]["judge"]["same_model_and_prompt_as_writer"] is False
    assert len(_cards(writer)) >= len(_cases()) * len(e6.VERSIONS)


def test_every_version_text_is_what_the_model_is_sent(repo: Path) -> None:
    writer = Scripted()
    _go(repo, writer=writer)
    sent = {m[0]["content"] for m in _cards(writer)}
    for v in e6.declared_versions():
        assert any(s.startswith(v.text) for s in sent), v.version
    assert len(sent) == len(e6.VERSIONS)


def test_a_version_the_model_cannot_answer_in_valid_json_is_an_attempt_not_a_stop(repo: Path) -> None:
    class Bad(Scripted):
        def _raw(self, messages, schema, max_tokens, temperature, timeout_s):
            if schema is not None and schema.__name__ == "Card" and "Keep the card short" in messages[0]["content"]:
                self.raw_calls += 1
                self.calls.append(messages)
                return LLMResult(text="not json", model=self.model)
            return super()._raw(messages, schema, max_tokens, temperature, timeout_s)

    result = _go(repo, writer=Bad())
    rec = json.loads(result["record"].read_text(encoding="utf-8"))
    bad = next(a for a in rec["attempts"] if a["setting"] == "v6")
    assert bad["ok"] is False and bad["error"] and "v6" not in rec["predictions"]
    assert len(rec["attempts"]) == len(e6.VERSIONS) and rec["config"]["unmeasured_versions"].keys() == {"v6"}
    assert check("E6", root=repo, min_n=len(_cases()), split="dev") == []


def test_a_broken_baseline_or_a_dead_writer_writes_nothing(repo: Path) -> None:
    with pytest.raises(e6.e2.RunBroken):
        _go(repo, writer=Scripted(die_after=5))
    assert not (repo / "results/runs").exists() or not list((repo / "results/runs").glob("E6-*"))


def test_a_missing_model_or_a_dirty_tree_stops_before_any_call(repo: Path) -> None:
    writer = Scripted()
    with pytest.raises(e6.e2.RunBroken, match="not installed"):
        _go(repo, writer=writer, wire=Wire(installed=["test-writer:1"]))
    (repo / "stray.txt").write_text("x", encoding="utf-8")
    with pytest.raises(e6.e2.RunBroken, match="uncommitted"):
        _go(repo, writer=writer)
    assert writer.raw_calls == 0


def test_rows_are_saved_as_they_arrive_and_a_rerun_at_the_same_commit_reuses_them(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = e6.write_run

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(e6, "write_run", boom)
    first = Scripted()
    with pytest.raises(RuntimeError, match="disk full"):
        _go(repo, writer=first)
    assert len((repo / e6.PARTIAL_REL).read_text(encoding="utf-8").splitlines()) == len(_cases()) * len(e6.VERSIONS)
    monkeypatch.setattr(e6, "write_run", real)
    again = Scripted()
    result = _go(repo, writer=again)
    assert _cards(again) == []  # no card was paid for twice
    assert check("E6", root=repo, min_n=len(_cases()), split="dev") == []
    assert json.loads(result["record"].read_text(encoding="utf-8"))["n"] == len(_cases())
    assert not (repo / e6.PARTIAL_REL).exists()


def test_rows_saved_at_another_commit_are_not_reused(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(e6, "write_run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop")))
    with pytest.raises(RuntimeError):
        _go(repo)
    (repo / "note.txt").write_text("a new commit", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "next")
    monkeypatch.undo()
    again = Scripted()
    _go(repo, writer=again)
    assert len(_cards(again)) >= len(_cases()) * len(e6.VERSIONS)


def test_call_counts_follow_the_loops() -> None:
    c = e6.call_counts(148, 7)
    assert c["writer_cards_min"] == 1036 and c["memory_extraction"] == 148
    assert c["writer_cards_worst"] == 1036 * e6.CALLS_PER_CARD_WORST


def test_the_runner_never_reads_the_test_split() -> None:
    src = Path(e6.__file__).read_text(encoding="utf-8")
    assert '"test"' not in src and "'test'" not in src and e6.SPLIT == "dev"
