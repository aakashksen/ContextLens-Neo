"""E6 judging stage: saved cards judged later by a fake `claude -p`. No model, no Ollama, no real call."""
import json
import threading
from pathlib import Path

import pytest

from contextlens.agents.graph import TeamAssistant
from contextlens.eval.check_run import check
from contextlens.experiments import e2_context as e2
from contextlens.experiments import e6_judge as j
from contextlens.experiments import e6_prompts as e6
from contextlens.experiments.judge_calibrate import UsageGuard
from contextlens.experiments.judge_check import Refused
from contextlens.llm.client import FakeLLM, LLMResult
from contextlens.memory.extract import MemoryStore
from contextlens.retrieval.hybrid import Retriever
from tests.test_e1 import HashEmbedder
from tests.test_e2 import RETRIEVAL, _cases, _git
from tests.test_e6 import PerVersion, Scripted, _go
from tests.test_e6 import repo as dev_repo  # noqa: F401  (a fixture)

ONLY = ("v1", "v3", "v4", "v6")


class FakeClaude(FakeLLM):
    """Answers the judge's schema; `rule` maps the claim text to the verdict. Counts its calls."""

    def __init__(self, model: str, rule=lambda text: True):
        super().__init__([])
        self.model, self.rule, self.n, self.lock = model, rule, 0, threading.Lock()
        self.calls_made = 0

    def _raw(self, messages, schema, max_tokens, temperature, timeout_s) -> LLMResult:
        with self.lock:
            self.n += 1
        text = messages[-1]["content"].split("<claim>", 1)[1].split("</claim>", 1)[0]
        return LLMResult(text=json.dumps({"supported": self.rule(text)}), model=self.model)


def _guard(read=lambda: 10.0) -> UsageGuard:
    return UsageGuard(90.0, read=read, interval_s=0.0)


@pytest.fixture
def generated(dev_repo: Path,  # noqa: F811
               monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    repo = dev_repo
    monkeypatch.setattr(e2, "load_retrieval_config", lambda path: RETRIEVAL)
    _go(repo, writer=PerVersion(), only=ONLY, generate_only=True)
    return repo, e6.head_commit(repo)


def _judge(repo: Path, commit: str, tmp_path: Path, haiku=None, sonnet=None, guard=None, **kw):
    return j.run_judge(root=repo, from_commit=commit, haiku=haiku or FakeClaude("haiku-x"), sonnet=sonnet,
                       guard=guard or _guard(), counter_path=tmp_path.parent / (tmp_path.name + "-calls.json"), say=lambda t: None, concurrency=3, **kw)


def test_saved_cards_are_judged_and_the_record_passes_check_run(generated, tmp_path: Path) -> None:
    repo, commit = generated
    haiku = FakeClaude("haiku-x")
    result = _judge(repo, commit, tmp_path, haiku=haiku)
    assert check("E6", root=repo, min_n=len(_cases()), split="dev") == []
    rec = json.loads(result["record"].read_text(encoding="utf-8"))
    cfg = rec["config"]
    assert sorted(rec["predictions"]) == sorted(ONLY) and cfg["versions_not_run"] == ["v2", "v5", "v7"]
    assert cfg["judge"]["model"] == "haiku-x" and cfg["judge"]["calls"] == haiku.n == rec["claude_calls"]
    assert cfg["judged_after_generation"]["generating_commit"] == commit
    notes = " ".join(cfg["notes"])
    assert "entry 56" in notes and "JCAL-20261105T044839474906Z" in notes and "a reading, not proof" in notes
    assert "local judge alone" not in notes
    assert cfg["selection_rule"] == e6.SELECTION_RULE and cfg["selection"]["selected"] == result["selected"]
    assert rec["metrics"]["v4"]["answer_when_answerable"] == 0.0  # the saved cards are what was judged
    assert "sonnet_check" not in cfg


def test_a_rerun_regrades_nothing_and_a_stop_resumes(generated, tmp_path: Path) -> None:
    repo, commit = generated
    first = FakeClaude("haiku-x")
    reads = iter([10.0] + [95.0] * 100)
    with pytest.raises(j.JudgeStopped, match="weekly usage"):
        _judge(repo, commit, tmp_path, haiku=first, guard=_guard(lambda: next(reads)))
    assert not list((repo / "results/runs").glob("E6-*.json")) if (repo / "results/runs").exists() else True
    second = FakeClaude("haiku-x")
    result = _judge(repo, commit, tmp_path, haiku=second)
    third = FakeClaude("haiku-x")
    _judge(repo, commit, tmp_path, haiku=third)
    assert first.n + second.n == json.loads(result["record"].read_text(encoding="utf-8"))["claude_calls"]
    assert third.n == 0  # every verdict was saved
    rows = (repo / j.JUDGE_DIR_REL).glob("*.haiku.rows.jsonl")
    text = "".join(p.read_text(encoding="utf-8") for p in rows)
    assert "the plan" not in text  # ids and verdicts only


def test_weekly_usage_at_the_limit_refuses_before_any_call(generated, tmp_path: Path) -> None:
    repo, commit = generated
    haiku = FakeClaude("haiku-x")
    with pytest.raises(Refused, match="weekly usage"):
        _judge(repo, commit, tmp_path, haiku=haiku, guard=_guard(lambda: 91.0))
    assert haiku.n == 0


def test_the_call_cap_is_checked_before_grading(generated, tmp_path: Path) -> None:
    repo, commit = generated
    haiku = FakeClaude("haiku-x")
    with pytest.raises(Refused, match="cap"):
        j.run_judge(root=repo, from_commit=commit, haiku=haiku, guard=_guard(), counter_path=tmp_path.parent / (tmp_path.name + "-calls.json"), cap=3,
                    say=lambda t: None)
    assert haiku.n == 0


def test_a_change_to_the_prompts_or_card_code_since_the_generating_commit_refuses(generated, tmp_path: Path) -> None:
    repo, commit = generated
    assert j.card_code_changed(repo, commit) == []
    (repo / "docs").mkdir()
    (repo / "docs/readme.md").write_text("not card code", encoding="utf-8")
    p = repo / "src/contextlens/prompts/v9.md"
    p.parent.mkdir(parents=True)
    p.write_text("a new prompt", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "prompt change")
    haiku = FakeClaude("haiku-x")
    with pytest.raises(Refused, match="prompts or card code changed"):
        _judge(repo, commit, tmp_path, haiku=haiku)
    assert haiku.n == 0
    assert j.card_code_changed(repo, commit) == ["src/contextlens/prompts/v9.md"]  # the docs file is not card code


def test_a_change_outside_the_card_code_is_allowed_and_a_wrong_commit_is_refused(generated, tmp_path: Path) -> None:
    repo, commit = generated
    (repo / "note.txt").write_text("unrelated", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "note")
    rec = json.loads(_judge(repo, commit, tmp_path)["record"].read_text(encoding="utf-8"))
    assert rec["config"]["judged_after_generation"]["generating_commit"] == commit != rec["git_commit"]
    with pytest.raises(Refused, match="not a commit"):
        _judge(repo, "0123abc0123abc", tmp_path)


def test_cards_saved_at_another_commit_or_missing_are_not_judged(generated, tmp_path: Path) -> None:
    repo, commit = generated
    (repo / "note.txt").write_text("next", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "next")
    with pytest.raises(Refused, match="no saved cards"):
        _judge(repo, e6.head_commit(repo), tmp_path)  # rows are keyed by the generating commit
    path = repo / e6.PARTIAL_REL
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(ln for ln in lines if '"v6"' not in ln or '"r0"' not in ln) + "\n", encoding="utf-8")
    with pytest.raises(Refused, match=r"saved cards are missing for: v6 \(1 cases\)"):
        _judge(repo, commit, tmp_path)


def test_rows_can_be_read_from_another_path(generated, tmp_path: Path) -> None:
    repo, commit = generated
    elsewhere = tmp_path.parent / (tmp_path.name + "-elsewhere") / "rows.jsonl"
    elsewhere.parent.mkdir()
    (repo / e6.PARTIAL_REL).replace(elsewhere)
    with pytest.raises(Refused, match="no saved cards"):
        _judge(repo, commit, tmp_path)
    assert _judge(repo, commit, tmp_path, rows_path=elsewhere)["record"].exists()


def test_sonnet_grades_the_chosen_version_and_v1_only_and_does_not_replace_haiku(
    generated, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, commit = generated
    real = e6.pick_best
    monkeypatch.setattr(e6, "pick_best", lambda m, c, o=e6.VERSIONS: (
        {"selected": "v3", "scores": {}, "qualifies": {}} if len(o) > 2 else real(m, c, o)))
    haiku, sonnet = FakeClaude("haiku-x"), FakeClaude("sonnet-x", rule=lambda text: False)
    result = _judge(repo, commit, tmp_path, haiku=haiku, sonnet=sonnet)
    rec = json.loads(result["record"].read_text(encoding="utf-8"))
    sc = rec["config"]["sonnet_check"]
    assert sc["versions"] == ["v1", "v3"] and sc["model"] == "sonnet-x"
    assert sc["same_version"] is (sc["selected_under_sonnet"] == "v3") and sc["claims_graded"] == sonnet.n > 0
    assert sonnet.n < haiku.n  # not every version
    assert rec["config"]["judge"]["model"] == "haiku-x" and rec["claude_calls"] == haiku.n + sonnet.n
    rows = {v: [json.loads(x) for x in (repo / rec["predictions"][v]).read_text(encoding="utf-8").splitlines()] for v in ONLY}
    for v, ps in rows.items():
        cards = [p for p in ps if p["card"] and p["card"]["findings"]]
        assert all(p["claim_supported"] and all(p["claim_supported"]) for p in cards)  # Haiku's, unchanged
        assert all((p.get("cloud_supported") is not None) == (v in ("v1", "v3")) for p in cards), v
        assert all(not any(p.get("cloud_supported") or []) for p in cards)
    assert check("E6", root=repo, min_n=len(_cases()), split="dev") == []


def test_when_haiku_keeps_the_baseline_sonnet_is_not_asked(generated, tmp_path: Path) -> None:
    repo, commit = generated
    sonnet = FakeClaude("sonnet-x")
    rec = json.loads(_judge(repo, commit, tmp_path, sonnet=sonnet)["record"].read_text(encoding="utf-8"))
    assert rec["config"]["selection"]["selected"] == "v1" and sonnet.n == 0
    assert rec["config"]["sonnet_check"]["same_version"] is None


def test_the_shipping_team_follow_up_builds_and_its_saved_cards_are_judged(
    generated, tmp_path: Path
) -> None:
    repo, commit = generated
    assert e6.FOLLOW_UP_SYSTEM == "team_shipping_changed" and e6.SYSTEMS == ("single", "retrieval_memory")
    sc = _cases()[0]
    ret = Retriever(e2.build_store(sc), HashEmbedder())
    built = {v: e6._build_assistant(e6.FOLLOW_UP_SYSTEM, v, sc, Scripted(), ret, MemoryStore(), RETRIEVAL) for v in ("v1", "v3")}
    for v, team in built.items():
        assert isinstance(team, TeamAssistant) and team.memory_gap_abstains and team.tools.checker is not None
        assert team._specs["memory"][1].startswith(e6.declared_versions()[0].text if v == "v1" else e6.declared_versions()[2].text)
    assert e6._system_message(built["v1"]) != e6._system_message(built["v3"])
    assert e6.call_counts(10, 2, e6.FOLLOW_UP_SYSTEM)["writer_cards_worst"] > 0
    # the same stage judges rows keyed by that system
    path = repo / e6.PARTIAL_REL
    keep = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()]
    path.write_text("".join(json.dumps({**e, "system": e6.FOLLOW_UP_SYSTEM}) + "\n" for e in keep if e["version"] in ("v1", "v3")),
                    encoding="utf-8")
    rec = json.loads(_judge(repo, commit, tmp_path, system=e6.FOLLOW_UP_SYSTEM)["record"].read_text(encoding="utf-8"))
    assert rec["config"]["system"] == e6.FOLLOW_UP_SYSTEM and sorted(rec["predictions"]) == ["v1", "v3"]
    assert check("E6", root=repo, min_n=len(_cases()), split="dev") == []
    with pytest.raises(e6.e2.RunBroken, match="unknown system"):
        _go(repo, system="team")
