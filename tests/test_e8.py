"""E8, the final test: the freeze, the once-only rule, resumable rows, the Sonnet scope, McNemar on false alerts.
No model, no Ollama, no `claude`, and no test split is read: scenarios here are invented fixtures."""
import json
import subprocess
from pathlib import Path

import pytest

from contextlens.api import production
from contextlens.experiments import e2_context as e2
from contextlens.experiments import e8_final as e8
from contextlens.experiments.judge_check import Refused
from contextlens.replay.runner import Prediction
from contextlens.schemas import ButtonRequest, Conversation, Scenario, Turn


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=root, check=True, capture_output=True)


def _scenario(sid: str, gold: str | None = None, source: str = "synthetic") -> Scenario:
    turns = [Turn(id=f"{sid}-t{i}", conversation_id=sid, seq=i, speaker="A", text=f"w{i}", start_ms=i * 1000,
                  end_ms=i * 1000 + 900, source="synthetic") for i in range(3)]
    return Scenario(id=sid, conversation=Conversation(id=sid, turns=turns, source=source),
                    request=ButtonRequest(conversation_id=sid, as_of_seq=2), category="c", gold_evidence_turn_ids=[],
                    gold_label=gold, gold_answer=None, origin="synthetic")  # type: ignore[arg-type]


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    (tmp_path / "README").write_text("x", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def _freeze_file(root: Path, body: dict | None = None) -> None:
    (root / "configs").mkdir(exist_ok=True)
    (root / e8.FREEZE_REL).write_text(json.dumps(body or {"kind": "e8-freeze"}), encoding="utf-8")


def _commit_all(root: Path, msg: str = "c") -> None:
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", msg)


def test_refuses_when_config_not_committed(repo: Path) -> None:
    with pytest.raises(Refused, match="does not exist"):
        e8.committed_freeze(repo)
    _freeze_file(repo)
    with pytest.raises(Refused, match="not committed"):
        e8.committed_freeze(repo)
    _commit_all(repo)
    assert e8.committed_freeze(repo)["kind"] == "e8-freeze"
    _freeze_file(repo, {"kind": "e8-freeze", "prompt_version": "v4"})  # changed after the commit
    with pytest.raises(Refused, match="differs from the committed"):
        e8.committed_freeze(repo)


def test_write_stops_on_an_uncommitted_config_before_any_test_data(repo: Path) -> None:
    _freeze_file(repo)  # untracked: the tree is dirty
    with pytest.raises(Refused):
        e8.run_write(root=repo, writer=None, embedder=None, ctl=None, make_hf_writer=None, make_checker=None,  # type: ignore[arg-type]
                     run_conditions="nothing else")


def _test_record(root: Path, run_id: str = "E8-20261110T000000000000Z", note: str | None = None) -> None:
    runs = root / "results/runs"
    runs.mkdir(parents=True, exist_ok=True)
    config = {"override_note": note} if note else {}
    (runs / f"{run_id}.json").write_text(json.dumps({"run_id": run_id, "split": "test", "config": config}), encoding="utf-8")


def test_refuses_second_test_run_without_override_note(repo: Path) -> None:
    commit = e8.head_commit(repo)
    e8.require_first_test_run(repo, commit, None)  # nothing has run
    _test_record(repo)
    with pytest.raises(Refused, match="already run"):
        e8.require_first_test_run(repo, commit, None)
    with pytest.raises(Refused, match="already run"):
        e8.require_first_test_run(repo, commit, "   ")
    e8.require_first_test_run(repo, commit, "rerun after a label fix, approved by the person")


def test_run_write_refuses_a_second_run_through_the_whole_gate(repo: Path) -> None:
    _freeze_file(repo)
    _test_record(repo)
    _commit_all(repo)
    with pytest.raises(Refused, match="already run"):
        e8.run_write(root=repo, writer=None, embedder=None, ctl=None, make_hf_writer=None, make_checker=None,  # type: ignore[arg-type]
                     run_conditions="nothing else")


def test_rows_of_another_commit_count_as_a_second_run_but_the_same_commit_is_a_resume(repo: Path) -> None:
    commit = e8.head_commit(repo)
    ran: list[str] = []

    def make(sc: Scenario) -> Prediction:
        ran.append(sc.id)
        return Prediction(scenario_id=sc.id, ranked_turn_ids=[f"{sc.id}-t0"])

    scs = [_scenario("a"), _scenario("b")]
    e8.run_rows(repo, commit, "search:qmsum", e8.SEARCH_SYSTEM, scs[:1], make, lambda t: None)
    e8.require_first_test_run(repo, commit, None)  # same commit: a resume
    rows = e8.run_rows(repo, commit, "search:qmsum", e8.SEARCH_SYSTEM, scs, make, lambda t: None)
    assert ran == ["a", "b"] and [r.scenario_id for r in rows] == ["a", "b"]  # "a" was not run again
    e8.run_rows(repo, commit, "search:qmsum", e8.SEARCH_SYSTEM, scs, make, lambda t: None)
    assert ran == ["a", "b"]
    with pytest.raises(Refused, match="saved rows of commit"):
        e8.require_first_test_run(repo, "f" * 40, None)


def test_freeze_refuses_trained_models_that_do_not_exist(repo: Path, tmp_path: Path) -> None:
    gone = tmp_path / "nowhere"
    with pytest.raises(Refused, match="T6.2 adapter"):
        e8.build_freeze(repo, adapter_base=repo, adapter=gone, checker=repo, checker_threshold=0.5)
    with pytest.raises(Refused, match="T5.2 consistency"):
        e8.build_freeze(repo, adapter_base=repo, adapter=repo, checker=gone, checker_threshold=0.5)


def _frozen_repo(repo: Path, adapter: Path, checker: Path) -> dict:
    _fixture_test_file(repo)
    (repo / "configs").mkdir(exist_ok=True)
    (repo / e2.CONFIG_REL).write_text('{"mode": "hybrid"}', encoding="utf-8")
    for src in ("scenarios", "qmsum", "decode"):
        rel = e8.SCENARIO_SOURCES[src][0]
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(json.dumps({"splits": {"test": {"hash": f"h-{src}"}}}), encoding="utf-8")
    return e8.build_freeze(repo, adapter_base=adapter, adapter=adapter, checker=checker, checker_threshold=0.5)


def test_freeze_records_the_choices_and_checks_the_trained_files_again(repo: Path, tmp_path: Path) -> None:
    adapter, checker = tmp_path / "adapter", tmp_path / "checker"
    adapter.mkdir()
    checker.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter-v1")
    (checker / "model.safetensors").write_bytes(b"checker-v1")
    freeze = _frozen_repo(repo, adapter, checker)
    assert freeze["systems"] == {"primary": "team_shipping_changed", "others": ["single", "team_trained"]}
    assert freeze["assistant_setting"]["value"] in production.ASSISTANTS  # the build the product can make (entry 57)
    assert freeze["prompt_version"] == "v3"
    assert freeze["test_hashes"] == {"scenarios": "h-scenarios", "qmsum": "h-qmsum", "decode": "h-decode"}
    e8.check_inputs(repo, freeze)
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter-v2")
    with pytest.raises(Refused, match="not the one that was frozen"):
        e8.check_inputs(repo, freeze)
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter-v1")
    (checker / "model.safetensors").write_bytes(b"checker-v2")
    with pytest.raises(Refused, match="not the one that was frozen"):
        e8.check_inputs(repo, freeze)
    for gone in (adapter, checker):
        for f in gone.iterdir():
            f.unlink()
        gone.rmdir()
    with pytest.raises(Refused, match="not found"):
        e8.check_inputs(repo, freeze)


def test_sonnet_sample_is_seeded_and_the_extra_step_continues_it_without_overlap() -> None:
    ids = [f"c{i}" for i in range(1000)]
    a, b = e8.sonnet_sample(ids, 300, 13), e8.sonnet_sample(ids, 300, 13)
    assert a == b and len(a) == 300 and set(a) <= set(ids)
    assert e8.sonnet_sample(ids, 300, 14) != a
    assert len(e8.sonnet_sample(ids[:120], 300)) == 120  # fewer claims than the sample: all of them
    more = e8.sonnet_extra(ids, 13, 300, 200)
    assert len(more) == 200 and not set(more) & set(a)  # additive: new claims only
    assert set(e8.sonnet_sample(ids, 500, 13)) == set(a) | set(more)  # and the original sample is a prefix of the order
    assert e8.sonnet_extra(ids[:120], 13, 300, 50) == []


def test_freeze_fixes_the_sonnet_sample_the_seed_and_the_weekly_stop(repo: Path, tmp_path: Path) -> None:
    import inspect

    adapter = tmp_path / "a"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"1")
    (adapter / "model.safetensors").write_bytes(b"1")
    judging = _frozen_repo(repo, adapter, adapter)["judging"]
    assert (judging["sonnet_sample"], judging["sample_seed"], judging["max_weekly_pct"]) == (300, 13, 98.0)
    assert "sonnet_scope" not in inspect.signature(e8.build_freeze).parameters  # no other way to choose
    with pytest.raises(SystemExit):
        e8.main(["freeze", "--adapter-base", "x", "--adapter", "x", "--checker", "x", "--sonnet-scope", "all"])


def test_judge_refuses_a_weekly_stop_other_than_the_frozen_one(judged_repo: tuple[Path, str, list[Scenario]], tmp_path: Path) -> None:
    from contextlens.experiments.judge_calibrate import UsageGuard
    from tests.test_e6_judge import FakeClaude

    root, commit, _ = judged_repo
    with pytest.raises(Refused, match="weekly stop"):
        e8.run_judge(root=root, from_commit=commit, haiku=FakeClaude("h"), sonnet=FakeClaude("s"),
                     guard=UsageGuard(95.0, read=lambda: 10.0, interval_s=0.0), counter_path=tmp_path / "c.json", say=lambda t: None)  # the old 95 is no longer allowed


def test_extra_report_keeps_the_original_figure_and_states_the_extra() -> None:
    haiku = {f"c{i}": i % 2 == 0 for i in range(6)}
    original = {"c0": True, "c1": True, "c2": True}  # agrees on c0, c2; not c1
    extra = {"c3": False, "c4": False, "c5": False}  # agrees on c3, c4; not c5
    out = e8.extra_report(haiku, original, extra)
    assert out["original"] == e8.agreement(haiku, original) and out["original"]["claims"] == 3
    assert out["extra"]["claims"] == 3 and out["combined"]["claims"] == 6
    assert out["combined"]["agreement"] == pytest.approx(4 / 6)
    assert "does not replace" in out["note"] and "extra" in out["note"]


def test_agreement_counts_the_direction_of_disagreement() -> None:
    haiku = {"a": True, "b": True, "c": False, "d": False, "e": True}
    sonnet = {"a": True, "b": False, "c": False, "d": True}
    out = e8.agreement(haiku, sonnet)  # e is not graded by Sonnet
    assert out["claims"] == 4 and out["agreement"] == 0.5
    assert (out["haiku_yes_sonnet_no"], out["haiku_no_sonnet_yes"]) == (1, 1)
    assert out["kappa"] == 0.0


def _label_row(sid: str, label: str) -> Prediction:
    return Prediction(scenario_id=sid, label=label)  # type: ignore[arg-type]


def test_false_alert_comparison_carries_mcnemar_and_needs_both_to_agree() -> None:
    scs = [_scenario(f"s{i}", "consistent") for i in range(40)]
    quiet = [_label_row(s.id, "consistent") for s in scs]
    noisy = [_label_row(s.id, "contradiction" if i < 12 else "consistent") for i, s in enumerate(scs)]
    out = e8.false_alert_compare(scs, noisy, quiet, "trained", "tfidf")
    assert out is not None and out["trained_only"] == 12 and out["tfidf_only"] == 0
    assert out["mcnemar_p"] < 0.01 and out["verdict"] == "worse"  # more false alerts is worse
    few = [_label_row(s.id, "contradiction" if i < 2 else "consistent") for i, s in enumerate(scs)]
    small = e8.false_alert_compare(scs, few, quiet, "trained", "tfidf")
    assert small is not None and small["mcnemar_p"] >= 0.05 and small["verdict"] == "no clear difference"


def test_longmemeval_cases_are_searched_only() -> None:
    scs = [_scenario("a"), _scenario("b", source="longmemeval"), _scenario("c")]
    cards, lme = e8.split_cards(scs)
    assert [s.id for s in cards] == ["a", "c"] and [s.id for s in lme] == ["b"]


# ---------------------------------------------------------------- the judge command, end to end on fixtures

def _answer(sid: str, text: str) -> Prediction:
    from contextlens.schemas import Card, EvidenceRef, Finding

    f = Finding(agent="single", kind="recall", statement=text, evidence=[EvidenceRef(turn_id=f"{sid}-t0", quote="w0", speaker="A", start_ms=0)],
                label=None, confidence=0.9)
    return Prediction(scenario_id=sid, card=Card(mode="recall", status="answered", findings=[f]), stage_latency_ms={"total": 5})


def _abstain(sid: str) -> Prediction:
    from contextlens.schemas import Card

    return Prediction(scenario_id=sid, card=Card(mode="recall", status="abstained", findings=[], missing=["no"]), stage_latency_ms={"total": 5})


@pytest.fixture
def judged_repo(tmp_path: Path) -> tuple[Path, str, list[Scenario]]:
    return _make_judged(tmp_path, trained_team=True)


@pytest.fixture
def judged_repo_two(tmp_path: Path) -> tuple[Path, str, list[Scenario]]:
    return _make_judged(tmp_path, trained_team=False)


def _make_judged(tmp_path: Path, trained_team: bool, search_later: bool = False) -> tuple[Path, str, list[Scenario]]:
    from contextlens.eval.records import write_local_manifest, write_scenarios_manifest

    root = tmp_path / "r"
    (root / "datasets/scenarios").mkdir(parents=True)
    cards = [_scenario(f"c{i}", "insufficient" if i % 2 else "consistent") for i in range(6)]
    lme = [_scenario("m0", "consistent", source="longmemeval")]
    qm = [_scenario("q0", "consistent")]
    dec = [_scenario(f"d{i}", "contradiction" if i % 2 else "consistent") for i in range(4)]
    for split in ("dev", "test"):
        (root / "datasets/scenarios" / f"{split}.jsonl").write_text(
            "".join(s.model_dump_json() + "\n" for s in ([_scenario("dev0")] if split == "dev" else cards + lme)), encoding="utf-8")
    write_scenarios_manifest(root)
    write_local_manifest("qmsum", {"test": qm}, root)
    write_local_manifest("decode", {"test": dec}, root)
    (root / e2.CONFIG_REL).parent.mkdir(exist_ok=True)
    (root / e2.CONFIG_REL).write_text('{"mode": "hybrid"}', encoding="utf-8")
    models = tmp_path / "m"
    models.mkdir()
    (models / "adapter_model.safetensors").write_bytes(b"x")
    (models / "model.safetensors").write_bytes(b"x")
    freeze = e8.build_freeze(root, adapter_base=models, adapter=models, checker=models, checker_threshold=0.5, trained_team=trained_team)
    e8.write_freeze(root, freeze)
    (root / ".gitignore").write_text("results/partial/\nresults/logs/\ndatasets/raw/\n", encoding="utf-8")
    _git(root, "init", "-q")
    _commit_all(root)
    commit = e8.head_commit(root)
    cards_commit = commit
    (root / e8.ROWS_REL).parent.mkdir(parents=True, exist_ok=True)
    for n in e8.systems_of(freeze):
        for sc in cards:
            row = _abstain(sc.id) if sc.gold_label == "insufficient" else _answer(sc.id, f"{n} says w0")
            e2._append_durable(root / e8.ROWS_REL, json.dumps({"commit": cards_commit, "subset": "cards", "winner": n, "model": "E8", "calls": 0,
                                                              "row": json.loads(row.model_dump_json())}))
    if search_later:  # the search stage reran after a commit that changes no card code
        (root / "NOTES.md").write_text("search-stage fix", encoding="utf-8")
        _commit_all(root)
        commit = e8.head_commit(root)
    (root / e8.ROWS_REL.with_name("e8-write.json")).write_text(json.dumps({"commit": commit, "cards_commit": cards_commit, "consistency_checker_device": "cpu"}), encoding="utf-8")
    parts = {("search:qmsum", e8.SEARCH_SYSTEM): qm, (f"search:{e8.LME}", e8.SEARCH_SYSTEM): lme}
    for (part, system), scs in parts.items():
        for sc in scs:
            row = Prediction(scenario_id=sc.id, ranked_turn_ids=[f"{sc.id}-t0"], stage_latency_ms={"total": 3})
            e2._append_durable(root / e8.ROWS_REL, json.dumps({"commit": commit, "subset": part, "winner": system, "model": "E8", "calls": 0,
                                                              "row": json.loads(row.model_dump_json())}))
    for part, scs in (("detector:scenarios", cards), ("detector:decode", dec)):
        for system in ("trained", *freeze["detectors"][part.split(":")[1]]):
            for sc in scs:
                e2._append_durable(root / e8.ROWS_REL, json.dumps({
                    "commit": commit, "subset": part, "winner": system, "model": "E8", "calls": 0,
                    "row": json.loads(_label_row(sc.id, sc.gold_label or "consistent").model_dump_json())}))
    return root, commit, cards


def test_judge_writes_one_record_with_every_system_and_each_source_apart(judged_repo: tuple[Path, str, list[Scenario]], tmp_path: Path) -> None:
    from contextlens.experiments.judge_calibrate import UsageGuard
    from tests.test_e6_judge import FakeClaude

    root, commit, cards = judged_repo
    haiku, sonnet = FakeClaude("haiku-x"), FakeClaude("sonnet-x", rule=lambda t: "single" not in t)
    path = e8.run_judge(root=root, from_commit=commit, haiku=haiku, sonnet=sonnet, guard=UsageGuard(98.0, read=lambda: 10.0, interval_s=0.0),
                        counter_path=tmp_path / "calls.json", say=lambda t: None, concurrency=2)
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert rec["split"] == "test" and rec["experiment"] == "E8" and set(rec["predictions"]) == set(e8.SYSTEMS)
    assert rec["n"] == 6  # the LongMemEval case is searched only: no card exists for it
    cfg = rec["config"]
    assert set(cfg["benchmarks"]) == {"qmsum", "longmemeval", "decode", "consistency_scenarios"}
    assert cfg["benchmarks"]["decode"]["systems"] == ["trained", "tfidf"]
    assert cfg["benchmarks"]["decode"]["false_alert_comparison"]["tfidf"]["mcnemar_p"] is not None
    assert cfg["benchmarks"]["consistency_scenarios"]["systems"] == ["trained", "prompted", "tfidf"]
    assert cfg["consistency_checker_device"] == "cpu" and "first use" in cfg["freeze"]["test_manifests"]["note"]
    assert cfg["sonnet"]["sample"] == 300 and cfg["sonnet"]["claims_graded"] == 9  # fewer claims than the sample: all of them
    assert cfg["sonnet"]["agreement_with_haiku"]["haiku_yes_sonnet_no"] == 3  # Sonnet's rule rejects `single`'s three claims
    assert rec["subset"] == "e8-cards" and cfg["freeze"]["cards_subset"]["name"] == "e8-cards"
    from contextlens.eval.check_run import check

    assert check("E8", root=root, min_n=1, split="test") == []  # the record passes the full check
    assert haiku.n == 9 and sonnet.n == 9  # 3 answered cards on each of 3 systems, one claim per call
    assert "failed the written calibration test" in cfg["judge_note"] and "reading, not proof" in cfg["judge_note"]
    with pytest.raises(Refused, match="no more claims"):  # all 9 claims were in the original sample of 300
        e8.run_sonnet_extra(root=root, from_commit=commit, sonnet=sonnet, more=100, guard=UsageGuard(98.0, read=lambda: 10.0, interval_s=0.0),
                            counter_path=tmp_path / "calls.json", say=lambda t: None)
    with pytest.raises(Refused, match="already run"):  # the record just written makes any second run need a note
        e8.run_judge(root=root, from_commit=commit, haiku=haiku, sonnet=sonnet, guard=UsageGuard(98.0, read=lambda: 10.0, interval_s=0.0),
                     counter_path=tmp_path / "calls.json", say=lambda t: None)


# ---------------------------------------------------------------- the card-case manifest, the product prompt (coordinator, 2026-11-06)

def _fixture_test_file(root: Path) -> None:
    (root / "datasets/scenarios").mkdir(parents=True, exist_ok=True)
    lines = []
    for sid, src in (("t1", "synthetic"), ("t2", "longmemeval"), ("t3", "synthetic")):
        d = json.loads(_scenario(sid, "consistent", source=src).model_dump_json())
        d["conversation"]["turns"][0]["text"] = "LEAKCHECK private words"
        lines.append(json.dumps(d))
    (root / "datasets/scenarios/test.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_cards_manifest_holds_ids_only_and_leaves_out_longmemeval(repo: Path) -> None:
    _fixture_test_file(repo)
    path = e8.write_cards_subset(repo)
    assert path == repo / "datasets/splits/subsets/e8-cards.json"
    text = path.read_text(encoding="utf-8")
    manifest = json.loads(text)
    assert manifest == {"name": "e8-cards", "split": "test", "ids": ["t1", "t3"]}
    assert "LEAKCHECK" not in text and "private" not in text


def test_freeze_names_the_manifest_and_the_record_uses_it(repo: Path, tmp_path: Path) -> None:
    adapter = tmp_path / "a"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"1")
    (adapter / "model.safetensors").write_bytes(b"1")
    freeze = _frozen_repo(repo, adapter, adapter)
    sub = freeze["cards_subset"]
    assert sub["name"] == "e8-cards" and sub["path"] == "datasets/splits/subsets/e8-cards.json"
    assert sub["sha256"] == e8.sha256_path(repo / sub["path"]) and "`conversation.source`" in sub["reads"]
    assert "same commit" in sub["committed"]


def test_refuses_when_the_manifest_is_not_committed_with_the_config(repo: Path, tmp_path: Path) -> None:
    adapter = tmp_path / "a"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"1")
    (adapter / "model.safetensors").write_bytes(b"1")
    freeze = _frozen_repo(repo, adapter, adapter)
    _freeze_file(repo, freeze)
    _git(repo, "add", str(e8.FREEZE_REL))
    _git(repo, "commit", "-q", "-m", "config only")
    with pytest.raises(Refused, match="e8-cards.json is not committed"):
        e8.committed_freeze(repo)
    _commit_all(repo)
    assert e8.committed_freeze(repo)["cards_subset"]["name"] == "e8-cards"
    (repo / freeze["cards_subset"]["path"]).write_text('{"ids": []}', encoding="utf-8")
    with pytest.raises(Refused, match="differs from the committed"):
        e8.committed_freeze(repo)


def test_product_prompt_is_v3_and_the_default_assistant_is_unchanged() -> None:
    assert production.PROMPT_VERSION == "v3" == e8.PROMPT  # E8 measures the prompt the product ships
    assert production.DEFAULT_ASSISTANT == "team_rag"


# ---------------------------------------------------------------- no lost work (the person, via the coordinator, 2026-11-06)

class _Abstains:
    def answer(self, req, ctx):  # noqa: ANN001
        from contextlens.schemas import Card

        return Card(mode="recall", status="abstained", findings=[], missing=["no"])


def test_card_rows_are_saved_as_they_finish_so_an_interrupted_run_resumes(repo: Path) -> None:
    commit = e8.head_commit(repo)
    scs = [_scenario(f"k{i}") for i in range(4)]
    made: list[str] = []

    def make_then_die(sc: Scenario):
        if sc.id == "k2":
            raise KeyboardInterrupt  # the machine is stopped in the middle
        made.append(sc.id)
        return _Abstains()

    with pytest.raises(KeyboardInterrupt):
        e8.run_part(repo, commit, "cards", e8.SINGLE, scs, make_then_die, lambda t: None)
    saved = e8.all_saved(repo / e8.ROWS_REL, commit, "cards", e8.SINGLE, {s.id for s in scs})
    assert sorted(saved) == ["k0", "k1"]  # on disk before the run ended
    made.clear()
    rows = e8.run_part(repo, commit, "cards", e8.SINGLE, scs, lambda sc: (made.append(sc.id), _Abstains())[1], lambda t: None)
    assert made == ["k2", "k3"] and [r.scenario_id for r in rows] == ["k0", "k1", "k2", "k3"]  # finished rows are not run again


def test_search_and_detector_rows_are_saved_as_they_finish(repo: Path) -> None:
    commit = e8.head_commit(repo)
    scs = [_scenario(f"s{i}") for i in range(3)]

    def die_on_second(sc: Scenario) -> Prediction:
        if sc.id == "s1":
            raise KeyboardInterrupt
        return Prediction(scenario_id=sc.id, label="consistent")

    for part, system in (("search:qmsum", e8.SEARCH_SYSTEM), ("detector:decode", "trained"), ("detector:scenarios", "prompted")):
        with pytest.raises(KeyboardInterrupt):
            e8.run_rows(repo, commit, part, system, scs, die_on_second, lambda t: None)
        assert sorted(e8.all_saved(repo / e8.ROWS_REL, commit, part, system, {s.id for s in scs})) == ["s0"]


def test_judge_refuses_a_dirty_tree_before_any_call(judged_repo: tuple[Path, str, list[Scenario]], tmp_path: Path) -> None:
    from contextlens.experiments.judge_calibrate import UsageGuard
    from tests.test_e6_judge import FakeClaude

    root, commit, _ = judged_repo
    (root / "stray.txt").write_text("uncommitted", encoding="utf-8")
    haiku, sonnet = FakeClaude("h"), FakeClaude("s")
    with pytest.raises(Refused, match="uncommitted"):
        e8.run_judge(root=root, from_commit=commit, haiku=haiku, sonnet=sonnet, guard=UsageGuard(98.0, read=lambda: 10.0, interval_s=0.0),
                     counter_path=tmp_path / "c.json", say=lambda t: None)
    assert haiku.n == 0 and sonnet.n == 0
    assert not (root / e8.JUDGE_DIR_REL).exists()


def test_write_refuses_a_dirty_tree_before_a_log_a_model_or_a_row(repo: Path) -> None:
    _freeze_file(repo)
    with pytest.raises(Refused, match="uncommitted"):
        e8.run_write(root=repo, writer=None, embedder=None, ctl=None, make_hf_writer=None, make_checker=None,  # type: ignore[arg-type]
                     run_conditions="nothing else")
    assert not (repo / "results").exists()


def test_progress_goes_to_a_log_under_results_logs_that_git_ignores(repo: Path) -> None:
    seen: list[str] = []
    say = e8.progress_log(repo, "write", "abc123456789def", seen.append)
    say("cards/single: 10/136")
    say("search:qmsum/hybrid_t1: 3 rows")
    path = repo / "results/logs/e8-write-abc123456789.log"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and lines[0].endswith("cards/single: 10/136") and lines[0][:4].isdigit()  # a timestamp first
    assert seen == ["cards/single: 10/136", "search:qmsum/hybrid_t1: 3 rows"]  # still shown on the console
    assert "results/logs/" in (Path(e8.ROOT) / ".gitignore").read_text(encoding="utf-8").splitlines()


def test_judge_stops_before_any_call_if_its_own_log_would_dirty_the_tree(judged_repo: tuple[Path, str, list[Scenario]], tmp_path: Path) -> None:
    from contextlens.experiments.judge_calibrate import UsageGuard
    from tests.test_e6_judge import FakeClaude

    root, commit, _ = judged_repo  # the rows belong to `commit`; the commit made here only changes .gitignore
    (root / ".gitignore").write_text("results/partial/\ndatasets/raw/\n", encoding="utf-8")  # logs are not ignored here
    _commit_all(root)
    haiku = FakeClaude("h")
    with pytest.raises(Refused, match="uncommitted"):
        e8.run_judge(root=root, from_commit=commit, haiku=haiku, sonnet=FakeClaude("s"),
                     guard=UsageGuard(98.0, read=lambda: 10.0, interval_s=0.0), counter_path=tmp_path / "c.json", say=lambda t: None)
    assert haiku.n == 0


def test_a_weekly_stop_in_the_sonnet_sample_still_writes_the_record_and_sonnet_extra_finishes_it(
        judged_repo: tuple[Path, str, list[Scenario]], tmp_path: Path) -> None:
    import threading

    from contextlens.experiments.judge_calibrate import UsageGuard
    from tests.test_e6_judge import FakeClaude

    root, commit, _ = judged_repo
    state = {"pct": 10.0}
    lock = threading.Lock()

    def sonnet_rule(text: str) -> bool:
        with lock:
            if sonnet.n >= 4:
                state["pct"] = 99.0  # the weekly stop is reached while Sonnet is grading
        return True

    haiku, sonnet = FakeClaude("haiku-x"), FakeClaude("sonnet-x", rule=sonnet_rule)
    guard = UsageGuard(98.0, read=lambda: state["pct"], interval_s=0.0)
    path = e8.run_judge(root=root, from_commit=commit, haiku=haiku, sonnet=sonnet, guard=guard,
                        counter_path=tmp_path / "calls.json", say=lambda t: None, concurrency=1)
    rec = json.loads(path.read_text(encoding="utf-8"))
    s = rec["config"]["sonnet"]
    assert haiku.n == 9  # Haiku graded every claim first
    assert 0 < s["claims_graded"] < 9 and s["complete"] is False and s["stopped"]
    assert s["remaining"] == 9 - s["claims_graded"] and "98" in s["stopped"]
    done_before = s["claims_graded"]

    state["pct"] = 10.0  # after the reset
    again = FakeClaude("sonnet-x")
    out = e8.run_sonnet_extra(root=root, from_commit=commit, sonnet=again, more=0,
                              guard=UsageGuard(98.0, read=lambda: state["pct"], interval_s=0.0), counter_path=tmp_path / "calls.json",
                              say=lambda t: None, concurrency=1)
    add = json.loads(out.read_text(encoding="utf-8"))
    assert again.n == 9 - done_before  # only the claims without a grade; none is graded twice
    assert add["finishes_original_sample"] == 9 - done_before and add["sample_complete"] is True
    assert add["at_record"]["claims"] == done_before  # the record's own figure is reported unchanged
    assert json.loads(path.read_text(encoding="utf-8"))["config"]["sonnet"]["claims_graded"] == done_before


# ---------------------------------------------------------------- the primary arm is the shipped one (entries 64 and 70)

def _press_parts(sc: Scenario):
    from contextlens.replay.runner import Context
    from contextlens.memory.store import InMemoryTranscriptStore

    store = InMemoryTranscriptStore()
    for t in sc.conversation.turns:
        store.add(t)
    return sc.request, Context(store=store, as_of_seq=sc.request.as_of_seq)


def test_primary_arm_is_built_from_the_shipped_settings_with_both_e7_fixes_off() -> None:
    from contextlens.agents.consistency_check import PromptedChecker
    from contextlens.agents.graph import TeamAssistant
    from contextlens.llm.client import FakeLLM
    from contextlens.memory.extract import MemoryStore
    from tests.test_e1 import HashEmbedder

    sc = _scenario("p1", "consistent")
    llm = FakeLLM([])
    arm = e8.build_system(e8.PRIMARY, sc, llm, HashEmbedder(), MemoryStore())
    assert type(arm).__name__ == "_TeamChangedPerPress"  # production's own class for CONTEXTLENS_ASSISTANT=team_shipping_changed
    req, ctx = _press_parts(sc)
    team = arm.team(req, ctx)
    assert isinstance(team.tools.checker, PromptedChecker) and team.memory_gap_abstains is True
    assert team.prefetch_memory is False and team.specialist_deadline_s is None  # entries 64 and 70: both fixes off
    assert production.TEAM_SETTINGS == {"prefetch_memory": False, "specialist_deadline_s": None}
    assert team.deadline_s == production.DEADLINE_S
    same = TeamAssistant({"memory": llm, "consistency": llm, "assistance": llm}, team.tools, "v3", production.DEADLINE_S,
                         memory_gap_abstains=True, **production.TEAM_SETTINGS)
    assert team._specs == same._specs  # prompt v3 in every agent's system message


def test_single_arm_is_the_shipped_single_and_the_trained_team_uses_the_same_settings() -> None:
    from contextlens.agents.single import SingleAgent
    from contextlens.llm.client import FakeLLM
    from contextlens.memory.extract import MemoryStore
    from tests.test_e1 import HashEmbedder

    sc = _scenario("p2", "consistent")
    req, ctx = _press_parts(sc)
    single = e8.build_system(e8.SINGLE, sc, FakeLLM([]), HashEmbedder(), MemoryStore())
    assert type(single).__name__ == "_SinglePerPress" and SingleAgent  # production's single, no checker
    seen: dict = {}

    class Spy:
        def __init__(self, *a, **kw):
            seen.update(kw)

    import contextlens.experiments.e8_final as mod

    orig = mod.TeamAssistant
    mod.TeamAssistant = Spy  # type: ignore[misc]
    try:
        e8.build_team_trained(sc, FakeLLM([]), object(), HashEmbedder(), MemoryStore())
    finally:
        mod.TeamAssistant = orig  # type: ignore[misc]
    assert seen["memory_gap_abstains"] is True and seen["prefetch_memory"] is False and seen["specialist_deadline_s"] is None
    assert req and ctx


def test_run_refuses_a_freeze_whose_team_settings_are_not_the_shipped_ones(repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = tmp_path / "a"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"1")
    (adapter / "model.safetensors").write_bytes(b"1")
    freeze = _frozen_repo(repo, adapter, adapter)
    assert freeze["team_settings"] == production.TEAM_SETTINGS and freeze["prompt_version"] == production.PROMPT_VERSION
    e8.check_inputs(repo, freeze)
    monkeypatch.setattr(production, "TEAM_SETTINGS", {"prefetch_memory": True, "specialist_deadline_s": 5.5})
    with pytest.raises(Refused, match="ships now"):
        e8.check_inputs(repo, freeze)


def test_the_base_model_may_be_a_cached_hugging_face_id_and_the_adapter_hash_is_e5s(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlens.experiments import e5_cards

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert not e8._base_present("Qwen/Qwen3.5-2B")
    (tmp_path / "hf/hub/models--Qwen--Qwen3.5-2B").mkdir(parents=True)
    assert e8._base_present("Qwen/Qwen3.5-2B") and not e8._base_present("Qwen/Other")
    adapter = tmp_path / "ad"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"w")
    assert e8.adapter_digest(adapter) == e5_cards.adapter_hash(adapter)  # the same rule as the E5 record's adapter_hash


# ---------------------------------------------------------------- test manifests, the two-system choice, the checker's device (coordinator, 2026-11-07)

FIXTURE_RAW = Path(__file__).parent / "fixtures" / "data"


def test_freeze_adds_qmsum_and_decode_test_manifests_with_ids_and_hashes_only(tmp_path: Path) -> None:
    from contextlens.data.qmsum_scenarios import build_qmsum_dev
    from contextlens.experiments import e4_consistency as e4

    root = tmp_path / "r"
    splits = root / "datasets/splits"
    splits.mkdir(parents=True)
    (splits / "qmsum.json").write_text(json.dumps({"splits": {"dev": ["qmsum-val-0"], "test": ["qmsum-val-0"]}}), encoding="utf-8")
    build_qmsum_dev(FIXTURE_RAW, splits, root)
    e4.prepare_decode_dev(FIXTURE_RAW, root)
    before = {s: json.loads((root / e8.SCENARIO_SOURCES[s][0]).read_text(encoding="utf-8"))["splits"]["dev"] for s in ("qmsum", "decode")}
    made = e8.prepare_test_manifests(root, FIXTURE_RAW)
    assert set(made) == {"qmsum", "decode"} and all(n > 0 for n in made.values())
    for src in ("qmsum", "decode"):
        text = (root / e8.SCENARIO_SOURCES[src][0]).read_text(encoding="utf-8")
        manifest = json.loads(text)
        assert manifest["splits"]["dev"] == before[src]  # dev is untouched
        assert set(manifest["splits"]["test"]) == {"hash", "n", "ids"} and manifest["splits"]["test"]["n"] == made[src]
        assert manifest["splits"]["test"]["hash"] == e8.split_hash(e8.SPLIT, root, src)
        assert "text" not in text and len(text) < 20_000  # ids and hashes, no raw text
        assert (root / e8.SCENARIO_SOURCES[src][1] / "test.jsonl").is_file()  # the text itself sits in the git-ignored folder
    assert all(i.startswith("decode-test-") for i in json.loads(
        (root / e8.SCENARIO_SOURCES["decode"][0]).read_text(encoding="utf-8"))["splits"]["test"]["ids"])


def test_refuses_when_the_test_manifests_are_not_committed_with_the_config(repo: Path, tmp_path: Path) -> None:
    adapter = tmp_path / "a"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"1")
    (adapter / "model.safetensors").write_bytes(b"1")
    freeze = _frozen_repo(repo, adapter, adapter)
    _freeze_file(repo, freeze)
    _commit_all(repo)
    assert e8.committed_freeze(repo)["kind"] == "e8-freeze"  # manifests and config in one commit
    (repo / e8.SCENARIO_SOURCES["qmsum"][0]).write_text('{"splits": {"test": {"hash": "other"}}}', encoding="utf-8")
    with pytest.raises(Refused, match="differs from the committed"):
        e8.committed_freeze(repo)


def test_two_system_choice_drops_the_adapter_and_keeps_the_consistency_measurement(repo: Path, tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "adapter_model.safetensors").write_bytes(b"1")
    (tmp_path / "a" / "model.safetensors").write_bytes(b"1")
    default = _frozen_repo(repo, tmp_path / "a", tmp_path / "a")
    assert e8.systems_of(default) == (e8.PRIMARY, e8.SINGLE, e8.TRAINED_TEAM) and "adapter" in default["trained"]  # the default is three
    checker = tmp_path / "c"
    checker.mkdir()
    (checker / "model.safetensors").write_bytes(b"1")
    two = e8.build_freeze(repo, adapter_base="x", adapter=tmp_path / "none", checker=checker, checker_threshold=0.5, trained_team=False)
    assert e8.systems_of(two) == (e8.PRIMARY, e8.SINGLE) and "adapter" not in two["trained"]
    assert two["detectors"] == {"scenarios": ["prompted", "tfidf"], "decode": ["tfidf"]} == default["detectors"]
    e8.check_inputs(repo, two)  # no adapter is needed, and none is looked for


def test_judge_with_two_systems_writes_a_record_without_the_trained_team(judged_repo_two: tuple[Path, str, list[Scenario]], tmp_path: Path) -> None:
    from contextlens.experiments.judge_calibrate import UsageGuard
    from tests.test_e6_judge import FakeClaude

    root, commit, _ = judged_repo_two
    haiku, sonnet = FakeClaude("haiku-x"), FakeClaude("sonnet-x")
    path = e8.run_judge(root=root, from_commit=commit, haiku=haiku, sonnet=sonnet, guard=UsageGuard(98.0, read=lambda: 10.0, interval_s=0.0),
                        counter_path=tmp_path / "calls.json", say=lambda t: None, concurrency=2)
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert set(rec["predictions"]) == {e8.PRIMARY, e8.SINGLE} and rec["config"]["systems"] == [e8.PRIMARY, e8.SINGLE]
    assert haiku.n == 6  # 3 answered cards on each of 2 systems
    from contextlens.eval.check_run import check

    assert check("E8", root=root, min_n=1, split="test") == []


def test_detector_order_puts_prompted_last_so_the_checker_can_leave_the_gpu_first() -> None:
    plan = e8.detector_plan({"scenarios": ["prompted", "tfidf"], "decode": ["tfidf"]})
    assert plan == [("scenarios", "trained"), ("decode", "trained"), ("scenarios", "tfidf"), ("decode", "tfidf"), ("scenarios", "prompted")]


def test_the_checker_uses_the_gpu_only_if_it_fits_under_5500_mb_and_says_which() -> None:
    assert e8.checker_device(700) == "cuda"  # the desktop alone
    assert e8.checker_device(3500) == "cuda"  # 3500 + 2000 = 5500 fits exactly
    assert e8.checker_device(3501) == "cpu" and e8.checker_device(5300) == "cpu"  # the adapter or the writer is loaded
    assert e8.checker_device(None) == "cpu"  # GPU use cannot be read: do not guess


# ---------------------------------------------------------------- embedding batches and resuming across a search-stage fix (coordinator, 2026-11-07)

class _CapEmbedder:
    """Ollama refuses 256 or more texts in one /api/embed request."""

    model = "cap-embedder"

    def __init__(self) -> None:
        from tests.test_e1 import HashEmbedder

        self.inner, self.sizes = HashEmbedder(), []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.sizes.append(len(texts))
        if len(texts) >= 256:
            raise RuntimeError("HTTP Error 400: Bad Request")
        return self.inner.embed(texts)


def _big_scenario(n: int, sid: str = "big") -> Scenario:
    turns = [Turn(id=f"{sid}:t{i}", conversation_id=sid, seq=i, speaker="A", text=f"word{i} about topic{i % 7}", start_ms=i * 1000,
                  end_ms=i * 1000 + 900, source="dataset") for i in range(n)]
    return Scenario(id=sid, conversation=Conversation(id=sid, turns=turns, source="qmsum"),
                    request=ButtonRequest(conversation_id=sid, as_of_seq=n - 1, question="what about topic3?"), category="qmsum-specific",
                    gold_evidence_turn_ids=[f"{sid}:t3"], gold_label=None, gold_answer=None, origin="dataset")


def test_search_stage_embeds_a_long_meeting_in_batches_of_32() -> None:
    emb = _CapEmbedder()
    rows = e8.search_rows([_big_scenario(300)], emb, {"mode": "hybrid", "rerank": False, "chunk_turns": 1}, lambda t: None)
    assert len(rows) == 1 and rows[0].ranked_turn_ids
    assert max(emb.sizes) <= 32 and sum(emb.sizes) >= 300  # every turn was embedded, none in a request of 256 or more


def test_preflight_embeds_every_case_in_batches_and_names_the_case_that_fails() -> None:
    emb = _CapEmbedder()
    e8.preflight_embeddings([_big_scenario(300, "a"), _big_scenario(40, "b")], emb)
    assert max(emb.sizes) <= 32

    class Broken:
        model = "x"

        def embed(self, texts):  # noqa: ANN001
            raise RuntimeError("HTTP Error 400: Bad Request")

    with pytest.raises(e2.RunBroken, match="case big"):
        e8.preflight_embeddings([_big_scenario(5)], Broken())


def _row_line(commit: str, part: str = "cards", sid: str = "k0") -> str:
    return json.dumps({"commit": commit, "subset": part, "winner": e8.SINGLE, "model": "E8", "calls": 0,
                       "row": json.loads(_abstain(sid).model_dump_json())})


def _src_commit(root: Path, rel: str, msg: str) -> str:
    f = root / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(msg, encoding="utf-8")
    _commit_all(root, msg)
    return e8.head_commit(root)


def test_card_rows_are_reused_only_if_the_code_that_shaped_them_is_unchanged(repo: Path) -> None:
    first = _src_commit(repo, "src/contextlens/agents/x.py", "agents v1")
    e2._append_durable(repo / e8.ROWS_REL, _row_line(first))
    assert e8.reusable_cards_commit(repo, first) == first  # nothing older to reuse: this commit
    docs = _src_commit(repo, "docs/note.md", "a note")  # the search-stage fix changes e8_final.py, which is not card code
    assert e8.reusable_cards_commit(repo, docs) == first
    e8.require_first_test_run(repo, docs, None, reuse={first})  # these rows are a resume, not a second run
    with pytest.raises(Refused, match="already run"):
        e8.require_first_test_run(repo, docs, None)
    for rel in ("src/contextlens/agents/x.py", "src/contextlens/prompts/v3/recall.md", "src/contextlens/api/production.py",
                "configs/e8_freeze.json"):
        changed = _src_commit(repo, rel, f"changed {rel}")
        with pytest.raises(Refused, match="card code changed"):
            e8.reusable_cards_commit(repo, changed)
        assert changed != first


def test_card_rows_at_two_commits_are_ambiguous_and_refused(repo: Path) -> None:
    a = _src_commit(repo, "docs/a.md", "a")
    b = _src_commit(repo, "docs/b.md", "b")
    e2._append_durable(repo / e8.ROWS_REL, _row_line(a))
    e2._append_durable(repo / e8.ROWS_REL, _row_line(b, sid="k1"))
    with pytest.raises(Refused, match="several commits"):
        e8.reusable_cards_commit(repo, b)


def test_judge_uses_cards_from_the_earlier_commit_and_the_record_says_so(tmp_path: Path) -> None:
    from contextlens.experiments.judge_calibrate import UsageGuard
    from tests.test_e6_judge import FakeClaude

    root, search_commit, _ = _make_judged(tmp_path, trained_team=False, search_later=True)
    cards_commit = e8.git(root, "rev-parse", "HEAD~1").stdout.strip()
    assert search_commit != cards_commit
    path = e8.run_judge(root=root, from_commit=search_commit, haiku=FakeClaude("h"), sonnet=FakeClaude("s"),
                        guard=UsageGuard(98.0, read=lambda: 10.0, interval_s=0.0), counter_path=tmp_path / "calls.json",
                        say=lambda t: None, concurrency=2)
    cfg = json.loads(path.read_text(encoding="utf-8"))["config"]
    assert cfg["generating_commit"] == cards_commit and cfg["search_and_detector_commit"] == search_commit
    assert "reran at" in cfg["resume_note"] and cards_commit[:12] in cfg["resume_note"] and search_commit[:12] in cfg["resume_note"]


# ---------------------------------------------------------------- the written verdict and the restatement (verifier, 2026-11-07)

def _cmp(a: str, b: str, metric: str, diff: float, ci: list[float], p: float | None) -> dict:
    return {"a": a, "b": b, "metric": metric, "diff": diff, "ci": ci, "mcnemar_p": p}


def test_primary_vs_single_reads_comparisons_ordered_either_way_with_the_sign_turned_to_primary_minus_single() -> None:
    P, S = e8.PRIMARY, e8.SINGLE
    as_saved = [_cmp(S, P, "grounded_rate", -0.10, [-0.20, -0.05], None),               # single minus primary, as every E8 comparison is
                _cmp(S, P, "abstain_when_insufficient", -0.0625, [-0.211, 0.0], 1.0),   # the real record's shape: interval touches zero
                _cmp(S, P, "unsupported_claim_rate", 0.10, [0.04, 0.2], None),          # lower is better: single worse, so primary better
                _cmp(S, P, "answer_when_answerable", -0.30, [-0.40, -0.20], 0.40)]      # interval clear but McNemar disagrees
    got = e8.primary_vs_single(as_saved)
    assert got == {"grounded_rate": "better", "abstain_when_insufficient": "no clear difference",
                   "unsupported_claim_rate": "better", "answer_when_answerable": "no clear difference"}
    assert e8.primary_vs_single([_cmp(P, S, "grounded_rate", 0.10, [0.05, 0.20], None)]) == {"grounded_rate": "better"}  # saved the other way
    detail = e8.primary_vs_single_detail(as_saved)["grounded_rate"]
    assert detail["diff"] == pytest.approx(0.10) and detail["ci"] == pytest.approx([0.05, 0.20]) and detail["verdict"] == "better"
    assert detail["difference"] == "team_shipping_changed minus single"


def test_a_judge_record_carries_a_non_empty_primary_vs_single(judged_repo: tuple[Path, str, list[Scenario]], tmp_path: Path) -> None:
    rec = _judge_fixture(judged_repo, tmp_path)
    cfg = json.loads(rec.read_text(encoding="utf-8"))["config"]
    assert cfg["primary_vs_single"] and set(cfg["primary_vs_single"]) <= {"grounded_rate", "unsupported_claim_rate",
                                                                          "abstain_when_insufficient", "answer_when_answerable",
                                                                          "tokens_per_card", "latency_p50_ms", "latency_p95_ms"}


def _judge_fixture(fix: tuple[Path, str, list[Scenario]], tmp_path: Path) -> Path:
    from contextlens.experiments.judge_calibrate import UsageGuard
    from tests.test_e6_judge import FakeClaude

    root, commit, _ = fix
    return e8.run_judge(root=root, from_commit=commit, haiku=FakeClaude("h"), sonnet=FakeClaude("s"),
                        guard=UsageGuard(98.0, read=lambda: 10.0, interval_s=0.0), counter_path=tmp_path / "calls.json",
                        say=lambda t: None, concurrency=2)


def test_restate_rewrites_the_verdict_from_saved_rows_with_no_new_call_and_supersedes_the_old_record(
        judged_repo: tuple[Path, str, list[Scenario]], tmp_path: Path) -> None:
    from contextlens.eval.check_run import check

    root = judged_repo[0]
    old_path = _judge_fixture(judged_repo, tmp_path)
    old = json.loads(old_path.read_text(encoding="utf-8"))
    old["config"]["primary_vs_single"] = {}  # the record as it was saved, with the wrong orientation
    old_path.write_text(json.dumps(old, indent=1) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="override note"):
        e8.restate(old["run_id"], "orientation", "", root=root)
    with pytest.raises(ValueError, match="reason"):
        e8.restate(old["run_id"], " ", "same single run", root=root)
    new_path = e8.restate(old["run_id"], "primary_vs_single was empty: wrong orientation", "a restatement of the same single test run, not a second run", root=root)
    new = json.loads(new_path.read_text(encoding="utf-8"))
    assert new["run_id"] != old["run_id"] and new["supersedes"] == [{"run_id": old["run_id"], "reason": "primary_vs_single was empty: wrong orientation"}]
    assert new["config"]["override_note"].startswith("a restatement") and new["config"]["restated_from"] == old["run_id"]
    assert new["config"]["primary_vs_single"] and new["config"]["primary_vs_single_detail"]
    assert new["claude_calls"] == 0 and new["config"]["claude_calls_in_original"] == old["claude_calls"]  # nothing was asked again
    for system, rel in old["predictions"].items():  # the same saved, already-judged rows
        assert (root / new["predictions"][system]).read_bytes() == (root / rel).read_bytes()
    assert new["metrics"] == old["metrics"] and new["subset"] == old["subset"] and new["dataset_hash"] == old["dataset_hash"]
    assert check("E8", root=root, min_n=1, split="test") == []
