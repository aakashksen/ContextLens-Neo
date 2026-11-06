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
    (adapter / "w.bin").write_bytes(b"adapter-v1")
    (checker / "model.safetensors").write_bytes(b"checker-v1")
    freeze = _frozen_repo(repo, adapter, checker)
    assert freeze["systems"] == {"primary": "team_shipping_changed", "others": ["single", "team_trained"]}
    assert freeze["assistant_setting"]["value"] in production.ASSISTANTS  # the build the product can make (entry 57)
    assert freeze["prompt_version"] == "v3"
    assert freeze["test_hashes"] == {"scenarios": "h-scenarios", "qmsum": "h-qmsum", "decode": "h-decode"}
    e8.check_inputs(repo, freeze)
    (adapter / "w.bin").write_bytes(b"adapter-v2")
    with pytest.raises(Refused, match="not the one that was frozen"):
        e8.check_inputs(repo, freeze)
    (adapter / "w.bin").write_bytes(b"adapter-v1")
    (checker / "model.safetensors").unlink()
    (checker / "other.bin").write_bytes(b"x")
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
    (adapter / "w").write_bytes(b"1")
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
    (models / "w.bin").write_bytes(b"x")
    freeze = e8.build_freeze(root, adapter_base=models, adapter=models, checker=models, checker_threshold=0.5)
    e8.write_freeze(root, freeze)
    (root / ".gitignore").write_text("results/partial/\nresults/logs/\ndatasets/raw/\n", encoding="utf-8")
    _git(root, "init", "-q")
    _commit_all(root)
    commit = e8.head_commit(root)
    for n in e8.SYSTEMS:
        for sc in cards:
            row = _abstain(sc.id) if sc.gold_label == "insufficient" else _answer(sc.id, f"{n} says w0")
            e2._append_durable(root / e8.ROWS_REL, json.dumps({"commit": commit, "subset": "cards", "winner": n, "model": "E8", "calls": 0,
                                                              "row": json.loads(row.model_dump_json())}))
    parts = {("search:qmsum", e8.SEARCH_SYSTEM): qm, (f"search:{e8.LME}", e8.SEARCH_SYSTEM): lme}
    for (part, system), scs in parts.items():
        for sc in scs:
            row = Prediction(scenario_id=sc.id, ranked_turn_ids=[f"{sc.id}-t0"], stage_latency_ms={"total": 3})
            e2._append_durable(root / e8.ROWS_REL, json.dumps({"commit": commit, "subset": part, "winner": system, "model": "E8", "calls": 0,
                                                              "row": json.loads(row.model_dump_json())}))
    for part, scs in (("detector:scenarios", cards), ("detector:decode", dec)):
        for system in ("trained", freeze["detectors"][part.split(":")[1]]):
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
    assert cfg["benchmarks"]["decode"]["false_alert_comparison"]["mcnemar_p"] is not None
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
    (adapter / "w").write_bytes(b"1")
    freeze = _frozen_repo(repo, adapter, adapter)
    sub = freeze["cards_subset"]
    assert sub["name"] == "e8-cards" and sub["path"] == "datasets/splits/subsets/e8-cards.json"
    assert sub["sha256"] == e8.sha256_path(repo / sub["path"]) and "`conversation.source`" in sub["reads"]
    assert "same commit" in sub["committed"]


def test_refuses_when_the_manifest_is_not_committed_with_the_config(repo: Path, tmp_path: Path) -> None:
    adapter = tmp_path / "a"
    adapter.mkdir()
    (adapter / "w").write_bytes(b"1")
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
    assert production.DEFAULT_ASSISTANT == "team"


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
