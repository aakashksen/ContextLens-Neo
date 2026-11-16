"""team_routed and single_plus: routing, the checker in code, retrieval before the first call, merge, guards."""

from contextlens.agents.precheck import prepare
from contextlens.agents.routed import RoutedTeam, merge
from contextlens.agents.single import SingleAgent
from contextlens.agents.tools import Tools
from contextlens.api import production
from contextlens.llm.client import FakeLLM
from contextlens.memory.extract import MemoryStore
from contextlens.memory.store import InMemoryTranscriptStore
from contextlens.replay.runner import Context
from contextlens.retrieval.hybrid import Retriever
from contextlens.schemas import ButtonRequest, EvidenceRef, Finding
from tests.test_team import NoEmbed, done, finding, turn

LONG = [turn(1, "The launch budget is five thousand dollars", "Ravi")] + [
    turn(i, f"Chatter about lunch number {i}", "Sue") for i in range(2, 10)
] + [turn(10, "The launch budget is now eight thousand dollars", "Ravi")]
QUESTION = "What is the launch budget?"


class FakeChecker:
    def __init__(self, label="consistent", boom=False):
        self.label, self.boom, self.calls = label, boom, []

    def check(self, statement, earlier):
        self.calls.append((statement.id, [t.id for t in earlier]))
        if self.boom:
            raise RuntimeError("down")
        return self.label, 0.97


def build(responses, checker, turns=LONG, as_of=None):
    store = InMemoryTranscriptStore()
    for t in turns:
        store.add(t)
    as_of = as_of or max(t.seq for t in turns)
    tools = Tools(store, Retriever(store, NoEmbed()), MemoryStore(), as_of, "c1", checker)
    llm = FakeLLM(list(responses))
    return RoutedTeam(llm, tools, "v3"), llm, Context(store=store, as_of_seq=as_of), tools


def req(question=None, as_of=10):
    return ButtonRequest(conversation_id="c1", as_of_seq=as_of, question=question)


OLD = finding("x", "t1", "five thousand dollars", "budget was five thousand")
NEW = finding("x", "t10", "now eight thousand dollars", "budget is now eight thousand")


def test_a_question_runs_only_the_memory_specialist():
    team, llm, ctx, _ = build([done(NEW)], FakeChecker("consistent"))
    card = team.answer(req(QUESTION), ctx)
    assert len(llm.calls) == 1 and card.status == "answered"
    assert ctx.stage_ms["ran_memory"] == 1 and "ran_consistency" not in ctx.stage_ms and "ran_assistance" not in ctx.stage_ms
    assert ctx.stage_ms["checker_consistent"] == 1 and "checker_ms" in ctx.stage_ms
    assert [f.agent for f in card.findings] == ["memory"]


def test_no_question_runs_the_assistance_specialist_and_no_decide_call():
    team, llm, ctx, _ = build([done(NEW)], FakeChecker("consistent"))
    card = team.answer(req(), ctx)
    assert len(llm.calls) == 1 and "assistance agent" in llm.calls[0][0]["content"]
    assert ctx.stage_ms["ran_assistance"] == 1 and "ran_memory" not in ctx.stage_ms
    assert card.findings[0].agent == "assistance"


def test_a_flagged_conflict_adds_the_consistency_specialist_and_its_finding_carries_the_checker_label():
    for label in ("contradiction", "update"):
        team, llm, ctx, _ = build([done(NEW), done(OLD, NEW)], FakeChecker(label))
        card = team.answer(req(QUESTION), ctx)
        assert len(llm.calls) == 2 and ctx.stage_ms["ran_consistency"] == 1
        assert f"{label} (confidence" in llm.calls[1][1]["content"] and "at least two evidence" in llm.calls[1][1]["content"]
        cons = [f for f in card.findings if f.kind == "consistency"]
        assert cons and all(f.label == label for f in cons)


def test_the_checker_label_overrides_whatever_the_model_wrote():
    wrong = {**done(OLD, NEW)["findings"][0], "label": "consistent"}
    team, _, ctx, _ = build([done(NEW), {"tool": "final", "findings": [wrong]}], FakeChecker("contradiction"))
    card = team.answer(req(QUESTION), ctx)
    assert [f.label for f in card.findings if f.kind == "consistency"] == ["contradiction"]


def test_findings_with_the_same_cited_turns_are_merged_and_the_labelled_one_stays():
    team, _, ctx, _ = build([done(NEW), done(NEW)], FakeChecker("update"))
    card = team.answer(req(QUESTION), ctx)
    assert len(card.findings) == 1 and card.findings[0].label == "update"


def test_merge_by_statement_and_keeps_order():
    def f(tid, text, label=None):
        return Finding(agent="a", kind="recall", statement=text, confidence=1, label=label,
                       evidence=[EvidenceRef(turn_id=tid, quote="q", speaker="s", start_ms=0)])

    out = merge([f("t1", "Same  thing"), f("t2", "same thing"), f("t3", "other")])
    assert [x.evidence[0].turn_id for x in out] == ["t1", "t3"]


def test_retrieval_runs_before_the_first_call_and_brings_in_turns_outside_the_window():
    team, llm, ctx, _ = build([done(NEW)], FakeChecker("consistent"))
    team.answer(req(QUESTION), ctx)
    first = llm.calls[0][1]["content"]
    assert "searching the whole conversation" in first and "five thousand dollars" in first  # t1 is not in the last 6


def test_the_checker_sees_the_latest_turn_and_the_search_hits():
    chk = FakeChecker("consistent")
    team, _, ctx, _ = build([done(NEW)], chk)
    team.answer(req(QUESTION), ctx)
    statement, earlier = chk.calls[0]
    assert statement == "t10" and "t1" in earlier and "t10" not in earlier


def test_the_evidence_guard_still_drops_a_finding_that_cites_a_turn_after_the_press():
    later = turn(11, "The launch budget is twelve thousand dollars", "Ravi")
    ghost = finding("x", "t11", "twelve thousand dollars", "budget is twelve thousand")
    team, _, ctx, _ = build([done(ghost)], FakeChecker("consistent"), turns=[*LONG, later], as_of=10)
    card = team.answer(req(QUESTION), ctx)
    assert card.status == "abstained" and not card.findings


def test_a_fabricated_quote_is_dropped():
    bad = finding("x", "t10", "ninety thousand dollars", "budget is ninety thousand")
    team, _, ctx, _ = build([done(bad)], FakeChecker("consistent"))
    assert team.answer(req(QUESTION), ctx).status == "abstained"


def test_memory_finding_nothing_abstains():
    team, _, ctx, _ = build([done()], FakeChecker("consistent"))
    assert team.answer(req(QUESTION), ctx).status == "abstained"


def test_a_failing_checker_falls_back_safely():
    team, llm, ctx, _ = build([done(NEW)], FakeChecker(boom=True))
    card = team.answer(req(QUESTION), ctx)
    assert len(llm.calls) == 1 and card.status == "answered"
    assert any("checker failed" in m for m in card.missing) and ctx.stage_ms["checker_failed"] == 1
    assert "Consistency check" not in llm.calls[0][1]["content"]


def test_without_a_checker_the_team_still_answers():
    team, llm, ctx, _ = build([done(NEW)], None)
    assert team.answer(req(QUESTION), ctx).status == "answered" and len(llm.calls) == 1


def test_single_plus_shows_the_model_the_verdict_and_the_hits_before_its_first_call():
    _, _, ctx, tools = build([], FakeChecker("contradiction"))
    card_json = {"mode": "recall", "status": "answered", "missing": [],
                 "findings": [{**OLD, "agent": "single"}, {**NEW, "agent": "single"}]}
    llm = FakeLLM([{"tool": "final", "card": card_json}])
    agent = SingleAgent(llm, Tools(tools.store, tools.retriever, tools.memory, 10, "c1"), "v3",
                        prepare=lambda r, t, e: prepare(tools, r, t, e))
    card = agent.answer(req(QUESTION), ctx)
    user = llm.calls[0][1]["content"]
    assert "contradiction (confidence" in user and "five thousand dollars" in user and len(llm.calls) == 1
    assert ctx.stage_ms["checker_contradiction"] == 1 and len(card.findings) == 2
    assert "check_consistency" not in llm.calls[0][0]["content"]


def test_both_names_are_registered_and_built_by_the_shared_factory():
    assert {"team_routed", "single_plus"} <= set(production.ASSISTANTS) and production.DEFAULT_ASSISTANT == "team_rag"
    store = InMemoryTranscriptStore()
    r = Retriever(store, NoEmbed())
    llm = FakeLLM([])
    llms = {"memory": llm, "consistency": llm, "assistance": llm}
    for name, cls in (("team_routed", production._RoutedPerPress), ("single_plus", production._SinglePlusPerPress)):
        assert isinstance(production.build_assistant(name, store, r, MemoryStore(), llms), cls)


def test_a_flagged_conflict_runs_the_consistency_specialist_first_and_alone_without_a_question():
    team, llm, ctx, _ = build([done(OLD, NEW)], FakeChecker("update"))
    card = team.answer(req(), ctx)
    assert len(llm.calls) == 1 and ctx.stage_ms["ran_consistency"] == 1 and "ran_assistance" not in ctx.stage_ms
    assert card.findings and all(f.label == "update" for f in card.findings)


def test_a_flagged_conflict_with_a_question_runs_consistency_before_memory():
    team, llm, ctx, _ = build([done(OLD, NEW), done(NEW)], FakeChecker("contradiction"))
    team.answer(req(QUESTION), ctx)
    assert "consistency" in llm.calls[0][0]["content"].lower() and ctx.stage_ms["ran_memory"] == 1


def test_code_labels_a_finding_that_cites_both_sides_when_no_finding_carries_the_label():
    from contextlens.agents.routed import label_conflict

    both = Finding.model_validate({**OLD, "agent": "memory", "evidence": OLD["evidence"] + NEW["evidence"]})
    one = Finding.model_validate({**NEW, "agent": "memory"})
    out = label_conflict([both, one], "contradiction", "t10")
    assert [f.label for f in out] == ["contradiction", None]
