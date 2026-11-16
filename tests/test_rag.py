"""team_rag: text cleaning and quote spans, code-built citations, score gate, writer failures, conflict labels."""

import pytest

from contextlens.agents import rag, rag_text
from contextlens.agents.rag import RagTeam, cited, statement_of
from contextlens.agents.tools import Tools
from contextlens.guard.citations import _norm
from contextlens.llm.client import FakeLLM
from contextlens.memory.extract import MemoryStore
from contextlens.memory.store import InMemoryTranscriptStore
from contextlens.replay.runner import Context
from contextlens.retrieval.hybrid import Retriever
from contextlens.schemas import ButtonRequest
from tests.test_routed import FakeChecker
from tests.test_team import NoEmbed, turn

RAW = ("Well , from experience , um I've had remote controls in the past that have had very {disfmarker} they've had lots and "
       "lots of buttons . {vocalsound} And they were very small . Hard to press .")


def test_clean_drops_markers_and_closes_punctuation():
    out = rag_text.clean(RAW)
    assert "{" not in out and " ," not in out and " ." not in out
    assert out.startswith("Well, from experience, um I've had")


def test_quote_span_is_a_real_substring_and_picks_the_matching_sentence():
    q = rag_text.quote_span(RAW, ["how small and hard to press"])
    assert _norm(q) in _norm(RAW) and "{" not in q
    assert "small" in q


def test_quote_span_never_crosses_a_marker_and_is_capped():
    q = rag_text.quote_span(RAW, ["remote controls"], max_chars=40)
    assert _norm(q) in _norm(RAW) and len(q) <= 40


def test_clip_for_prompt_keeps_the_best_sentence_of_a_long_turn():
    text = " ".join(f"Filler sentence number {i} is here." for i in range(40)) + " The budget is nine hundred pounds. " + "Tail. " * 30
    out = rag_text.clip_for_prompt(text, ["budget pounds"], 200)
    assert "nine hundred pounds" in out and len(out) < 260


def test_cited_keeps_valid_numbers_in_order_once():
    assert cited("A [2]. B [1, 2]. C [9].", {1, 2, 3}) == [2, 1]
    assert statement_of("The budget grew [1] a lot [2] .") == "The budget grew a lot."


TURNS = (
    [turn(1, "The launch budget is five thousand dollars , {disfmarker} I think .", "Ravi")]
    + [turn(i, f"Chatter about lunch number {i}", "Sue") for i in range(2, 9)]
    + [turn(9, "The launch budget is now eight thousand dollars", "Ravi")]
    + [turn(10, "Okay thanks", "Sue")]
)


class Scorer:
    """A scorer that likes turns that share words with the query; scores 0..1."""

    def __init__(self, fail=False):
        self.fail, self.seen = fail, []

    def score(self, query, turns):
        if self.fail:
            raise RuntimeError("no model")
        self.seen.append(query)
        want = set(rag_text.words(query))
        return [len(want & set(rag_text.words(t.text))) / max(len(want), 1) for t in turns]


def build(responses, checker=None, scorer=None, turns=TURNS):
    store = InMemoryTranscriptStore()
    for t in turns:
        store.add(t)
    as_of = max(t.seq for t in turns)
    tools = Tools(store, Retriever(store, NoEmbed()), MemoryStore(), as_of, "c1", checker or FakeChecker("consistent"))
    llm = FakeLLM(list(responses))
    return RagTeam(llm, tools, scorer or Scorer()), llm, Context(store=store, as_of_seq=as_of)


def req(question=None, as_of=10):
    return ButtonRequest(conversation_id="c1", as_of_seq=as_of, question=question)


Q = "What is the launch budget?"


def test_the_writer_cites_a_number_and_code_fills_the_quote_from_the_real_turn(monkeypatch):
    team, llm, ctx = build(["The budget is now eight thousand dollars [1]."])
    card = team.answer(req("What is the launch budget now?"), ctx)
    assert card.status == "answered" and len(llm.calls) == 1
    f = card.findings[0]
    assert f.statement == "The budget is now eight thousand dollars." and "t9" in {e.turn_id for e in f.evidence}
    assert all(_norm(e.quote) in _norm(next(t.text for t in TURNS if t.id == e.turn_id)) for e in f.evidence)
    assert ctx.stage_ms["rag_candidates"] > 0 and "rag_top_milli" in ctx.stage_ms


def test_each_best_turn_is_shown_with_its_neighbours_once():
    team, llm, ctx = build(["Five thousand dollars [1]."])
    team.answer(req("What is the launch budget now?"), ctx)
    user = llm.calls[0][1]["content"]
    assert user.count("Ravi:") == 2 and "Chatter about lunch number 8" in user and "Chatter about lunch number 2" in user


def test_evidence_comes_first_in_the_prompt_and_markers_are_cleaned():
    team, llm, ctx = build(["Five thousand dollars [1]."])
    team.answer(req(Q), ctx)
    user = llm.calls[0][1]["content"]
    assert user.index("<evidence>") < user.index("<recent>") < user.index("Question:")
    assert "{disfmarker}" not in user and "[1]" in user


def test_a_model_that_cites_nothing_still_gets_a_real_quote_from_the_best_turn():
    team, _, ctx = build(["The budget was discussed."])
    card = team.answer(req(Q), ctx)
    assert card.status == "answered" and card.findings[0].evidence[0].turn_id in {"t1", "t9"}
    assert ctx.stage_ms["rag_uncited"] == 1


def test_an_empty_or_failed_writer_gives_an_extractive_answer_not_an_abstention():
    for responses in ([""], []):
        team, _, ctx = build(responses)
        card = team.answer(req(Q), ctx)
        assert card.status in ("answered", "partial") and card.findings
        assert ctx.stage_ms["rag_extractive"] == 1
    assert any("deadline" in n or "llm error" in n for n in card.missing)


def test_a_number_the_evidence_does_not_hold_is_replaced_by_the_quote():
    team, _, ctx = build(["The budget is 99 thousand [1]."])
    card = team.answer(req(Q), ctx)
    assert card.status == "answered" and "99" not in card.findings[0].statement


def test_nothing_relevant_abstains_in_code_with_the_closest_turn(monkeypatch):
    monkeypatch.setattr(rag, "GATE_ASKED", 0.5)
    team, llm, ctx = build([])
    card = team.answer(req("Where is the picnic?"), ctx)
    assert card.status == "abstained" and card.missing[0] == "Not discussed yet" and not llm.calls
    assert ctx.stage_ms["rag_gate_abstain"] == 1


def test_writer_refusal_abstains_only_below_the_confident_score(monkeypatch):
    monkeypatch.setattr(rag, "CONFIDENT", 0.9)
    team, _, ctx = build([rag.NOT_ANSWERED], scorer=Scorer())
    assert team.answer(req("launch budget picnic zebra"), ctx).status == "abstained"
    team, _, ctx = build([rag.NOT_ANSWERED], scorer=Scorer())
    assert team.answer(req("launch budget"), ctx).status == "answered"  # confident reranker: the refusal is ignored


def test_a_broken_scorer_keeps_the_hybrid_order_and_does_not_gate(monkeypatch):
    monkeypatch.setattr(rag, "GATE_ASKED", 0.99)
    team, _, ctx = build(["Eight thousand [1]."], scorer=Scorer(fail=True))
    card = team.answer(req(Q), ctx)
    assert card.status == "answered" and any("reranker failed" in n for n in card.missing)


@pytest.mark.parametrize("label", ["contradiction", "update"])
def test_a_flagged_conflict_cites_the_latest_and_an_earlier_turn_with_the_checkers_label(label):
    team, llm, ctx = build(["The latest turn changes the budget [1]."], checker=FakeChecker(label))
    card = team.answer(req(None, as_of=9), ctx)
    f = card.findings[0]
    ids = {e.turn_id for e in f.evidence}
    assert f.label == label and "t9" in ids and len(ids) >= 2
    assert label in llm.calls[0][1]["content"]


def test_consistent_verdict_leaves_no_label():
    team, _, ctx = build(["Same figure [1]."], checker=FakeChecker("consistent"))
    assert team.answer(req(None, as_of=9), ctx).findings[0].label is None


def test_registered_without_changing_the_default():
    from contextlens.api import production

    assert "team_rag" in production.ASSISTANTS and production.DEFAULT_ASSISTANT == "team_routed"


def _notes_team(use_memory):
    from contextlens.schemas import MemoryItem

    team, llm, ctx = build(["Ravi gave the budget [1]."])
    team.use_memory = use_memory
    team.tools.memory.add([MemoryItem(id="m1", conversation_id="c1", kind="fact", subject="launch budget",
                                      statement="The launch budget was set by Ravi", speaker="Ravi", source_turn_ids=["t5"], as_of_seq=9)])
    return team, llm, ctx


def test_topic_questions_add_memory_notes_as_evidence_pointing_to_their_source_turn():
    team, llm, ctx = _notes_team(True)
    card = team.answer(req("What did the meeting discuss about the launch budget?"), ctx)
    user = llm.calls[0][1]["content"]
    assert "earlier summary note: The launch budget was set by Ravi" in user and ctx.stage_ms["rag_notes"] == 1
    assert card.status == "answered"


def test_without_the_flag_or_for_a_non_topic_question_notes_are_not_used():
    team, llm, ctx = _notes_team(False)
    team.answer(req("What did the meeting discuss about the launch budget?"), ctx)
    assert "summary note" not in llm.calls[0][1]["content"]
    team, llm, ctx = _notes_team(True)
    team.answer(req("What is the launch budget?"), ctx)
    assert "summary note" not in llm.calls[0][1]["content"]


def test_both_arms_are_registered():
    from contextlens.api import production

    assert {"team_rag", "team_rag_nomem"} <= set(production.PER_PRESS)
    assert production.PER_PRESS["team_rag"].use_memory and not production.PER_PRESS["team_rag_nomem"].use_memory


def test_verdict_line_is_read_and_dropped_from_the_statement():
    assert rag.refused("VERDICT: NOT_IN_EVIDENCE") and not rag.refused("VERDICT: ANSWERED\nFive thousand [1].")
    assert statement_of("VERDICT: ANSWERED\nThe budget is five thousand dollars [1].") == "The budget is five thousand dollars."
    team, _, ctx = build(["VERDICT: NOT_IN_EVIDENCE"])
    assert team.answer(req(Q), ctx).status == "abstained"


class Picky(Scorer):
    """Scores only the lunch chatter, so the earlier budget turn is never among the best turns."""

    def score(self, query, turns):
        return [0.9 if "lunch" in t.text else 0.0 for t in turns]


def test_a_finder_replaces_the_checker_and_its_earlier_turn_is_shown_and_cited():
    earlier = TURNS[0]
    team, llm, ctx = build(["The budget changed [1]."], checker=FakeChecker("consistent"), scorer=Picky())
    team.finder = lambda statement, window: ("contradiction", 0.93, [earlier.id])
    f = team.answer(req(None, as_of=9), ctx).findings[0]
    ids = {e.turn_id for e in f.evidence}
    assert f.label == "contradiction" and {earlier.id, TURNS[-2].id} <= ids
    assert earlier.text.split(",")[0].strip().lower() in llm.calls[0][1]["content"].lower()
    assert ctx.stage_ms.get("checker_contradiction") == 1


def test_a_failing_finder_flags_nothing():
    def broken(statement, window):
        raise RuntimeError("model gone")

    team, _, ctx = build(["Same figure [1]."], checker=FakeChecker("contradiction"))
    team.finder = broken
    card = team.answer(req(None, as_of=9), ctx)
    assert card.findings[0].label is None and any("checker failed" in m for m in card.missing)


def test_an_asked_question_does_not_carry_a_conflict_unrelated_to_it():
    team, llm, ctx = build(["Five thousand dollars [1]."], checker=FakeChecker("consistent"), scorer=Scorer())
    team.finder = lambda statement, window: ("contradiction", 0.95, [TURNS[3].id])  # lunch chatter vs "Okay thanks"
    card = team.answer(req(Q), ctx)
    assert all(f.label is None for f in card.findings) and ctx.stage_ms.get("rag_conflict_off_topic") == 1
    assert TURNS[3].id not in {e.turn_id for f in card.findings for e in f.evidence}


def test_an_asked_question_keeps_a_conflict_about_it():
    team, _, ctx = build(["The budget is now eight thousand dollars [1]."], scorer=Scorer())
    team.finder = lambda statement, window: ("update", 0.95, [TURNS[0].id])  # the budget turns are the evidence
    card = team.answer(req("What is the launch budget now?", as_of=9), ctx)
    assert card.findings[0].label == "update" and "rag_conflict_off_topic" not in ctx.stage_ms
