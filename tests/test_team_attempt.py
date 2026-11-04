"""Entry 40: the team's merge can carry memory's "found nothing" to the card. Off, the team is what E3 ran."""

from contextlens.llm.client import FakeLLM
from tests.test_team import CON, MEM, PLAN, build, done, finding, llms, req

EMPTY = done()
GAP = done(finding("consistency", "t2", "moved the deadline to Wednesday", "The transcript does not say who set it"))
INSUFFICIENT = done({**finding("consistency", "t2", "moved the deadline to Wednesday", "Not enough to say"), "label": "insufficient"})


def run(ls, **kw):
    team, ctx, _ = build(ls, **kw)
    card = team.answer(req(), ctx)
    out = card.model_dump()
    out.pop("latency_ms"), out.pop("tokens")
    out["findings"] = sorted(out["findings"], key=lambda f: f["agent"])
    return card, out, ctx


CON_FINDING = {"agent": "consistency", "confidence": 0.9, "kind": "consistency", "label": None, "statement": "Sue moved it",
               "evidence": [{"quote": "moved the deadline to Wednesday", "speaker": "Sue", "start_ms": 2000, "turn_id": "t2"}]}
MEM_FINDING = {"agent": "memory", "confidence": 0.9, "kind": "recall", "label": None, "statement": "Ravi said Monday",
               "evidence": [{"quote": "The deadline is Monday", "speaker": "Ravi", "start_ms": 1000, "turn_id": "t1"}]}
# What the team returned for these scripts before the change existed (taken from the code at the commit before it).
BEFORE = {
    "memory_empty": {"cost_usd": 0.0, "findings": [CON_FINDING], "missing": [], "mode": "recall", "status": "answered", "suggestion": None},
    "consistency_empty": {"cost_usd": 0.0, "findings": [MEM_FINDING], "missing": [], "mode": "recall", "status": "answered", "suggestion": None},
    "both": {"cost_usd": 0.0, "findings": [CON_FINDING, MEM_FINDING], "missing": [], "mode": "recall", "status": "answered", "suggestion": None},
    "all_empty": {"cost_usd": 0.0, "findings": [], "missing": ["no verified findings"], "mode": "recall", "status": "abstained", "suggestion": None},
}
SCRIPTS = {
    "memory_empty": dict(memory=[EMPTY], consistency=[CON], assistance=[PLAN]),
    "consistency_empty": dict(memory=[MEM], consistency=[EMPTY], assistance=[PLAN]),
    "both": dict(memory=[MEM], consistency=[CON], assistance=[PLAN]),
    "all_empty": dict(memory=[EMPTY], consistency=[EMPTY], assistance=[]),
}


def test_an_unchanged_team_gives_exactly_the_cards_it_gave_before_the_change():
    for name, script in SCRIPTS.items():
        _, default, ctx = run(llms(**script))
        _, off, _ = run(llms(**script), memory_gap_abstains=False)
        assert default == off == BEFORE[name], name
        assert ctx.drops == []


def test_the_changed_team_abstains_when_memory_found_nothing_and_a_specialist_says_so():
    ls = llms([EMPTY], [GAP], [PLAN])
    card, _, ctx = run(ls, memory_gap_abstains=True)
    assert card.status == "abstained" and card.findings == [] and card.suggestion is None
    assert "memory found nothing" in " ".join(card.missing)
    assert ls["assistance"].calls == []  # no model call is spent on a card that is withheld
    assert [(d.reason, d.statement, d.turn_ids) for d in ctx.drops] == [
        ("memory_empty", "The transcript does not say who set it", ["t2"])]


def test_the_changed_team_abstains_on_a_consistency_finding_labelled_insufficient_too():
    card, _, ctx = run(llms([EMPTY], [INSUFFICIENT], [PLAN]), memory_gap_abstains=True)
    assert card.status == "abstained" and [d.reason for d in ctx.drops] == ["memory_empty"]


def test_the_changed_team_still_answers_when_a_verified_finding_exists_on_the_answerable_twin():
    ls = llms([MEM], [CON], [PLAN])
    card, out, ctx = run(ls, memory_gap_abstains=True)
    assert out == BEFORE["both"] and len(ls["assistance"].calls) == 1 and ctx.drops == []
    card, out, _ = run(llms([MEM], [INSUFFICIENT], [PLAN]), memory_gap_abstains=True)
    assert card.status == "answered" and {f.agent for f in card.findings} == {"memory", "consistency"}


def test_the_changed_team_is_the_old_team_when_memory_did_not_finish_cleanly():
    for memory in (["not json", "still not json"], [done({**MEM["findings"][0], "evidence": [
            {"turn_id": "t1", "quote": "a quote that is not in the turn", "speaker": "Ravi", "start_ms": 0}]})], []):
        old, _, _ = run(llms(memory, [CON], [PLAN]))
        new, _, ctx = run(llms(memory, [CON], [PLAN]), memory_gap_abstains=True)
        assert new.status == old.status == "partial" and [f.agent for f in new.findings] == ["consistency"]
        assert not any(d.reason == "memory_empty" for d in ctx.drops)


def test_the_changed_team_leaves_a_card_with_no_findings_alone():
    card, out, ctx = run(llms([EMPTY], [EMPTY], []), memory_gap_abstains=True)
    assert out == BEFORE["all_empty"] and ctx.drops == []


def test_a_follow_up_to_memory_cannot_rescue_a_withheld_card():
    ls = {"memory": FakeLLM([EMPTY, MEM]), "consistency": FakeLLM([CON]),
          "assistance": FakeLLM([{"mode": "recall", "followup": {"agent": "memory", "request": "look again"}}])}
    card, _, _ = run(ls, memory_gap_abstains=True)
    assert card.status == "abstained" and len(ls["memory"].calls) == 1
