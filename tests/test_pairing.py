from types import SimpleNamespace

from contextlens.consistency import pairing as P
from contextlens.schemas import Turn


def turn(seq: int, text: str, speaker: str = "A") -> Turn:
    return Turn(id=f"t{seq}", conversation_id="c", seq=seq, speaker=speaker, text=text, start_ms=seq, end_ms=seq + 1,
                source="synthetic")


POOL = [
    turn(0, "The venue deposit for the offsite is 12 hundred euros."),
    turn(1, "Yeah so we should probably talk about the weather."),
    turn(2, "I think the printer on the second floor is broken again.", "B"),
    turn(3, "Right okay.", "B"),
]
LATEST = turn(10, "The venue deposit for the offsite went up to 18 hundred euros, they revised the quote.")


def test_key_terms_drop_fillers_and_keep_numbers():
    terms = P.key_terms("Yeah, right, the deposit is 18 hundred euros.")
    assert "18" in terms and "deposit" in terms and "yeah" not in terms and "the" not in terms


def test_rank_puts_the_matching_turn_first_and_drops_unrelated_ones():
    assert [t.id for t in P.rank_candidates(LATEST, POOL, k=4)] == ["t0"]


def test_rank_ignores_later_turns_and_the_statement_itself():
    assert P.rank_candidates(LATEST, [LATEST, turn(11, "The venue deposit is 12.")]) == []


def test_same_speaker_breaks_a_tie():
    a, b = turn(0, "Budget is 5 euros.", "X"), turn(1, "Budget is 5 euros.", "Y")
    assert P.rank_candidates(turn(5, "Budget is 7 euros.", "Y"), [a, b], k=1)[0].id == "t1"


def pair(trained: dict, nli: list[dict] | None = None, earlier: Turn | None = None) -> P.PairScore:
    return P.PairScore(earlier or POOL[0], trained, nli or [])


def nli(c: float) -> dict:
    return {"entailment": 0.0, "neutral": 1 - c, "contradiction": c}


def test_nli_conflict_is_a_contradiction_unless_trained_model_says_update():
    rule = P.Rule()
    assert P.decide([pair({"update": 0.1, "contradiction": 0.1}, [nli(0.95)])], rule)[0] == "contradiction"
    assert P.decide([pair({"update": 0.9, "contradiction": 0.0}, [nli(0.95)])], rule)[0] == "update"


def test_update_rescue_needs_a_high_trained_update_and_no_restatement():
    rule = P.Rule()
    low = P.decide([pair({"update": 0.95, "consistent": 0.05}, [nli(0.1)])], rule, LATEST)
    assert low[0] == "update" and low[2] == ["t0"]
    restated = turn(10, "Correct, the venue deposit for the offsite is 12 hundred euros.")
    assert P.decide([pair({"update": 0.95, "consistent": 0.05}, [nli(0.1)])], rule, restated)[0] == "consistent"
    assert P.decide([pair({"update": 0.95}, [nli(0.1)])], P.Rule(update_rescue=None), LATEST)[0] != "update"


def test_nli_flagged_pair_beats_a_rescued_one_even_with_lower_score():
    rescued = pair({"update": 1.0}, [nli(0.0)], POOL[2])
    flagged = pair({"update": 0.0, "contradiction": 0.0}, [nli(0.6)], POOL[0])
    assert P.decide([rescued, flagged], P.Rule(), LATEST)[2] == ["t0"]


def test_no_candidates_is_insufficient_and_no_conflict_is_consistent():
    assert P.decide([], P.Rule()) == ("insufficient", 0.0, [])
    lab, _, ids = P.decide([pair({"consistent": 0.9, "insufficient": 0.1}, [nli(0.1)])], P.Rule())
    assert (lab, ids) == ("consistent", [])


def test_trained_only_rule():
    assert P.decide([pair({"contradiction": 0.7, "update": 0.1})], P.Rule())[0] == "contradiction"
    assert P.decide([pair({"consistent": 0.9, "contradiction": 0.05, "update": 0.05})], P.Rule())[0] == "consistent"


def test_gates():
    assert P.is_hedged("I'd eat a whole cow for that, joking of course") and not P.is_hedged("The deposit is 12.")
    assert not P.same_claim_possible(turn(2, "I'm coming to the retreat", "Jo"), turn(1, "I'm not coming", "Ines"))
    assert P.same_claim_possible(turn(2, "I'm on the late shift", "Kai"), turn(1, "I'm on the early shift", "Kai"))
    assert P.same_claim_possible(turn(2, "We moved the deadline", "Sue"), turn(1, "The deadline is Monday", "Ravi"))
    assert P.restates(turn(5, "Yes, the deposit is 12 euros."), turn(1, "The deposit is 12 euros."))
    assert not P.restates(turn(5, "The deposit is not 12 euros."), turn(1, "The deposit is 12 euros."))


def test_find_conflict_uses_search_hits_and_returns_the_cited_turn(monkeypatch):
    far = turn(0, "The venue deposit for the offsite is 12 hundred euros.")
    seen = {}

    def fake_score(trained, nlis, statement, cands):
        seen["cands"] = [c.id for c in cands]
        return [P.PairScore(c, {"update": 0.0, "contradiction": 0.0}, [nli(0.99)]) for c in cands]

    monkeypatch.setattr(P, "score_pairs", fake_score)
    tools = SimpleNamespace(search_transcript=lambda q, k: [SimpleNamespace(turn=far)])
    label, conf, ids = P.find_conflict(tools, LATEST, [turn(1, "Yeah so the weather.")], trained=None)
    assert (label, ids) == ("contradiction", ["t0"]) and conf > 0.9 and seen["cands"] == ["t0"]


def test_find_conflict_survives_a_failing_search_and_skips_hedged_statements(monkeypatch):
    def boom(q, k):
        raise RuntimeError("down")

    monkeypatch.setattr(P, "score_pairs", lambda *a: [])
    assert P.find_conflict(SimpleNamespace(search_transcript=boom), LATEST, POOL, trained=None)[0] == "insufficient"
    joke = turn(11, "The venue deposit for the offsite is 99 euros, just joking.")
    monkeypatch.setattr(P, "score_pairs", lambda *a: (_ for _ in ()).throw(AssertionError("must not run")))
    assert P.find_conflict(None, joke, POOL, trained=None)[0] == "consistent"
