from contextlens.replay.runner import Context, replay
from contextlens.schemas import ButtonRequest, Card, Conversation, Scenario, Turn


def _scenario(sid: str, n: int = 6, as_of: int = 2) -> Scenario:
    turns = [
        Turn(id=f"{sid}-t{i}", conversation_id=sid, seq=i, speaker="A", text=f"w{i}",
             start_ms=i * 1000, end_ms=i * 1000 + 900, source="synthetic")
        for i in range(n)
    ]
    return Scenario(
        id=sid, conversation=Conversation(id=sid, turns=turns, source="test"),
        request=ButtonRequest(conversation_id=sid, as_of_seq=as_of), category="c",
        gold_evidence_turn_ids=[], gold_label=None, gold_answer=None, origin="synthetic",
    )


class Spy:
    seen: list[int] = []

    def answer(self, req: ButtonRequest, ctx: Context) -> Card:
        Spy.seen = [t.seq for t in ctx.store.turns(req.conversation_id, 10_000)]
        return Card(mode="recall", status="partial", findings=[])


class Boom:
    def answer(self, req: ButtonRequest, ctx: Context) -> Card:
        raise RuntimeError("kaboom")


class Echo:
    def answer(self, req: ButtonRequest, ctx: Context) -> Card:
        return Card(mode="recall", status="partial", findings=[], missing=[req.conversation_id])


def test_assistant_cannot_see_future_turns() -> None:
    replay([_scenario("s1")], Spy)
    assert Spy.seen and max(Spy.seen) <= 2


def test_exception_becomes_abstained_card() -> None:
    [p] = replay([_scenario("s1")], Boom)
    assert p.card.status == "abstained"
    assert any("kaboom" in m for m in p.card.missing)


def test_results_are_order_stable() -> None:
    scs = [_scenario(f"s{i}") for i in range(4)]
    preds = replay(scs, Echo)
    assert [p.scenario_id for p in preds] == [s.id for s in scs]
    assert [p.card.missing[0] for p in preds] == [s.id for s in scs]
