import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, Field

from contextlens.memory.store import InMemoryTranscriptStore, TranscriptStore
from contextlens.schemas import ButtonRequest, Card, Scenario


@dataclass
class Context:
    store: TranscriptStore
    as_of_seq: int


class Assistant(Protocol):
    def answer(self, req: ButtonRequest, ctx: Context) -> Card: ...


class Prediction(BaseModel):
    scenario_id: str
    card: Card
    stage_latency_ms: dict[str, int] = Field(default_factory=dict)


def replay(
    scenarios: Iterable[Scenario], make_assistant: Callable[[], Assistant]
) -> list[Prediction]:
    out: list[Prediction] = []
    for sc in scenarios:
        store = InMemoryTranscriptStore()
        for turn in sorted(sc.conversation.turns, key=lambda t: t.seq):
            if turn.seq > sc.request.as_of_seq:
                break
            store.add(turn)
        ctx = Context(store=store, as_of_seq=sc.request.as_of_seq)
        start = time.perf_counter()
        try:
            card = make_assistant().answer(sc.request, ctx)
        except Exception as exc:
            card = Card(
                mode="recall", status="abstained", findings=[],
                missing=[f"error: {type(exc).__name__}: {exc}"],
            )
        ms = int((time.perf_counter() - start) * 1000)
        out.append(Prediction(scenario_id=sc.id, card=card, stage_latency_ms={"total": ms}))
    return out
