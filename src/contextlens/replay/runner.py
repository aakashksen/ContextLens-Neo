import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Protocol, Self

from pydantic import BaseModel, Field, model_validator

from contextlens.memory.store import InMemoryTranscriptStore, TranscriptStore
from contextlens.schemas import ButtonRequest, Card, ConsistencyLabel, Scenario


class DroppedFinding(BaseModel):
    """A finding a check removed from a card, kept so a wrongful drop can be counted afterwards."""

    reason: str  # citation, instruction or number
    statement: str
    turn_ids: list[str]


@dataclass
class Context:
    store: TranscriptStore
    as_of_seq: int
    stage_ms: dict[str, int] = field(default_factory=dict)  # an assistant adds its stage timings here
    drops: list[DroppedFinding] = field(default_factory=list)  # an assistant adds the findings its checks dropped


class Assistant(Protocol):
    def answer(self, req: ButtonRequest, ctx: Context) -> Card: ...


class Prediction(BaseModel):
    scenario_id: str
    card: Card | None = None  # a retrieval-only row holds ranked turns instead of a card
    ranked_turn_ids: list[str] | None = None
    claim_supported: list[bool] | None = None  # rubric judgement per finding, in card order
    cloud_supported: list[bool] | None = None  # the cloud grader's judgement per finding; set only by the judge check
    raw_output: str | None = None  # model text, kept so schema validity can be recomputed
    stage_latency_ms: dict[str, int] = Field(default_factory=dict)
    errored: bool = False  # the card is a runner-made error card whose mode is a placeholder
    label: ConsistencyLabel | None = None  # a bare consistency verdict, for rows that may hold no text
    probability: float | None = Field(default=None, ge=0, le=1)  # the model's probability for the label it gave
    drops: list[DroppedFinding] = Field(default_factory=list)  # findings the checks removed, with their text

    @model_validator(mode="after")
    def _has_output(self) -> Self:
        if self.card is None and self.ranked_turn_ids is None and self.label is None:
            raise ValueError("a prediction row needs a card, ranked_turn_ids or a label")
        return self


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
            errored = False
        except Exception as exc:
            card = Card(
                mode="recall", status="abstained", findings=[],
                missing=[f"error: {type(exc).__name__}: {exc}"],
            )
            errored = True
        ms = int((time.perf_counter() - start) * 1000)
        out.append(Prediction(
            scenario_id=sc.id, card=card, stage_latency_ms={**ctx.stage_ms, "total": ms}, errored=errored,
            drops=ctx.drops,
        ))
    return out
