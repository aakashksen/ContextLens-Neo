"""The real pieces behind the live mode, wired to the production services: the stream API's own store and memory, the
production assistants by name, and the shipped cascade checker. `make_rig` takes the services and the builders as
arguments so a test can hand in fakes; `connect_ws` is the one place that opens a real websocket."""

import asyncio
import time
from collections.abc import AsyncIterator, Callable

from contextlens.agents.precheck import prepare
from contextlens.agents.tools import Tools
from contextlens.api.stream import ALL, Services, _FrozenStore
from contextlens.replay.runner import Context
from contextlens.room.live import LiveRig, Transport
from contextlens.schemas import ButtonRequest, Card, Turn

FLAG_WINDOW = 6  # recent turns the checker's latest-turn check is given as context
FLAG_DEADLINE_S = 12.0
CONNECT_TRIES = 60  # the stream API refuses connections until its models are loaded


class WsTransport:
    def __init__(self, ws) -> None:
        self.ws = ws

    async def send(self, text: str) -> None:
        await self.ws.send(text)

    async def messages(self) -> AsyncIterator[str]:
        async for m in self.ws:
            yield m if isinstance(m, str) else m.decode("utf-8")

    async def close(self) -> None:
        await self.ws.close()


async def connect_ws(base: str, conv: str) -> Transport:
    """A websocket to the stream API's /ws/{conv}. Retries for up to a minute while the models warm up."""
    import websockets

    last: Exception | None = None
    for _ in range(CONNECT_TRIES):
        try:
            return WsTransport(await websockets.connect(f"{base}/ws/{conv}", max_size=2**22))
        except Exception as exc:
            last = exc
            await asyncio.sleep(1.0)
    raise ConnectionError(type(last).__name__)


def make_rig(svc: Services, llms: dict, base: str, *, build: Callable, checker: Callable, connect: Callable = connect_ws,
             deadline_s: float = 8.0) -> LiveRig:
    """`build(name, store, retriever, memory, llms, deadline_s)` is production.build_assistant; `checker()` returns the
    consistency checker (production.cascade_checker)."""

    def check_flag(conv: str, turn: Turn) -> dict | None:
        window = svc.store.turns(conv, turn.seq)[-FLAG_WINDOW:]
        if not window or window[-1].id != turn.id:
            return None
        tools = Tools(svc.store, svc.retriever, svc.memory, turn.seq, conv, checker())
        prep = prepare(tools, ButtonRequest(conversation_id=conv, as_of_seq=turn.seq), window, time.perf_counter() + FLAG_DEADLINE_S)
        if not prep.flagged:
            return None
        return {"label": prep.label, "confidence": round(prep.confidence, 3), "earlier_ids": prep.earlier_ids,
                "checker_ms": prep.checker_ms, "search_ms": prep.search_ms}

    def answer(name: str, conv: str, as_of: int, question: str | None) -> tuple[Card, Context]:
        assistant = build(name, svc.store, svc.retriever, svc.memory, llms, deadline_s)
        ctx = Context(store=_FrozenStore(svc.store, as_of), as_of_seq=as_of)
        return assistant.answer(ButtonRequest(conversation_id=conv, as_of_seq=as_of, question=question), ctx), ctx

    async def connector(conv: str) -> Transport:
        return await connect(base, conv)

    return LiveRig(connect=connector, turns=lambda conv: svc.store.turns(conv, ALL),
                   memory=lambda conv: svc.memory.items(conv, ALL), check_flag=check_flag, answer=answer)
