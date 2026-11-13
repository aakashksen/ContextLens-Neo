"""FastAPI app for /ws/{conversation_id}; see docs/client-contract.md."""

import asyncio
import contextlib
import logging
import time

from fastapi import FastAPI, Response, WebSocket

from contextlens.api.stream import Connection, ConversationState, Services

log = logging.getLogger(__name__)


def _sweep(conversations: dict[str, ConversationState], idle_evict_s: float) -> None:
    """Forget in-memory state of conversations nobody is connected to. Stored turns stay; a later
    connection rebuilds its state from them."""
    for cid, conv in list(conversations.items()):
        if conv.idle(idle_evict_s):
            conv.stop_background()
            del conversations[cid]


async def _shutdown(services: Services, conversations: dict[str, ConversationState]) -> None:
    if services.warmup is not None:
        services.warmup.stop()
    for conv in conversations.values():
        conv.stop_background()
    if services.on_shutdown is not None:
        try:
            await asyncio.to_thread(services.on_shutdown)
        except Exception as exc:
            log.warning("shutdown step failed: %s", type(exc).__name__)


def create_app(services: Services) -> FastAPI:
    conversations: dict[str, ConversationState] = {}
    guard = asyncio.Lock()

    def sweep() -> None:
        _sweep(conversations, services.idle_evict_s)

    async def sweeper() -> None:
        while True:
            await asyncio.sleep(max(1.0, services.idle_evict_s / 2))
            sweep()

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        task = asyncio.get_running_loop().create_task(sweeper())
        if services.warmup is not None:
            services.warmup.start()
        try:
            yield
        finally:
            task.cancel()
            await _shutdown(services, conversations)

    app = FastAPI(title="ContextLens Neo", lifespan=lifespan)
    app.state.conversations = conversations

    @app.get("/ready")
    def ready(response: Response) -> dict:
        """200 once the models are loaded (or when there is nothing to warm); 503 before, with what each load took."""
        body = services.warmup.status() if services.warmup else {"ready": True, "cold_start_ms": {}, "errors": {}}
        response.status_code = 200 if body["ready"] else 503
        return body

    @app.websocket("/ws/{conversation_id}")
    async def ws(websocket: WebSocket, conversation_id: str) -> None:
        if services.warmup is not None and not services.warmup.ready.is_set():
            await websocket.close(code=1013)  # try again later: a press now would race a cold model load
            return
        async with guard:
            sweep()
            conv = conversations.get(conversation_id)
            if conv is None:
                conv = conversations[conversation_id] = await asyncio.to_thread(
                    ConversationState, conversation_id, services
                )
        conv.last_active = time.monotonic()
        conv.start_backend_load()
        await Connection(websocket, conv).run()

    return app
