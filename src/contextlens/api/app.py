"""FastAPI app for /ws/{conversation_id}; see docs/client-contract.md."""

import asyncio
import contextlib
import logging
import time

from fastapi import FastAPI, WebSocket

from contextlens.api.stream import Connection, ConversationState, Services

log = logging.getLogger(__name__)


def create_app(services: Services) -> FastAPI:
    conversations: dict[str, ConversationState] = {}
    guard = asyncio.Lock()

    def sweep() -> None:
        """Forget in-memory state of conversations nobody is connected to. Stored turns stay; a later
        connection rebuilds its state from them."""
        for cid, conv in list(conversations.items()):
            if conv.idle(services.idle_evict_s):
                conv.stop_background()
                del conversations[cid]

    async def sweeper() -> None:
        while True:
            await asyncio.sleep(max(1.0, services.idle_evict_s / 2))
            sweep()

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        task = asyncio.get_running_loop().create_task(sweeper())
        try:
            yield
        finally:
            task.cancel()
            for conv in conversations.values():
                conv.stop_background()
            if services.on_shutdown is not None:
                try:
                    await asyncio.to_thread(services.on_shutdown)
                except Exception as exc:
                    log.warning("shutdown step failed: %s", type(exc).__name__)

    app = FastAPI(title="ContextLens", lifespan=lifespan)
    app.state.conversations = conversations

    @app.websocket("/ws/{conversation_id}")
    async def ws(websocket: WebSocket, conversation_id: str) -> None:
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
