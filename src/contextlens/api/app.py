"""FastAPI app for /ws/{conversation_id}; see docs/client-contract.md."""

import asyncio

from fastapi import FastAPI, WebSocket

from contextlens.api.stream import Connection, ConversationState, Services


def create_app(services: Services) -> FastAPI:
    app = FastAPI(title="ContextLens")
    conversations: dict[str, ConversationState] = {}
    guard = asyncio.Lock()

    @app.websocket("/ws/{conversation_id}")
    async def ws(websocket: WebSocket, conversation_id: str) -> None:
        async with guard:
            conv = conversations.get(conversation_id)
            if conv is None:
                conv = conversations[conversation_id] = await asyncio.to_thread(
                    ConversationState, conversation_id, services
                )
        await Connection(websocket, conv).run()

    return app
