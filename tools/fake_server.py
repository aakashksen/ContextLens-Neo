"""Serves the /ws/{conversation_id} contract with invented, scripted content.

Run: python tools/fake_server.py  (port 8000)
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from contextlens.api.contract import (
    CardMessage,
    StatusMessage,
    TranscriptMessage,
    client_adapter,
)
from contextlens.schemas import Card, EvidenceRef, Finding, Turn

SCRIPT = [
    ("Ana", "Let us move the launch to Friday."),
    ("Ben", "Friday works, but I need the budget sheet first."),
    ("Ana", "I will send the budget sheet tomorrow morning."),
    ("Ben", "Actually, make that Monday, I am out tomorrow."),
]


def scripted_turn(conversation_id: str, i: int) -> Turn:
    speaker, text = SCRIPT[i % len(SCRIPT)]
    return Turn(
        id=f"{conversation_id}-t{i}",
        conversation_id=conversation_id,
        seq=i,
        speaker=speaker,
        text=text,
        start_ms=i * 4000,
        end_ms=i * 4000 + 3500,
        source="synthetic",
    )


def scripted_card(conversation_id: str, n: int) -> Card:
    """Cycle answered, partial, abstained on successive presses."""
    kind = n % 3
    evidence = [
        EvidenceRef(
            turn_id=f"{conversation_id}-t2",
            quote=SCRIPT[2][1],
            speaker=SCRIPT[2][0],
            start_ms=8000,
        )
    ]
    finding = Finding(
        agent="recall",
        kind="recall",
        statement="Ana promised to send the budget sheet.",
        evidence=evidence,
        confidence=0.9,
    )
    if kind == 0:
        return Card(
            mode="recall",
            status="answered",
            findings=[finding],
            suggestion="Ask Ana whether the sheet is ready.",
            latency_ms=1200,
            tokens=350,
        )
    if kind == 1:
        return Card(
            mode="recall",
            status="partial",
            findings=[finding],
            missing=["the new due date"],
            latency_ms=1500,
            tokens=410,
        )
    return Card(
        mode="recall",
        status="abstained",
        findings=[],
        missing=["no matching statement in the conversation"],
        latency_ms=800,
        tokens=120,
    )


def create_app() -> FastAPI:
    app = FastAPI(title="ContextLens fake server")

    @app.websocket("/ws/{conversation_id}")
    async def ws(websocket: WebSocket, conversation_id: str) -> None:
        await websocket.accept()
        heard = 0
        presses = 0
        paused = False

        async def send(msg) -> None:
            await websocket.send_text(msg.model_dump_json())

        await send(StatusMessage(state="recording"))
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    msg = client_adapter.validate_json(raw)
                except ValidationError as e:
                    await send(StatusMessage(state="error", detail=e.errors()[0]["msg"]))
                    continue
                if msg.type == "audio":
                    if not paused:
                        turn = scripted_turn(conversation_id, heard)
                        heard += 1
                        await send(TranscriptMessage(turn=turn))
                elif msg.type == "press":
                    card = scripted_card(conversation_id, presses)
                    presses += 1
                    await send(CardMessage(card=card))
                elif msg.type == "pause":
                    paused = True
                    await send(StatusMessage(state="paused"))
                elif msg.type == "resume":
                    paused = False
                    await send(StatusMessage(state="recording"))
                elif msg.type == "delete":
                    await send(StatusMessage(state="deleted"))
                    await websocket.close()
                    return
        except WebSocketDisconnect:
            return

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
