"""The one source of truth for the /ws/{conversation_id} messages."""

import json
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter

from contextlens.schemas import ButtonRequest, Card, Turn


class AudioMessage(BaseModel):
    type: Literal["audio"] = "audio"
    data: str  # base64 of 16-bit mono PCM
    sample_rate: int = 16000


class PressMessage(BaseModel):
    type: Literal["press"] = "press"
    request: ButtonRequest


class PauseMessage(BaseModel):
    type: Literal["pause"] = "pause"


class ResumeMessage(BaseModel):
    type: Literal["resume"] = "resume"


class DeleteMessage(BaseModel):
    type: Literal["delete"] = "delete"


class TranscriptMessage(BaseModel):
    type: Literal["transcript"] = "transcript"
    turn: Turn


class TurnUpdateMessage(BaseModel):
    """A turn already sent as `transcript` now has a (new) speaker. Same id and seq; replace, do not append."""

    type: Literal["turn_update"] = "turn_update"
    turn: Turn


class CardMessage(BaseModel):
    type: Literal["card"] = "card"
    card: Card


class StatusMessage(BaseModel):
    type: Literal["status"] = "status"
    state: Literal["recording", "paused", "deleted", "error"]
    detail: str | None = None


ClientMessage = Annotated[
    AudioMessage | PressMessage | PauseMessage | ResumeMessage | DeleteMessage,
    Field(discriminator="type"),
]
ServerMessage = Annotated[
    TranscriptMessage | TurnUpdateMessage | CardMessage | StatusMessage, Field(discriminator="type")
]

client_adapter: TypeAdapter = TypeAdapter(ClientMessage)
server_adapter: TypeAdapter = TypeAdapter(ServerMessage)


def contract_schema() -> dict:
    return {
        "title": "ContextLens /ws/{conversation_id}",
        "client_to_server": client_adapter.json_schema(),
        "server_to_client": server_adapter.json_schema(),
    }


def schema_json() -> str:
    return json.dumps(contract_schema(), indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    import sys

    sys.stdout.write(schema_json())
