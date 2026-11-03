import sys
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import TypeAdapter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from contextlens.api.contract import (  # noqa: E402
    CardMessage,
    ServerMessage,
    StatusMessage,
    TranscriptMessage,
    schema_json,
)
from fake_server import create_app  # noqa: E402

PRESS = {"type": "press", "request": {"conversation_id": "c1", "as_of_seq": 3}}
server_msg = TypeAdapter(ServerMessage)


def test_schema_file_matches_models():
    on_disk = (ROOT / "docs" / "client-contract.schema.json").read_text(encoding="utf-8")
    assert on_disk == schema_json()


def _press(ws):
    ws.send_json(PRESS)
    return server_msg.validate_json(ws.receive_text())


def test_fake_server_answers_press_with_valid_card():
    client = TestClient(create_app())
    with client.websocket_connect("/ws/c1") as ws:
        assert isinstance(server_msg.validate_json(ws.receive_text()), StatusMessage)
        ws.send_json({"type": "audio", "data": "AAAA"})
        t = server_msg.validate_json(ws.receive_text())
        assert isinstance(t, TranscriptMessage) and t.turn.conversation_id == "c1"
        msg = _press(ws)
        assert isinstance(msg, CardMessage)
        assert msg.card.status == "answered" and msg.card.findings


def test_fake_server_can_send_answered_partial_and_abstained():
    client = TestClient(create_app())
    with client.websocket_connect("/ws/c1") as ws:
        ws.receive_text()
        statuses = [_press(ws).card.status for _ in range(3)]
    assert statuses == ["answered", "partial", "abstained"]
