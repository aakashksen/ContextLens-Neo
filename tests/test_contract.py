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


def _hear(ws, n):
    turns = []
    for _ in range(n):
        ws.send_json({"type": "audio", "data": "AAAA"})
        turns.append(server_msg.validate_json(ws.receive_text()).turn)
    return turns


def _press_as(ws, as_of_seq, conversation_id="c1"):
    ws.send_json(
        {"type": "press", "request": {"conversation_id": conversation_id, "as_of_seq": as_of_seq}}
    )
    return server_msg.validate_json(ws.receive_text())


def _is_error(msg):
    return isinstance(msg, StatusMessage) and msg.state == "error"


def test_card_cites_only_sent_turn_at_or_before_as_of_seq():
    with TestClient(create_app()).websocket_connect("/ws/c1") as ws:
        ws.receive_text()
        turns = _hear(ws, 4)
        msg = _press_as(ws, 1)
        ev = msg.card.findings[0].evidence[0]
        assert (ev.turn_id, ev.quote, ev.speaker, ev.start_ms) == (
            turns[1].id,
            turns[1].text,
            turns[1].speaker,
            turns[1].start_ms,
        )


def test_press_before_any_turn_is_an_error():
    with TestClient(create_app()).websocket_connect("/ws/c1") as ws:
        ws.receive_text()
        assert _is_error(_press_as(ws, 0))


def test_press_past_last_turn_is_an_error():
    with TestClient(create_app()).websocket_connect("/ws/c1") as ws:
        ws.receive_text()
        _hear(ws, 2)
        assert _is_error(_press_as(ws, 99))
        assert isinstance(_press_as(ws, 1), CardMessage)


def test_negative_as_of_seq_is_an_error():
    with TestClient(create_app()).websocket_connect("/ws/c1") as ws:
        ws.receive_text()
        _hear(ws, 2)
        assert _is_error(_press_as(ws, -5))


def test_press_for_other_conversation_is_an_error():
    with TestClient(create_app()).websocket_connect("/ws/c1") as ws:
        ws.receive_text()
        _hear(ws, 2)
        assert _is_error(_press_as(ws, 1, conversation_id="other"))


def test_delete_clears_turns_that_could_be_cited():
    client = TestClient(create_app())
    with client.websocket_connect("/ws/c1") as ws:
        ws.receive_text()
        _hear(ws, 2)
        ws.send_json({"type": "delete"})
        assert server_msg.validate_json(ws.receive_text()).state == "deleted"
    with client.websocket_connect("/ws/c1") as ws:
        ws.receive_text()
        assert _is_error(_press_as(ws, 1))


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
        msg = _press_as(ws, t.turn.seq)
        assert isinstance(msg, CardMessage)
        assert msg.card.status == "answered" and msg.card.findings


def test_fake_server_can_send_answered_partial_and_abstained():
    client = TestClient(create_app())
    with client.websocket_connect("/ws/c1") as ws:
        ws.receive_text()
        _hear(ws, 4)
        statuses = [_press(ws).card.status for _ in range(3)]
    assert statuses == ["answered", "partial", "abstained"]


def test_binary_frame_gets_error_status_and_connection_stays_open():
    with TestClient(create_app()).websocket_connect("/ws/c1") as ws:
        ws.receive_text()
        ws.send_bytes(b"\x00\x01")
        msg = server_msg.validate_json(ws.receive_text())
        assert _is_error(msg)
        _hear(ws, 1)  # still usable


def test_client_disconnect_ends_cleanly_and_turns_survive_reconnect():
    client = TestClient(create_app())
    with client.websocket_connect("/ws/c1") as ws:
        ws.receive_text()
        _hear(ws, 2)
    with client.websocket_connect("/ws/c1") as ws:
        ws.receive_text()
        assert isinstance(_press_as(ws, 1), CardMessage)
        # the press cycle restarts on each connection
        assert _press_as(ws, 1).card.status == "partial"
    with client.websocket_connect("/ws/c1") as ws:
        ws.receive_text()
        assert _press_as(ws, 1).card.status == "answered"


def test_every_message_the_fake_sends_validates_against_schema_file():
    import json

    import jsonschema

    schema = json.loads((ROOT / "docs" / "client-contract.schema.json").read_text("utf-8"))
    server_schema = {**schema["server_to_client"]}
    client_schema = {**schema["client_to_server"]}
    seen = set()
    with TestClient(create_app()).websocket_connect("/ws/c1") as ws:

        def check_server(raw):
            data = json.loads(raw)
            jsonschema.validate(data, server_schema)
            seen.add(data["type"])
            return data

        def send(payload):
            jsonschema.validate(payload, client_schema)
            ws.send_json(payload)

        check_server(ws.receive_text())
        for _ in range(4):
            send({"type": "audio", "data": "AAAA"})
            check_server(ws.receive_text())
        for _ in range(3):
            send(PRESS)
            check_server(ws.receive_text())
        send({"type": "pause"})
        check_server(ws.receive_text())
        send({"type": "resume"})
        check_server(ws.receive_text())
        ws.send_text("not json")
        data = check_server(ws.receive_text())
        assert data["state"] == "error"
        send({"type": "delete"})
        check_server(ws.receive_text())
    assert seen == {"transcript", "card", "status"}
