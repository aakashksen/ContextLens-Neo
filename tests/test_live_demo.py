"""The local live demo API, with the assistant faked (no model). One real press is marked `real`."""
import json
import sys
import threading
import urllib.error
import urllib.request

import pytest

sys.path.insert(0, "tools")
import live_demo  # noqa: E402

from contextlens.schemas import Card, EvidenceRef, Finding  # noqa: E402


class FakeAssistant:
    def __init__(self, kind, store):
        self.kind, self.store = kind, store

    def answer(self, req, ctx):
        turns = ctx.store.turns(req.conversation_id, req.as_of_seq)
        t = turns[-1]
        ev = EvidenceRef(turn_id=t.id, quote=t.text, speaker=t.speaker, start_ms=t.start_ms)
        f = Finding(agent=self.kind, kind="recall", statement=f"q={req.question}", evidence=[ev], confidence=0.9)
        return Card(mode="recall", status="answered", findings=[f], latency_ms=123, tokens=45)


def fake_press(body):
    return live_demo.press(body, factory=lambda kind, store: FakeAssistant(kind, store))


BODY = {"turns": [{"speaker": "Maya", "text": "Ship Friday."}, {"speaker": "Jo", "text": "Make it Monday."}],
        "question": "When?", "assistant": "team"}


@pytest.fixture()
def server():
    srv = live_demo.make_server(0, press_fn=fake_press)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def call(url, body=None):
    data = None if body is None else json.dumps(body).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_health_and_static_site(server):
    code, body = call(server + "/api/health")
    assert code == 200 and json.loads(body)["ok"] is True
    code, body = call(server + "/live.html")
    assert code == 200 and b"live-local.js" in body


def test_press_returns_card_latency_tokens(server):
    code, body = call(server + "/api/press", BODY)
    out = json.loads(body)
    assert code == 200 and out["assistant"] == "team" and out["question"] == "When?"
    assert out["card"]["findings"][0]["evidence"][0]["turn_id"] == "t2"
    assert out["latency_ms"] == 123 and out["tokens"] == 45


def test_null_question_and_single(server):
    code, body = call(server + "/api/press", {**BODY, "question": None, "assistant": "single"})
    out = json.loads(body)
    assert code == 200 and out["question"] is None and out["card"]["findings"][0]["agent"] == "single"


@pytest.mark.parametrize("bad", [
    {**BODY, "assistant": "other"}, {**BODY, "turns": []}, {**BODY, "turns": [{"speaker": "", "text": "x"}]},
    {**BODY, "turns": "no"}, {**BODY, "question": 5}, {**BODY, "turns": [{"speaker": "a", "text": "b"}] * 41},
])
def test_bad_requests_are_400(server, bad):
    assert call(server + "/api/press", bad)[0] == 400


def test_unknown_post_and_bad_json(server):
    assert call(server + "/api/nope", BODY)[0] == 404
    req = urllib.request.Request(server + "/api/press", data=b"{", method="POST")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=10)
    assert e.value.code == 400


def test_model_failure_is_502_without_content(server):
    def boom(body):
        raise RuntimeError("secret meeting text")

    srv = live_demo.make_server(0, press_fn=boom)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        code, body = call(f"http://127.0.0.1:{srv.server_address[1]}/api/press", BODY)
    finally:
        srv.shutdown()
        srv.server_close()
    assert code == 502 and b"secret" not in body


def test_listens_on_loopback_only(server):
    assert live_demo.make_server(0).server_address[0] == "127.0.0.1"
