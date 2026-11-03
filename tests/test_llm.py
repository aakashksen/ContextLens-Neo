import json
import socket
import subprocess
import urllib.error
import urllib.request

import pytest
from pydantic import BaseModel

from contextlens.llm import client as c
from contextlens.llm.client import (
    ClaudeCodeClient,
    Embedder,
    FakeLLM,
    LLMError,
    OllamaClient,
    OpenAICompatClient,
)

MSGS = [{"role": "user", "content": "hi"}]


class Answer(BaseModel):
    answer: str


def test_invalid_json_retries_once_then_raises():
    fake = FakeLLM(["not json", '{"wrong": 1}'])
    with pytest.raises(LLMError):
        fake.complete(MSGS, schema=Answer)
    assert len(fake.calls) == 2
    assert "not json" in json.dumps(fake.calls[1])  # retry shows the bad output


def test_retry_succeeds_with_valid_second_answer():
    fake = FakeLLM(["nope", '{"answer": "ok"}'])
    res = fake.complete(MSGS, schema=Answer)
    assert res.parsed == Answer(answer="ok")


def test_timeout_raises_llm_error(monkeypatch):
    def boom(*a, **k):
        raise socket.timeout("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(LLMError):
        OllamaClient("m").complete(MSGS, timeout_s=0.1)


def test_connection_refused_raises_llm_error(monkeypatch):
    def boom(*a, **k):
        raise urllib.error.URLError(ConnectionRefusedError("refused"))

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(LLMError):
        OpenAICompatClient("http://localhost:1/v1", "m").complete(MSGS)
    with pytest.raises(LLMError):
        Embedder().embed(["a"])


class Proc:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr


def _ok(text="hello"):
    return json.dumps(
        {
            "is_error": False,
            "result": text,
            "total_cost_usd": 0.002,
            "usage": {"input_tokens": 11, "output_tokens": 3},
        }
    )


@pytest.fixture
def claude(tmp_path, monkeypatch):
    monkeypatch.setattr(c.shutil, "which", lambda name: "/bin/claude")
    return ClaudeCodeClient(calls_path=tmp_path / "claude_calls.json")


def test_call_cap_refuses_and_makes_no_call(tmp_path, monkeypatch):
    path = tmp_path / "claude_calls.json"
    path.write_text(json.dumps({"count": 1000, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0}))
    monkeypatch.setattr(c.shutil, "which", lambda name: "/bin/claude")

    def no_run(*a, **k):
        raise AssertionError("process must not start")

    monkeypatch.setattr(subprocess, "run", no_run)
    with pytest.raises(LLMError):
        ClaudeCodeClient(calls_path=path).complete(MSGS)
    assert json.loads(path.read_text())["count"] == 1000


def test_claude_client_builds_expected_command_with_no_tools(claude, monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"], seen["kw"] = cmd, kw
        return Proc(_ok())

    monkeypatch.setattr(subprocess, "run", fake_run)
    res = claude.complete(MSGS)
    cmd = seen["cmd"]
    assert cmd[0] == "/bin/claude" and "-p" in cmd
    assert cmd[cmd.index("--model") + 1] == "haiku"
    assert cmd[cmd.index("--output-format") + 1] == "json"
    assert cmd[cmd.index("--tools") + 1] == ""
    assert cmd[cmd.index("--setting-sources") + 1] == ""
    assert cmd[cmd.index("--max-turns") + 1] == "1"
    for flag in ("--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"):
        assert flag in cmd
    assert "hi" in seen["kw"]["input"]
    assert str(seen["kw"]["cwd"]) != str(c.REPO_ROOT)
    assert res.text == "hello"


def test_usage_limit_raises_llm_error(claude, monkeypatch):
    limit = json.dumps({"is_error": True, "result": "Claude AI usage limit reached"})
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: Proc(limit))
    with pytest.raises(LLMError):
        claude.complete(MSGS)
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: Proc("", 1, "boom"))
    with pytest.raises(LLMError):
        claude.complete(MSGS)


def test_usage_is_recorded(claude, monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: Proc(_ok()))
    res = claude.complete(MSGS)
    claude.complete(MSGS)
    data = json.loads((tmp_path / "claude_calls.json").read_text())
    assert data["count"] == 2
    assert data["input_tokens"] == 22 and data["output_tokens"] == 6
    assert data["cost_usd"] == pytest.approx(0.004)
    assert (res.prompt_tokens, res.completion_tokens, res.cost_usd) == (11, 3, 0.002)


@pytest.mark.real
def test_ollama_round_trip():
    res = OllamaClient("qwen3.5:4b-q4_K_M").complete(MSGS, timeout_s=60)
    assert res.text


@pytest.mark.real
def test_claude_haiku_round_trip():
    res = ClaudeCodeClient().complete(MSGS, timeout_s=120)
    assert res.text
