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


class Counting(FakeLLM):
    def _raw(self, *a):
        res = super()._raw(*a)
        return res.model_copy(update={"prompt_tokens": 10, "completion_tokens": 5, "cost_usd": 0.5})


def test_retry_success_reports_both_attempts():
    res = Counting(["nope", '{"answer": "ok"}']).complete(MSGS, schema=Answer)
    assert (res.prompt_tokens, res.completion_tokens, res.cost_usd) == (20, 10, 1.0)


def test_failed_attempts_are_carried_on_the_error():
    with pytest.raises(LLMError) as err:
        Counting(["nope", "still nope"]).complete(MSGS, schema=Answer)
    assert [r.prompt_tokens for r in err.value.results] == [10, 10]
    with pytest.raises(LLMError) as err:
        Counting(["nope"]).complete(MSGS, schema=Answer)  # the retry itself fails
    assert [r.prompt_tokens for r in err.value.results] == [10]
    assert LLMError("plain").results == []


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
    path.write_text(json.dumps({"count": c.CLAUDE_CALL_CAP, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0}))
    monkeypatch.setattr(c.shutil, "which", lambda name: "/bin/claude")

    def no_run(*a, **k):
        raise AssertionError("process must not start")

    monkeypatch.setattr(subprocess, "run", no_run)
    with pytest.raises(LLMError):
        ClaudeCodeClient(calls_path=path).complete(MSGS)
    assert json.loads(path.read_text())["count"] == c.CLAUDE_CALL_CAP


def test_the_cap_is_7000_and_the_default_model_is_sonnet():
    assert c.CLAUDE_CALL_CAP == 7000
    assert ClaudeCodeClient().model == "sonnet"


def test_claude_client_builds_expected_command_with_no_tools(claude, monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"], seen["kw"] = cmd, kw
        return Proc(_ok())

    monkeypatch.setattr(subprocess, "run", fake_run)
    res = claude.complete(MSGS)
    cmd = seen["cmd"]
    assert cmd[0] == "/bin/claude" and "-p" in cmd
    assert cmd[cmd.index("--model") + 1] == "sonnet"
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
def test_claude_sonnet_round_trip():
    res = ClaudeCodeClient().complete(MSGS, timeout_s=120)
    assert res.text


def _no_process(monkeypatch):
    def no_run(*a, **k):
        raise AssertionError("process must not start")

    monkeypatch.setattr(subprocess, "run", no_run)


@pytest.mark.parametrize(
    "content",
    [
        "{garbage",
        "{}",
        '{"count": "5", "input_tokens": 0, "output_tokens": 0, "cost_usd": 0}',
        '{"count": -1, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0}',
        '{"count": true, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0}',
        "[]",
        "",
    ],
)
def test_corrupt_counter_raises_llm_error_and_starts_no_process(claude, monkeypatch, tmp_path, content):
    path = tmp_path / "claude_calls.json"
    path.write_text(content)
    _no_process(monkeypatch)
    with pytest.raises(LLMError):
        claude.complete(MSGS)
    assert path.read_text() == content


def test_reply_mentioning_usage_limit_is_not_an_error(claude, monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda cmd, **kw: Proc(_ok("Explain what a usage limit is"))
    )
    assert claude.complete(MSGS).text == "Explain what a usage limit is"


def test_lock_that_cannot_be_acquired_raises_llm_error(claude, monkeypatch, tmp_path):
    monkeypatch.setattr(c, "LOCK_TIMEOUT_S", 0.3)
    _no_process(monkeypatch)
    with c._locked(tmp_path / "claude_calls.json", 5.0):
        with pytest.raises(LLMError):
            claude.complete(MSGS)


_WORKER = """
import sys
from pathlib import Path
from contextlens.llm import client as c
c.shutil.which = lambda name: "/bin/claude"
class P:
    stdout = '{"is_error": false, "result": "x", "usage": {}}'
    returncode = 0
    stderr = ""
c.subprocess.run = lambda cmd, **kw: P()
ok = 0
for _ in range(int(sys.argv[2])):
    try:
        c.ClaudeCodeClient(calls_path=Path(sys.argv[1]), cap=int(sys.argv[3])).complete([{"role": "user", "content": "hi"}])
        ok += 1
    except c.LLMError:
        pass
print(ok)
"""


def test_cap_holds_across_processes(tmp_path):
    import sys

    path = tmp_path / "claude_calls.json"
    path.write_text(json.dumps({"count": 90, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}))
    script = tmp_path / "worker.py"
    script.write_text(_WORKER)
    procs = [
        subprocess.Popen(
            [sys.executable, str(script), str(path), "10", "100"],
            stdout=subprocess.PIPE,
            text=True,
        )
        for _ in range(6)
    ]
    started = sum(int(p.communicate()[0]) for p in procs)
    assert started == 10
    assert json.loads(path.read_text())["count"] == 100


def test_cap_holds_across_threads(claude, monkeypatch, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "claude_calls.json"
    path.write_text(json.dumps({"count": c.CLAUDE_CALL_CAP - 10, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}))
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: Proc(_ok()))

    def one(_):
        try:
            claude.complete(MSGS)
            return 1
        except LLMError:
            return 0

    with ThreadPoolExecutor(40) as pool:
        started = sum(pool.map(one, range(40)))
    assert started == 10
    assert json.loads(path.read_text())["count"] == c.CLAUDE_CALL_CAP


def test_ollama_defaults_use_ipv4_loopback():
    import inspect

    for cls in (c.OllamaClient, c.Embedder):
        default = inspect.signature(cls.__init__).parameters["base_url"].default
        assert default == "http://127.0.0.1:11434", cls.__name__


# ---------------------------------------------------------------- one project-wide counter

def _git(root, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=root, check=True, capture_output=True)


def test_a_worktree_shares_the_counter_of_the_main_folder(tmp_path):
    main = tmp_path / "main"
    (main / ".git" / "worktrees" / "w").mkdir(parents=True)
    (main / ".git" / "worktrees" / "w" / "commondir").write_text("../..\n")
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {(main / '.git' / 'worktrees' / 'w').as_posix()}\n")
    assert c.shared_root(wt).resolve() == main.resolve()
    assert c.shared_root(main).resolve() == main.resolve()  # a plain checkout is its own root
    assert c.CLAUDE_CALLS_PATH == c.shared_root(c.REPO_ROOT) / "results" / "claude_calls.json"


def test_the_files_the_client_creates_do_not_dirty_a_git_tree(tmp_path, monkeypatch):
    from pathlib import Path

    from contextlens.eval.records import is_dirty

    (tmp_path / ".gitignore").write_text((Path(c.REPO_ROOT) / ".gitignore").read_text())
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    monkeypatch.setattr(c.shutil, "which", lambda name: "/bin/claude")

    class P:
        stdout = '{"is_error": false, "result": "x", "usage": {}}'
        returncode = 0
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: P())
    counter = tmp_path / "results" / "claude_calls.json"
    ClaudeCodeClient(calls_path=counter, cap=1).complete(MSGS)
    with pytest.raises(LLMError):
        ClaudeCodeClient(calls_path=counter, cap=1).complete(MSGS)  # the cap path locks too
    assert (counter.parent / "claude_calls.json.lock").exists()
    monkeypatch.undo()  # git runs through subprocess too
    assert not is_dirty(tmp_path)


def test_claude_reply_with_null_cost_or_usage_is_still_a_result(claude, monkeypatch):
    body = json.dumps({"is_error": False, "result": "hi", "total_cost_usd": None, "usage": None})
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: Proc(body))
    res = claude.complete(MSGS)
    assert res.text == "hi" and res.cost_usd == 0.0


def test_claude_reply_that_is_not_an_object_is_an_llm_error(claude, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: Proc("[1, 2]"))
    with pytest.raises(LLMError):
        claude.complete(MSGS)
