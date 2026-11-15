"""Model clients: Ollama, an OpenAI-compatible server (vLLM), Claude Code, and a fake for tests."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ValidationError

REPO_ROOT = Path(__file__).resolve().parents[3]


def shared_root(repo_root: Path) -> Path:
    """The main folder: the repo root itself, or, in a linked worktree, the folder that holds the common .git."""
    dot_git = repo_root / ".git"
    if not dot_git.is_file():
        return repo_root
    try:
        gitdir = Path(dot_git.read_text().split(":", 1)[1].strip())
        if not gitdir.is_absolute():
            gitdir = repo_root / gitdir
        return (gitdir / (gitdir / "commondir").read_text().strip()).resolve().parent
    except (OSError, IndexError):
        return repo_root


CLAUDE_CALLS_PATH = shared_root(REPO_ROOT) / "results" / "claude_calls.json"  # one counter for every worktree
CLAUDE_CALL_CAP = 7000  # a runaway guard, not a budget (raised from 5,000 by the person, 2026-11-05)


_caller = threading.local()


def stop_retries_when(stopped: threading.Event) -> None:
    """On this thread, once `stopped` is set, `complete` makes no further attempt.

    A caller that gave up waiting sets it, so an abandoned call does not start a paid retry."""
    _caller.stopped = stopped


class LLMError(Exception):
    def __init__(self, *args: object, results: list[LLMResult] | None = None) -> None:
        super().__init__(*args)
        self.results = results or []  # model replies received before the failure


class LLMResult(BaseModel):
    text: str
    parsed: BaseModel | None = None
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0


class LLMClient(Protocol):
    def complete(
        self,
        messages: list[dict],
        *,
        schema: type[BaseModel] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout_s: float = 10.0,
    ) -> LLMResult: ...


def _parse(text: str, schema: type[BaseModel]) -> BaseModel:
    body = text.strip()
    if body.startswith("```"):
        body = body.strip("`").removeprefix("json").strip()
    return schema.model_validate_json(body)


class _Base:
    """Shared schema validation: one retry that shows the model its error."""

    model = ""

    def _raw(self, messages, schema, max_tokens, temperature, timeout_s) -> LLMResult:
        raise NotImplementedError

    def complete(
        self,
        messages: list[dict],
        *,
        schema: type[BaseModel] | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout_s: float = 10.0,
    ) -> LLMResult:
        res = self._raw(messages, schema, max_tokens, temperature, timeout_s)
        if schema is None:
            return res
        try:
            res.parsed = _parse(res.text, schema)
            return res
        except (ValidationError, ValueError) as err:
            note = (
                f"That was not valid for the required JSON schema: {err}\n"
                "Reply again with only valid JSON."
            )
        stopped = getattr(_caller, "stopped", None)
        if stopped is not None and stopped.is_set():
            raise LLMError("no retry: the caller stopped waiting", results=[res])
        retry = [
            *messages,
            {"role": "assistant", "content": res.text},
            {"role": "user", "content": note},
        ]
        try:
            res2 = self._raw(retry, schema, max_tokens, temperature, timeout_s)
        except LLMError as err:
            err.results = [res]
            raise
        try:
            res2.parsed = _parse(res2.text, schema)
        except (ValidationError, ValueError) as err:
            raise LLMError(f"output failed schema validation twice: {err}", results=[res, res2]) from err
        res2.prompt_tokens += res.prompt_tokens  # both attempts were paid for
        res2.completion_tokens += res.completion_tokens
        res2.cost_usd += res.cost_usd
        res2.latency_ms += res.latency_ms
        return res2


class FakeLLM(_Base):
    def __init__(self, responses: list[str | BaseModel | dict]):
        self._responses = list(responses)
        self.calls: list[list[dict]] = []
        self.model = "fake"

    def _raw(self, messages, schema, max_tokens, temperature, timeout_s) -> LLMResult:
        self.calls.append(messages)
        if not self._responses:
            raise LLMError("FakeLLM has no responses left")
        r = self._responses.pop(0)
        if isinstance(r, BaseModel):
            r = r.model_dump_json()
        elif isinstance(r, dict):
            r = json.dumps(r)
        return LLMResult(text=r, model=self.model)


def _post_json(url: str, payload: dict, timeout_s: float) -> dict:
    req = urllib.request.Request(
        url, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as err:
        raise LLMError(f"request to {url} failed: {err}") from err


def unload_ollama(model: str, base_url: str = "http://127.0.0.1:11434") -> bool:
    """Ask Ollama to drop `model` from memory (keep_alive 0). False when the server cannot be reached; never raises."""
    try:
        _post_json(f"{base_url.rstrip('/')}/api/generate", {"model": model, "keep_alive": 0}, 30.0)
    except LLMError:
        return False
    return True


class OllamaClient(_Base):
    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434", keep_alive: str | None = None,
                 num_ctx: int | None = None):
        self.model, self.base_url, self.keep_alive, self.num_ctx = model, base_url.rstrip("/"), keep_alive, num_ctx

    def load(self, timeout_s: float = 300.0) -> None:
        """Loads the model into memory without a prompt (an empty message list does that)."""
        payload: dict = {"model": self.model, "messages": [], "stream": False}
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        _post_json(f"{self.base_url}/api/chat", payload, timeout_s)

    def _raw(self, messages, schema, max_tokens, temperature, timeout_s) -> LLMResult:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        if self.num_ctx:
            payload["options"]["num_ctx"] = self.num_ctx
        if schema is not None:
            payload["format"] = schema.model_json_schema()
        t0 = time.perf_counter()
        data = _post_json(f"{self.base_url}/api/chat", payload, timeout_s)
        try:
            text = data["message"]["content"]
        except (KeyError, TypeError) as err:
            raise LLMError(f"unexpected Ollama reply: {data}") from err
        return LLMResult(
            text=text,
            model=self.model,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


class OpenAICompatClient(_Base):
    def __init__(self, base_url: str, model: str):
        self.base_url, self.model = base_url.rstrip("/"), model

    def _raw(self, messages, schema, max_tokens, temperature, timeout_s) -> LLMResult:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
            }
        t0 = time.perf_counter()
        data = _post_json(f"{self.base_url}/chat/completions", payload, timeout_s)
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as err:
            raise LLMError(f"unexpected reply: {data}") from err
        usage = data.get("usage") or {}
        return LLMResult(
            text=text,
            model=self.model,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


class Embedder:
    def __init__(
        self, model: str = "nomic-embed-text", base_url: str = "http://127.0.0.1:11434", keep_alive: str | None = None
    ):
        self.model, self.base_url, self.keep_alive = model, base_url.rstrip("/"), keep_alive

    def embed(self, texts: list[str]) -> list[list[float]]:
        payload: dict = {"model": self.model, "input": texts}
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        data = _post_json(f"{self.base_url}/api/embed", payload, 60.0)
        vectors = data.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise LLMError(f"unexpected embedding reply: {str(data)[:200]}")
        return vectors


LOCK_TIMEOUT_S = 30.0
_COUNTER_KEYS = ("count", "input_tokens", "output_tokens", "cost_usd")


def _try_lock(fh) -> bool:
    try:
        if os.name == "nt":
            import msvcrt

            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock(fh) -> None:
    if os.name == "nt":
        import msvcrt

        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def _locked(path: Path, timeout_s: float):
    """Exclusive lock across threads and processes; the OS drops it if the holder dies."""
    lock_path = path.with_name(path.name + ".lock")
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(lock_path, "a+b")  # noqa: SIM115
    except OSError as err:
        raise LLMError(f"cannot open lock file {lock_path}: {err}") from err
    try:
        deadline = time.monotonic() + timeout_s
        while not _try_lock(fh):
            if time.monotonic() >= deadline:
                raise LLMError(f"could not lock {lock_path} within {timeout_s}s")
            time.sleep(0.01)
        try:
            yield
        finally:
            _unlock(fh)
    finally:
        fh.close()


def _load_calls(path: Path) -> dict:
    if not path.exists():
        return {"count": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as err:
        raise LLMError(f"call counter {path} is unreadable: {err}") from err
    if not isinstance(data, dict):
        raise LLMError(f"call counter {path} is not an object")
    for key in _COUNTER_KEYS:
        v = data.get(key)
        if isinstance(v, bool) or not isinstance(v, int | float) or v < 0:
            raise LLMError(f"call counter {path} has an invalid {key!r}")
    if not isinstance(data["count"], int):
        raise LLMError(f"call counter {path} has a non-integer count")
    return data


def _save_calls(path: Path, calls: dict) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(calls, indent=2))
        os.replace(tmp, path)
    except OSError as err:
        raise LLMError(f"cannot write call counter {path}: {err}") from err


class ClaudeCodeClient(_Base):
    def __init__(
        self,
        model: str = "sonnet",
        calls_path: Path = CLAUDE_CALLS_PATH,
        cap: int = CLAUDE_CALL_CAP,
    ):
        self.model, self.calls_path, self.cap = model, Path(calls_path), cap

    def _raw(self, messages, schema, max_tokens, temperature, timeout_s) -> LLMResult:
        exe = shutil.which("claude")
        if exe is None:
            raise LLMError("claude command not found")
        with _locked(self.calls_path, LOCK_TIMEOUT_S):
            calls = _load_calls(self.calls_path)
            if calls["count"] >= self.cap:
                raise LLMError(f"Claude Code call cap of {self.cap} reached; only the person can raise it")
            calls["count"] += 1  # reserved before the process starts; a started call counts even if it fails
            _save_calls(self.calls_path, calls)
        prompt = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)
        if schema is not None:
            prompt += "\n\nReply with only JSON matching this schema:\n" + json.dumps(
                schema.model_json_schema()
            )
        cmd = [
            exe,
            "-p",
            "--model", self.model,
            "--output-format", "json",
            "--tools", "",
            "--setting-sources", "",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--max-turns", "1",
        ]  # fmt: skip
        t0 = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                proc = subprocess.run(
                    cmd,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=timeout_s,
                    cwd=tmp,
                )
        except (subprocess.TimeoutExpired, OSError) as err:
            raise LLMError(f"claude call failed: {err}") from err
        if proc.returncode != 0:
            raise LLMError(f"claude exited {proc.returncode}: {(proc.stderr or proc.stdout)[:300]}")
        try:
            data = json.loads(proc.stdout)
        except ValueError as err:
            raise LLMError(f"claude output was not JSON: {proc.stdout[:200]}") from err
        if not isinstance(data, dict):
            raise LLMError(f"claude output was not an object: {proc.stdout[:200]}")
        text = str(data.get("result", ""))
        if data.get("is_error"):
            raise LLMError(f"claude reported an error: {text[:300]}")
        usage = data.get("usage") or {}
        res = LLMResult(
            text=text,
            model=self.model,
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            latency_ms=(time.perf_counter() - t0) * 1000,
            cost_usd=float(data.get("total_cost_usd") or 0.0),
        )
        with _locked(self.calls_path, LOCK_TIMEOUT_S):
            calls = _load_calls(self.calls_path)
            calls["input_tokens"] += res.prompt_tokens
            calls["output_tokens"] += res.completion_tokens
            calls["cost_usd"] += res.cost_usd
            _save_calls(self.calls_path, calls)
        return res
