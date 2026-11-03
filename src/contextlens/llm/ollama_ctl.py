"""Load and unload local Ollama models, so a cold start is timed apart from the work and GPU memory is given back."""

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable

from contextlens.llm.client import LLMError

BASE_URL = "http://127.0.0.1:11434"
LOAD_TIMEOUT_S = 180.0
KEEP_LOADED = "30m"

Request = Callable[[str, str, dict | None, float], dict]


def _http(base_url: str) -> Request:
    def call(method: str, path: str, payload: dict | None, timeout_s: float) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(base_url + path, data, {"Content-Type": "application/json"}, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read() or b"{}")
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as err:
            raise LLMError(f"request to {base_url}{path} failed: {err}") from err

    return call


def full_name(model: str) -> str:
    return model if ":" in model else f"{model}:latest"


class OllamaCtl:
    def __init__(self, base_url: str = BASE_URL, request: Request | None = None) -> None:
        self._call = request or _http(base_url.rstrip("/"))

    def installed(self) -> set[str]:
        return {full_name(m["name"]) for m in self._call("GET", "/api/tags", None, 10.0).get("models", [])}

    def loaded(self) -> set[str]:
        return {full_name(m["name"]) for m in self._call("GET", "/api/ps", None, 10.0).get("models", [])}

    def load(self, model: str, *, embedding: bool = False) -> float:
        """Load the model and keep it loaded; the seconds this took are its cold start."""
        t0 = time.perf_counter()
        if embedding:
            self._call("POST", "/api/embed", {"model": model, "input": ["ping"], "keep_alive": KEEP_LOADED}, LOAD_TIMEOUT_S)
        else:
            self._call("POST", "/api/generate", {"model": model, "keep_alive": KEEP_LOADED, "stream": False}, LOAD_TIMEOUT_S)
        return time.perf_counter() - t0

    def unload(self, model: str, *, embedding: bool = False) -> None:
        """keep_alive 0 frees the model's GPU memory right away."""
        if embedding:
            self._call("POST", "/api/embed", {"model": model, "input": ["ping"], "keep_alive": 0}, 30.0)
        else:
            self._call("POST", "/api/generate", {"model": model, "keep_alive": 0, "stream": False}, 30.0)
