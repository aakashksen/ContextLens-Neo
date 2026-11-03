"""E2 runner: context strategies on dev, with a separate local judge and a capped cloud comparison. No model is called for real."""
import pytest

from contextlens.llm.client import LLMError
from contextlens.llm.ollama_ctl import OllamaCtl


class Wire:
    """Stands in for Ollama's HTTP API and records every request."""

    def __init__(self, loaded=(), installed=("qwen:1",)):
        self.calls: list[tuple[str, str, dict | None]] = []
        self.loaded, self.installed = list(loaded), list(installed)

    def __call__(self, method, path, payload, timeout_s):
        self.calls.append((method, path, payload))
        if path == "/api/ps":
            return {"models": [{"name": n} for n in self.loaded]}
        if path == "/api/tags":
            return {"models": [{"name": n} for n in self.installed]}
        return {}


def test_unload_sends_keep_alive_zero_for_a_chat_model_and_an_embedder() -> None:
    wire = Wire()
    ctl = OllamaCtl(request=wire)
    ctl.unload("qwen:1")
    ctl.unload("nomic-embed-text", embedding=True)
    assert wire.calls[0][1:] == ("/api/generate", {"model": "qwen:1", "keep_alive": 0, "stream": False})
    assert wire.calls[1][1] == "/api/embed" and wire.calls[1][2]["keep_alive"] == 0


def test_load_keeps_the_model_and_returns_the_cold_start_seconds() -> None:
    wire = Wire()
    seconds = OllamaCtl(request=wire).load("qwen:1")
    assert seconds >= 0 and wire.calls[0][2]["keep_alive"] not in (0, "0")


def test_installed_and_loaded_names_get_a_tag() -> None:
    ctl = OllamaCtl(request=Wire(loaded=["nomic-embed-text"], installed=["nomic-embed-text:latest", "qwen:1"]))
    assert ctl.installed() == {"nomic-embed-text:latest", "qwen:1"} and ctl.loaded() == {"nomic-embed-text:latest"}


def test_an_unreachable_server_raises_llm_error() -> None:
    with pytest.raises(LLMError):
        OllamaCtl(base_url="http://127.0.0.1:9").installed()
