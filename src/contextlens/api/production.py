"""The backend as it runs for real: the agent team on local models, a real store, the speech service, one shared
speaker model, telemetry and the scene store. `build_services` takes every outside piece as an argument so tests
can swap in fakes; `create_production_app` passes the real ones. Live paths use local models only."""

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from contextlens.agents.consistency_check import PromptedChecker
from contextlens.agents.cascade import CascadeChecker
from contextlens.agents.graph import TeamAssistant
from contextlens.agents.precheck import prepare
from contextlens.agents.routed import RoutedTeam
from contextlens.agents.single import SingleAgent
from contextlens.agents.tools import Tools
from contextlens.api.app import create_app
from contextlens.api.stream import Services
from contextlens.llm.client import Embedder, LLMClient, OllamaClient, unload_ollama
from contextlens.llm.warm import Warmup
from contextlens.memory.extract import MemoryExtractor, MemoryStore
from contextlens.memory.store import TranscriptStore
from contextlens.memory.summary import RollingSummary
from contextlens.obs.tracing import Telemetry, TracedAssistant, serve_metrics
from contextlens.replay.runner import Assistant, Context
from contextlens.retrieval.hybrid import Retriever
from contextlens.schemas import Card, ButtonRequest
from contextlens.speech.client import GrpcSpeech, SpeechService
from contextlens.speech.diarize import SharedSpeakerModel
from contextlens.vision.local import SceneStore

MODEL = "qwen3.5:4b-q4_K_M"
OLLAMA_URL = "http://127.0.0.1:11434"
PROMPT_VERSION = "v3"
DEADLINE_S = 8.0
SPECIALIST_DEADLINE_S = 5.5  # the value the dropped sub-deadline fix was measured at; the code stays, switched off
# What the shipped team passes to TeamAssistant besides its llms, tools, prompt and deadline. Both team fixes (a search before
# the memory model call, and a specialist sub-deadline) were tried and dropped under docs/DECISIONS.md entry 64, so both are
# off; a test pins this.
TEAM_SETTINGS = {"prefetch_memory": False, "specialist_deadline_s": None}
SHUTDOWN_FLUSH_S = 5.0
EMBED_MODEL = "nomic-embed-text"
ASSISTANTS = ("team", "single", "team_shipping_changed", "team_routed", "single_plus")
DEFAULT_ASSISTANT = "team"
KEEP_ALIVE = "30m"  # Ollama's default of 5 minutes would unload the models between presses


def database_url() -> str:
    return os.environ.get("DATABASE_URL", "postgresql://contextlens:contextlens@127.0.0.1:5432/contextlens")


def assistant_kind() -> str:
    """CONTEXTLENS_ASSISTANT picks the system behind the button: `team` (the default), `single`, `team_shipping_changed`, `team_routed` or `single_plus`."""
    kind = os.environ.get("CONTEXTLENS_ASSISTANT", DEFAULT_ASSISTANT)
    if kind not in ASSISTANTS:
        raise ValueError(f"CONTEXTLENS_ASSISTANT must be one of {ASSISTANTS}, got {kind!r}")
    return kind


def local_llms() -> dict[str, LLMClient]:
    llm = OllamaClient(MODEL, OLLAMA_URL, keep_alive=KEEP_ALIVE)
    return {"memory": llm, "consistency": llm, "assistance": llm}


def local_embedder() -> Embedder:
    return Embedder(base_url=OLLAMA_URL, keep_alive=KEEP_ALIVE)


def unload_local_models() -> None:
    """Drop both models from Ollama's memory (they were loaded with a 30 minute keep_alive), so the GPU is free."""
    for model in (MODEL, EMBED_MODEL):
        unload_ollama(model, OLLAMA_URL)


def local_warmup(llm: OllamaClient, embedder: Embedder, telemetry: Telemetry | None = None) -> Warmup:
    return Warmup(
        {"chat": llm.load, "embed": lambda: embedder.embed(["warm up"])},
        on_loaded=telemetry.record_warmup if telemetry else None,
    )


def serve_metrics_if_asked(telemetry: Telemetry) -> None:
    """Off unless CONTEXTLENS_METRICS_PORT is set. The host stays on this machine unless the person names one."""
    port = os.environ.get("CONTEXTLENS_METRICS_PORT")
    if port:
        serve_metrics(telemetry, int(port), os.environ.get("CONTEXTLENS_METRICS_HOST", "127.0.0.1"))


@dataclass
class Production:
    services: Services
    traced: TracedAssistant
    telemetry: Telemetry
    scenes: SceneStore


class TeamPerPress:
    """A team bound to the conversation and frozen seq of each press; nothing carries over between presses."""

    def __init__(self, store: TranscriptStore, retriever: Retriever, memory: MemoryStore, llms: dict[str, LLMClient],
                 deadline_s: float) -> None:  # fmt: skip
        self.store, self.retriever, self.memory, self.llms, self.deadline_s = store, retriever, memory, llms, deadline_s

    memory_gap_abstains = False

    def team(self, req: ButtonRequest, ctx: Context) -> TeamAssistant:
        checker = PromptedChecker(self.llms["consistency"])
        tools = Tools(self.store, self.retriever, self.memory, ctx.as_of_seq, req.conversation_id, checker)
        return TeamAssistant(self.llms, tools, PROMPT_VERSION, self.deadline_s, memory_gap_abstains=self.memory_gap_abstains,
                             **TEAM_SETTINGS)

    def answer(self, req: ButtonRequest, ctx: Context) -> Card:
        return self.team(req, ctx).answer(req, ctx)


class _TeamChangedPerPress(TeamPerPress):
    """The arm T9.1 measures (entry 51): the team above with a card that abstains when memory looked and found nothing."""

    memory_gap_abstains = True


class _SinglePerPress(TeamPerPress):
    """The single agent as E3 measured it: the same tools, with no consistency checker."""

    def answer(self, req: ButtonRequest, ctx: Context) -> Card:
        tools = Tools(self.store, self.retriever, self.memory, ctx.as_of_seq, req.conversation_id)
        return SingleAgent(self.llms["assistance"], tools, PROMPT_VERSION, deadline_s=self.deadline_s).answer(req, ctx)


class _SinglePlusPerPress(TeamPerPress):
    """The single agent plus code before its first call: a search with the question, and the cascade checker whose verdict
    the model is shown. The checker is not a tool here."""

    def answer(self, req: ButtonRequest, ctx: Context) -> Card:
        args = (self.store, self.retriever, self.memory, ctx.as_of_seq, req.conversation_id)
        checking = Tools(*args, cascade_checker(self.llms["consistency"]))
        agent = SingleAgent(self.llms["assistance"], Tools(*args), PROMPT_VERSION, deadline_s=self.deadline_s,
                            prepare=lambda req, turns, end: prepare(checking, req, turns, end))
        return agent.answer(req, ctx)


class _RoutedPerPress(TeamPerPress):
    """`team_routed`: see agents/routed.py."""

    def answer(self, req: ButtonRequest, ctx: Context) -> Card:
        tools = Tools(self.store, self.retriever, self.memory, ctx.as_of_seq, req.conversation_id,
                      cascade_checker(self.llms["consistency"]))
        return RoutedTeam(self.llms["assistance"], tools, PROMPT_VERSION, self.deadline_s).answer(req, ctx)


PER_PRESS: dict[str, type[TeamPerPress]] = {
    "team": TeamPerPress, "single": _SinglePerPress, "team_shipping_changed": _TeamChangedPerPress,
    "team_routed": _RoutedPerPress, "single_plus": _SinglePlusPerPress,
}


def build_assistant(name: str, store: TranscriptStore, retriever: Retriever, memory: MemoryStore,
                    llms: dict[str, LLMClient], deadline_s: float = DEADLINE_S) -> Assistant:
    """The one way production and the evaluation runners build an assistant by name."""
    if name not in PER_PRESS:
        raise ValueError(f"assistant must be one of {ASSISTANTS}, got {name!r}")
    return PER_PRESS[name](store, retriever, memory, llms, deadline_s)


_TRAINED: dict[str, Any] = {}


def cascade_checker(llm: LLMClient):
    """The shipped cascade (configs/cascade.json): the trained model screens, the prompted checker confirms what it flags.
    If the trained checkpoint is missing or does not load, the prompted checker alone (one model call per press)."""
    root = Path(__file__).resolve().parents[3]
    cfg = json.loads((root / "configs" / "cascade.json").read_text(encoding="utf-8"))
    second = PromptedChecker(llm)
    path = Path(os.environ.get("CONTEXTLENS_CHECKER_DIR") or root / cfg["trained_checkpoint"])
    if "first" not in _TRAINED:
        try:
            from contextlens.consistency.model import TrainedChecker

            _TRAINED["first"] = TrainedChecker.from_checkpoint(path, strip_instructions=False)
        except Exception as exc:
            logging.getLogger(__name__).warning("trained checker not loaded (%s: %s); using the prompted checker alone", type(exc).__name__, exc)
            _TRAINED["first"] = None
    first = _TRAINED["first"]
    return second if first is None else CascadeChecker(first, second, cfg["flag_threshold"])


def build_services(
    *, store: TranscriptStore, embedder, llms: dict[str, LLMClient], background_llm: LLMClient | None,
    speech: SpeechService, make_backend=None, telemetry: Telemetry | None = None,
    deadline_s: float = DEADLINE_S, assistant: str = DEFAULT_ASSISTANT, unload: Callable[[], None] | None = None,
    **service_kw,
) -> Production:  # fmt: skip
    if assistant not in ASSISTANTS:
        raise ValueError(f"assistant must be one of {ASSISTANTS}, got {assistant!r}")
    retriever, memory = Retriever(store, embedder), MemoryStore()
    telemetry = telemetry or Telemetry()
    inner = build_assistant(assistant, store, retriever, memory, llms, deadline_s)
    traced, scenes = TracedAssistant(inner, telemetry), SceneStore()

    def forget(conversation_id: str) -> None:
        scenes.delete(conversation_id)
        traced.discard(conversation_id)

    def shutdown() -> None:
        try:
            traced.flush(SHUTDOWN_FLUSH_S)
        finally:
            if unload is not None:
                unload()

    services = Services(
        store=store, retriever=retriever, memory=memory, speech=speech,
        make_assistant=lambda conv_id, as_of: traced,
        extractor=MemoryExtractor(background_llm) if background_llm else None,
        make_summary=(lambda: RollingSummary(background_llm)) if background_llm else None,
        make_backend=make_backend, forget=forget, on_shutdown=shutdown,
        **service_kw,
    )  # fmt: skip
    return Production(services, traced, telemetry, scenes)


def create_production_app() -> FastAPI:
    from contextlens.memory.store import PostgresTranscriptStore
    from contextlens.speech.diarize import PyannoteBackend

    embedder = local_embedder()
    llms = local_llms()
    telemetry = Telemetry()
    serve_metrics_if_asked(telemetry)
    prod = build_services(
        store=PostgresTranscriptStore(database_url(), embedder), embedder=embedder, llms=llms,
        telemetry=telemetry, assistant=assistant_kind(), warmup=local_warmup(llms["memory"], embedder, telemetry),
        background_llm=llms["memory"], speech=GrpcSpeech(os.environ.get("SPEECH_ADDR", "localhost:50051")),
        make_backend=SharedSpeakerModel(PyannoteBackend), unload=unload_local_models,
    )  # fmt: skip
    return create_app(prod.services)


def main() -> None:
    import uvicorn

    uvicorn.run(create_production_app, factory=True, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
