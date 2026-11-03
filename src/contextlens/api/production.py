"""The backend as it runs for real: the agent team on local models, a real store, the speech service, one shared
speaker model, telemetry and the scene store. `build_services` takes every outside piece as an argument so tests
can swap in fakes; `create_production_app` passes the real ones. Live paths use local models only."""

import os
from dataclasses import dataclass

from fastapi import FastAPI

from contextlens.agents.consistency_check import PromptedChecker
from contextlens.agents.graph import TeamAssistant
from contextlens.agents.tools import Tools
from contextlens.api.app import create_app
from contextlens.api.stream import Services
from contextlens.llm.client import Embedder, LLMClient, OllamaClient
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
PROMPT_VERSION = "v1"
DEADLINE_S = 8.0
SHUTDOWN_FLUSH_S = 5.0
KEEP_ALIVE = "30m"  # Ollama's default of 5 minutes would unload the models between presses


def database_url() -> str:
    return os.environ.get("DATABASE_URL", "postgresql://contextlens:contextlens@127.0.0.1:5432/contextlens")


def local_llms() -> dict[str, LLMClient]:
    llm = OllamaClient(MODEL, OLLAMA_URL, keep_alive=KEEP_ALIVE)
    return {"memory": llm, "consistency": llm, "assistance": llm}


def local_embedder() -> Embedder:
    return Embedder(base_url=OLLAMA_URL, keep_alive=KEEP_ALIVE)


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


class _TeamPerPress:
    """A team bound to the conversation and frozen seq of each press; nothing carries over between presses."""

    def __init__(self, store: TranscriptStore, retriever: Retriever, memory: MemoryStore, llms: dict[str, LLMClient],
                 deadline_s: float) -> None:  # fmt: skip
        self.store, self.retriever, self.memory, self.llms, self.deadline_s = store, retriever, memory, llms, deadline_s

    def answer(self, req: ButtonRequest, ctx: Context) -> Card:
        checker = PromptedChecker(self.llms["consistency"])
        tools = Tools(self.store, self.retriever, self.memory, ctx.as_of_seq, req.conversation_id, checker)
        return TeamAssistant(self.llms, tools, PROMPT_VERSION, self.deadline_s).answer(req, ctx)


def build_services(
    *, store: TranscriptStore, embedder, llms: dict[str, LLMClient], background_llm: LLMClient | None,
    speech: SpeechService, make_backend=None, telemetry: Telemetry | None = None,
    deadline_s: float = DEADLINE_S, **service_kw,
) -> Production:  # fmt: skip
    retriever, memory = Retriever(store, embedder), MemoryStore()
    telemetry = telemetry or Telemetry()
    inner: Assistant = _TeamPerPress(store, retriever, memory, llms, deadline_s)
    traced, scenes = TracedAssistant(inner, telemetry), SceneStore()

    def forget(conversation_id: str) -> None:
        scenes.delete(conversation_id)
        traced.discard(conversation_id)

    services = Services(
        store=store, retriever=retriever, memory=memory, speech=speech,
        make_assistant=lambda conv_id, as_of: traced,
        extractor=MemoryExtractor(background_llm) if background_llm else None,
        make_summary=(lambda: RollingSummary(background_llm)) if background_llm else None,
        make_backend=make_backend, forget=forget, on_shutdown=lambda: traced.flush(SHUTDOWN_FLUSH_S),
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
        telemetry=telemetry, warmup=local_warmup(llms["memory"], embedder, telemetry),
        background_llm=llms["memory"], speech=GrpcSpeech(os.environ.get("SPEECH_ADDR", "localhost:50051")),
        make_backend=SharedSpeakerModel(PyannoteBackend),
    )  # fmt: skip
    return create_app(prod.services)


def main() -> None:
    import uvicorn

    uvicorn.run(create_production_app, factory=True, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
