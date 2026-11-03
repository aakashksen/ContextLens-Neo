"""Fixes found while cleaning up the service parts (api, speech, memory, retrieval, obs)."""

import asyncio
import subprocess
import sys

import pytest

from contextlens.api.stream import ConversationState, Services
from contextlens.llm.client import FakeLLM
from contextlens.memory.extract import MemoryExtractor, MemoryStore
from contextlens.memory.store import InMemoryTranscriptStore
from contextlens.memory.summary import RollingSummary
from contextlens.retrieval.hybrid import Retriever
from contextlens.schemas import Turn
from contextlens.speech.client import GrpcSpeech, SpeechUnavailable
from contextlens.speech.diarize import BYTES_PER_MS, Diarizer


class _Backend:
    def spans(self, pcm):
        return []


def test_window_start_does_not_drift_when_chunks_are_not_whole_milliseconds():
    d = Diarizer("c", _Backend())
    total = 0
    for _ in range(8000):  # 1000-byte chunks are 31.25 ms; about 250 s, so the 120 s window is trimmed throughout
        d.feed(b"\x00" * 1000)
        total += 1000
    pcm, start = d.window()
    assert start == (total - len(pcm)) // BYTES_PER_MS


def test_the_clock_keeps_exact_time_while_no_backend_is_loaded():
    d = Diarizer("c")
    for _ in range(64):
        d.feed(b"\x00" * 1000)
    d.set_backend(_Backend())
    d.feed(b"\x00" * 1000)
    pcm, start = d.window()
    assert (start, len(pcm)) == (64 * 1000 // BYTES_PER_MS, 1000)


def _state(summary_llm: FakeLLM, extractor_llm: FakeLLM) -> tuple[ConversationState, MemoryStore]:
    store, memory = InMemoryTranscriptStore(), MemoryStore()
    store.add(Turn(id="c-t0", conversation_id="c", seq=0, speaker="a", speaker_confidence=1.0,
                   text="The deadline is Monday", start_ms=0, end_ms=900, source="asr"))
    svc = Services(
        store=store, retriever=Retriever(store, None), memory=memory, make_assistant=lambda c, a: None, speech=None,
        extractor=MemoryExtractor(extractor_llm), make_summary=lambda: RollingSummary(summary_llm),
    )
    return ConversationState("c", svc), memory


def test_a_failing_summary_does_not_stop_memory_from_being_stored():
    item = {"items": [{"kind": "fact", "subject": "deadline", "statement": "Monday", "source_turn_ids": ["c-t0"]}]}
    state, memory = _state(FakeLLM([]), FakeLLM([item]))  # the summary model has nothing to say: it raises
    state._memory_step(state.epoch)
    assert [m.subject for m in memory.items("c", 10)] == ["deadline"]
    assert state.memory_upto == 0


def test_grpc_open_closes_its_channel_when_the_wait_fails_for_any_reason(monkeypatch):
    import grpc

    closed = []

    class Channel:
        async def channel_ready(self):
            raise OSError("boom")

        async def close(self):
            closed.append(True)

    monkeypatch.setattr(grpc.aio, "insecure_channel", lambda addr: Channel())
    with pytest.raises(OSError):
        asyncio.run(GrpcSpeech("x:1", 1.0).open())
    assert closed == [True]


def test_grpc_open_still_reports_a_timeout_as_unavailable(monkeypatch):
    import grpc

    class Channel:
        async def channel_ready(self):
            await asyncio.sleep(10)

        async def close(self):
            pass

    monkeypatch.setattr(grpc.aio, "insecure_channel", lambda addr: Channel())
    with pytest.raises(SpeechUnavailable):
        asyncio.run(GrpcSpeech("x:1", 0.05).open())


def test_importing_the_api_app_loads_no_model_library():
    code = (
        "import sys, contextlens.api.app;"
        "heavy = {'torch', 'transformers', 'sentence_transformers', 'faster_whisper', 'pyannote'};"
        "print(sorted(heavy & set(sys.modules)))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "[]"
