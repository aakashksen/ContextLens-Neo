"""One /ws/{conversation_id} connection, and the per-conversation state connections share."""

import asyncio
import base64
import binascii
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError

from contextlens.api.contract import (
    AudioMessage,
    CardMessage,
    StatusMessage,
    TranscriptMessage,
    client_adapter,
)
from contextlens.memory.extract import MemoryExtractor, MemoryStore
from contextlens.memory.store import TranscriptStore
from contextlens.memory.summary import RollingSummary
from contextlens.replay.runner import Assistant, Context
from contextlens.retrieval.hybrid import Retriever
from contextlens.schemas import ButtonRequest, Card, Turn
from contextlens.speech.client import Segment, SpeechService, SpeechStream, SpeechUnavailable
from contextlens.speech.diarize import BYTES_PER_MS, Diarizer, SpeakerBackend, WindowBackend

log = logging.getLogger(__name__)  # never given transcript text
ALL = 2**62
SAMPLE_RATE = 16000


@dataclass
class Services:
    store: TranscriptStore
    retriever: Retriever  # its delete() clears the store and the cached vectors
    memory: MemoryStore
    make_assistant: Callable[[str, int], Assistant]  # (conversation_id, as_of_seq); local models only
    speech: SpeechService
    extractor: MemoryExtractor | None = None
    make_summary: Callable[[], RollingSummary] | None = None
    make_backend: Callable[[], SpeakerBackend | WindowBackend | None] | None = None
    speech_timeout_s: float = 3.0
    speech_retry_s: float = 10.0
    answer_timeout_s: float = 60.0


class _FrozenStore:
    """What an assistant reads during one press: nothing after the frozen seq, and no writes."""

    def __init__(self, inner: TranscriptStore, as_of_seq: int) -> None:
        self._inner, self._limit = inner, as_of_seq

    def add(self, turn: Turn) -> None:
        raise PermissionError("read only")

    def turns(self, conv_id: str, as_of_seq: int, start_seq: int = 0) -> list[Turn]:
        return self._inner.turns(conv_id, min(as_of_seq, self._limit), start_seq)

    def relabel(self, conv_id: str, seq: int, speaker: str, confidence: float) -> None:
        raise PermissionError("read only")

    def delete(self, conv_id: str) -> None:
        raise PermissionError("read only")


class ConversationState:
    """Shared by every connection to one conversation. Writes happen under `lock` and only if the epoch is
    still the one the work started in, so work that finishes after a delete changes nothing."""

    def __init__(self, conv_id: str, svc: Services) -> None:
        self.conv_id, self.svc = conv_id, svc
        self.lock = threading.Lock()
        self.epoch = 0
        self.diarizer = Diarizer(conv_id)  # gets its backend from load_backend(), off the connection's path
        self._backend_task: asyncio.Task | None = None
        self.summary = svc.make_summary() if svc.make_summary else None
        self.memory_upto = -1
        self._memory_dirty = False
        self._memory_task: asyncio.Task | None = None
        self._speaker_dirty = False
        self._speaker_task: asyncio.Task | None = None
        existing = svc.store.turns(conv_id, ALL)
        self.last_sent_seq = existing[-1].seq if existing else -1
        self.diarizer.next_seq = self.last_sent_seq + 1
        self.diarizer.last_end_ms = existing[-1].end_ms if existing else 0

    # -- diarizer backend, loaded once in the background --
    # Until it is loaded, or if it fails, turns stay "unknown". Clients are not told: the contract has no
    # message for it, and "unknown" is already the honest label.

    def start_backend_load(self) -> None:
        if self.svc.make_backend is None or self._backend_task is not None:
            return
        self._backend_task = asyncio.get_running_loop().create_task(self._load_backend())

    async def _load_backend(self) -> None:
        try:
            backend = await asyncio.to_thread(self.svc.make_backend)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # one attempt only; the type, never the message
            log.warning("speaker model not loaded, speakers stay unknown: %s", type(exc).__name__)
            return
        if backend is not None:
            self.diarizer.set_backend(backend)

    # -- ingest --

    def ingest(self, segments: list[Segment], epoch: int, allow: Callable[[], bool]) -> list[Turn]:
        with self.lock:
            if epoch != self.epoch or not allow():
                return []
            turns = self.diarizer.assign(segments)
            for t in turns:
                self.svc.store.add(t)
            return turns

    # -- memory, in the background --

    def schedule_memory(self) -> None:
        if self.svc.extractor is None:
            return
        self._memory_dirty = True
        if self._memory_task is None or self._memory_task.done():
            self._memory_task = asyncio.get_running_loop().create_task(self._memory_loop())

    async def _memory_loop(self) -> None:
        while self._memory_dirty:
            self._memory_dirty = False
            try:
                await asyncio.to_thread(self._memory_step, self.epoch)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("memory update failed: %s", type(exc).__name__)

    def _memory_step(self, epoch: int) -> None:
        svc = self.svc
        with self.lock:
            if epoch != self.epoch:
                return
            new = svc.store.turns(self.conv_id, ALL, self.memory_upto + 1)
            existing = svc.memory.items(self.conv_id, ALL)
            summary = self.summary
        if not new:
            return
        items = svc.extractor.update(new, existing)
        if summary is not None:
            summary.update(new)
        with self.lock:
            if epoch != self.epoch:
                return
            svc.memory.add(items)
            self.memory_upto = new[-1].seq

    # -- speakers, in the background --
    # A turn is sent as "unknown". When the pipeline later names its speaker, only the stored turn changes: the
    # contract has no message for updating a sent turn, and resending it would show twice in clients that append.
    # Cards cite the stored turn, so they carry the speaker once it is known.

    def schedule_speakers(self) -> None:
        if not self.diarizer.background:
            return
        self._speaker_dirty = True
        if self._speaker_task is None or self._speaker_task.done():
            self._speaker_task = asyncio.get_running_loop().create_task(self._speaker_loop())

    async def _speaker_loop(self) -> None:
        while self._speaker_dirty:
            self._speaker_dirty = False
            try:
                await asyncio.to_thread(self._speaker_step, self.epoch)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("speaker assignment failed: %s", type(exc).__name__)

    def _speaker_step(self, epoch: int) -> None:
        pcm, start = self.diarizer.window()
        if not pcm:
            return
        spans = self.diarizer.backend.spans(pcm)
        with self.lock:
            if epoch != self.epoch:
                return  # deleted while the pipeline ran: its result is dropped
            turns = self.svc.store.turns(self.conv_id, ALL)
            for seq, speaker, confidence in self.diarizer.speaker_updates(spans, start, start + len(pcm) // BYTES_PER_MS, turns):
                self.svc.store.relabel(self.conv_id, seq, speaker, confidence)

    # -- delete --

    def wipe(self) -> None:
        with self.lock:
            self.epoch += 1
            self.svc.retriever.delete(self.conv_id)
            self.svc.store.delete(self.conv_id)
            self.svc.memory.delete(self.conv_id)
            self.diarizer.reset()
            self.summary = self.svc.make_summary() if self.svc.make_summary else None
            self.memory_upto = -1
            self.last_sent_seq = -1

    def stop_background(self) -> None:
        self._memory_dirty = self._speaker_dirty = False
        for task in (self._memory_task, self._speaker_task):
            if task is not None:
                task.cancel()


class Connection:
    def __init__(self, ws: WebSocket, conv: ConversationState) -> None:
        self.ws, self.conv, self.svc = ws, conv, conv.svc
        self.paused = False
        self.closed = False
        self.stream: SpeechStream | None = None
        self.stream_epoch = 0
        self.reader: asyncio.Task | None = None
        self.answers: set[asyncio.Task] = set()
        self.send_lock = asyncio.Lock()
        self.down_until = 0.0
        self.reported_down = False

    async def send(self, msg: BaseModel, epoch: int | None = None) -> bool:
        """Sends unless the connection is closed or `epoch` (when given) is no longer the conversation's."""
        async with self.send_lock:
            if self.closed or (epoch is not None and epoch != self.conv.epoch):
                return False
            try:
                await self.ws.send_text(msg.model_dump_json())
            except (RuntimeError, WebSocketDisconnect):
                self.closed = True
                return False
            return True

    async def error(self, detail: str, epoch: int | None = None) -> None:
        await self.send(StatusMessage(state="error", detail=detail), epoch)

    async def run(self) -> None:
        await self.ws.accept()
        await self.send(StatusMessage(state="recording"))
        try:
            while True:
                frame = await self.ws.receive()
                if frame["type"] == "websocket.disconnect":
                    return
                raw = frame.get("text")
                if raw is None:
                    await self.error("messages must be text frames")
                    continue
                try:
                    msg = client_adapter.validate_json(raw)
                except ValidationError as e:
                    await self.error(e.errors()[0]["msg"])
                    continue
                if msg.type == "audio":
                    if not self.paused:
                        await self._audio(msg)
                elif msg.type == "press":
                    await self._press(msg.request)
                elif msg.type == "pause":
                    self.paused = True
                    await self._close_stream()  # audio already sent but not yet transcribed is dropped
                    await self.send(StatusMessage(state="paused"))
                elif msg.type == "resume":
                    self.paused = False
                    await self.send(StatusMessage(state="recording"))
                elif msg.type == "delete":
                    await self._delete()
                    return
        except WebSocketDisconnect:
            return
        finally:
            self.closed = True
            await self._close_stream()
            for t in list(self.answers):
                t.cancel()

    # -- audio --

    async def _audio(self, msg: AudioMessage) -> None:
        if msg.sample_rate != SAMPLE_RATE:
            await self.error(f"sample_rate must be {SAMPLE_RATE}")
            return
        try:
            pcm = base64.b64decode(msg.data, validate=True)
        except (binascii.Error, ValueError):
            await self.error("audio data is not valid base64")
            return
        if len(pcm) % 2:
            await self.error("audio data must be 16-bit samples")
            return
        if self.stream is not None and self.stream_epoch != self.conv.epoch:
            await self._close_stream()  # a delete happened; audio from before it must not come back
        if self.stream is None and not await self._open_stream():
            return
        try:
            self.conv.diarizer.feed(pcm)
            self.stream.send(pcm)
        except SpeechUnavailable as exc:
            await self._speech_failed(str(exc))

    async def _open_stream(self) -> bool:
        if time.monotonic() < self.down_until:
            return False
        try:
            stream = await asyncio.wait_for(self.svc.speech.open(), self.svc.speech_timeout_s)
        except SpeechUnavailable as exc:
            await self._speech_failed(str(exc))
            return False
        except Exception as exc:  # TimeoutError, OSError, ...
            await self._speech_failed(type(exc).__name__)
            return False
        self.stream, self.stream_epoch, self.reported_down = stream, self.conv.epoch, False
        self.conv.diarizer.new_stream()
        self.reader = asyncio.get_running_loop().create_task(self._read(stream, self.stream_epoch))
        return True

    async def _speech_failed(self, why: str) -> None:
        await self._close_stream()
        self.down_until = time.monotonic() + self.svc.speech_retry_s
        if not self.reported_down:
            self.reported_down = True
            await self.error(f"speech service unavailable: {why}; audio is not being transcribed")

    async def _close_stream(self) -> None:
        stream, reader = self.stream, self.reader
        self.stream = self.reader = None
        if reader is not None and reader is not asyncio.current_task():
            reader.cancel()
        if stream is not None:
            try:
                await asyncio.wait_for(stream.close(), 2.0)
            except Exception:
                pass

    async def _read(self, stream: SpeechStream, epoch: int) -> None:
        try:
            async for seg in stream.segments():
                try:
                    turns = await asyncio.to_thread(
                        self.conv.ingest, [seg], epoch, lambda: not self.paused and self.stream is stream
                    )
                except Exception as exc:
                    log.warning("could not store a turn: %s", type(exc).__name__)
                    await self.error("could not store the transcript", epoch)
                    continue
                for turn in turns:
                    if await self.send(TranscriptMessage(turn=turn), epoch):
                        self.conv.last_sent_seq = max(self.conv.last_sent_seq, turn.seq)
                if turns:
                    self.conv.schedule_memory()
                    self.conv.schedule_speakers()
        except SpeechUnavailable as exc:
            if self.stream is stream:
                await self._speech_failed(str(exc))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("speech stream failed: %s", type(exc).__name__)
            if self.stream is stream:
                await self._speech_failed(type(exc).__name__)

    # -- press --

    async def _press(self, req: ButtonRequest) -> None:
        if req.conversation_id != self.conv.conv_id:
            problem = "conversation_id does not match the connection"
        elif req.as_of_seq < 0:
            problem = "as_of_seq must not be negative"
        elif req.as_of_seq > self.conv.last_sent_seq:
            problem = "as_of_seq is past the last turn sent"
        else:
            problem = None
        if problem:
            await self.error(problem)
            return
        task = asyncio.get_running_loop().create_task(self._answer(req, self.conv.epoch))
        self.answers.add(task)
        task.add_done_callback(self.answers.discard)

    def _compute_card(self, req: ButtonRequest) -> Card:
        svc, as_of = self.svc, req.as_of_seq  # frozen here; turns that arrive later are out of reach
        frozen = _FrozenStore(svc.store, as_of)
        card = svc.make_assistant(self.conv.conv_id, as_of).answer(req, Context(store=frozen, as_of_seq=as_of))
        known = {t.id for t in frozen.turns(self.conv.conv_id, as_of)}
        if any(e.turn_id not in known for f in card.findings for e in f.evidence):
            raise ValueError("card cites a turn outside the frozen range")
        return card

    async def _answer(self, req: ButtonRequest, epoch: int) -> None:
        try:
            card = await asyncio.wait_for(asyncio.to_thread(self._compute_card, req), self.svc.answer_timeout_s)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("could not answer: %s", type(exc).__name__)
            await self.error("could not answer", epoch)
            return
        await self.send(CardMessage(card=card), epoch)

    # -- delete --

    async def _delete(self) -> None:
        for t in list(self.answers):
            t.cancel()
        await self._close_stream()
        self.conv.stop_background()
        await asyncio.to_thread(self.conv.wipe)
        await self.send(StatusMessage(state="deleted"))
        self.closed = True
        try:
            await self.ws.close()
        except RuntimeError:
            pass
