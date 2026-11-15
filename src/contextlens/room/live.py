"""Live mode: the meeting audio goes through the real stream API (/ws/{conversation_id}), which sends it to the real
speech service and diarizer, stores the turns and extracts memory in the background. This module is the client side
of that: it feeds audio at the meeting's own pace, turns what comes back into an event log the page polls, watches the
growing transcript for contradictions and updates with the cascade checker, and answers presses against the same store.

Every outside piece is passed in (`LiveRig`), so tests run the whole thing with fakes."""

import asyncio
import base64
import json
import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from contextlens.api.contract import server_adapter
from contextlens.replay.runner import Context
from contextlens.room.meetings import wav_chunks
from contextlens.room.replay import memory_json, turn_json
from contextlens.room.trace import build_trace
from contextlens.schemas import Card, MemoryItem, Turn

CHUNK_MS = 500
ALL = 2**62
MIN_FLAG_WORDS = 5  # a turn shorter than this is not worth a checker call
FLAG_BACKLOG = 2  # unchecked turns beyond this many are skipped, newest first, so flags keep up with the meeting
GPU = threading.Lock()  # one local GPU: a press waits its turn; the flag watcher only runs when nothing else does


class Transport(Protocol):
    """One connection to /ws/{conversation_id}. Frames are the contract's JSON text."""

    async def send(self, text: str) -> None: ...

    def messages(self) -> AsyncIterator[str]: ...

    async def close(self) -> None: ...


@dataclass
class LiveRig:
    connect: Callable[[str], Awaitable[Transport]]
    turns: Callable[[str], list[Turn]]  # the stored turns of a conversation
    memory: Callable[[str], list[MemoryItem]]
    check_flag: Callable[[str, Turn], dict | None]  # blocking; the cascade checker on one new turn, None if not flagged
    answer: Callable[[str, str, int, str | None], tuple[Card, Context]]  # (assistant, conversation, as_of_seq, question)


class EventLog:
    def __init__(self) -> None:
        self._events: list[dict] = []
        self._lock = threading.Lock()

    def add(self, event: dict) -> None:
        with self._lock:
            self._events.append({**event, "n": len(self._events)})

    def since(self, n: int) -> tuple[list[dict], int]:
        with self._lock:
            return self._events[n:], len(self._events)


@dataclass
class LiveSession:
    sid: str
    meeting_id: str
    wav: Path
    duration_ms: int
    rig: LiveRig
    pace: Callable[[float], Awaitable[None]] = asyncio.sleep
    log: EventLog = field(default_factory=EventLog)
    offset_ms: int = 0  # where in the meeting this conversation's audio started
    fed_ms: int = 0  # audio sent since then
    last_lag_ms: int | None = None
    _gen: int = 0
    _conv: str | None = None
    _transport: Transport | None = None
    _feeder: asyncio.Task | None = None
    _reader: asyncio.Task | None = None
    _watchers: set = field(default_factory=set)
    _seen_memory: set = field(default_factory=set)
    _checked: set = field(default_factory=set)
    _flagging: bool = False

    # -- control --

    @property
    def position_ms(self) -> int:
        return min(self.offset_ms + self.fed_ms, self.duration_ms)

    async def play(self, position_ms: int) -> None:
        if self._transport is not None and abs(position_ms - self.position_ms) > 2000:
            await self._reset(position_ms)
        if self._transport is None:
            await self._open(position_ms)
        if self._feeder is None or self._feeder.done():
            self._feeder = asyncio.get_running_loop().create_task(self._feed(self._gen))
        self.log.add({"type": "state", "playing": True})

    async def pause(self) -> None:
        await self._stop_feed()
        self.log.add({"type": "state", "playing": False})

    async def seek(self, position_ms: int, playing: bool) -> None:
        await self._reset(position_ms)
        if playing:
            await self.play(position_ms)

    async def close(self) -> None:
        await self._stop_feed()
        await self._close_transport(delete=True)

    # -- plumbing --

    async def _open(self, position_ms: int) -> None:
        self._gen += 1
        self.offset_ms, self.fed_ms = max(0, min(position_ms, self.duration_ms)), 0
        self._conv = f"room-{self.sid}-{self._gen}"
        self._seen_memory, self._checked, self.last_lag_ms = set(), set(), None
        try:
            self._transport = await self.rig.connect(self._conv)
        except Exception as exc:  # the type, never the message
            self.log.add({"type": "status", "state": "error", "detail": f"could not reach the stream API ({type(exc).__name__})"})
            raise
        loop = asyncio.get_running_loop()
        self._reader = loop.create_task(self._read(self._transport, self._gen))
        self.log.add({"type": "reset", "position_ms": self.offset_ms})

    async def _stop_feed(self) -> None:
        feeder, self._feeder = self._feeder, None
        if feeder is not None and feeder is not asyncio.current_task():
            feeder.cancel()
            await asyncio.gather(feeder, return_exceptions=True)

    async def _close_transport(self, delete: bool) -> None:
        transport, reader, self._transport, self._reader = self._transport, self._reader, None, None
        for w in list(self._watchers):
            w.cancel()
        if reader is not None:
            reader.cancel()
        if transport is not None:
            if delete:  # the stream API's own delete: the transcript, memory and vectors go with it
                try:
                    await asyncio.wait_for(transport.send(json.dumps({"type": "delete"})), 2.0)
                except Exception:
                    pass
            try:
                await asyncio.wait_for(transport.close(), 2.0)
            except Exception:
                pass

    async def _reset(self, position_ms: int) -> None:
        await self._stop_feed()
        await self._close_transport(delete=True)
        await self._open(position_ms)

    async def _feed(self, gen: int) -> None:
        transport = self._transport
        start = self.offset_ms + self.fed_ms
        for pcm in wav_chunks(self.wav, start, CHUNK_MS):
            if gen != self._gen or transport is None:
                return
            msg = json.dumps({"type": "audio", "data": base64.b64encode(pcm).decode("ascii"), "sample_rate": 16000})
            try:
                await transport.send(msg)
            except Exception as exc:
                self.log.add({"type": "status", "state": "error", "detail": f"could not send audio ({type(exc).__name__})"})
                return
            self.fed_ms += len(pcm) * 1000 // 32000
            await self.pace(CHUNK_MS / 1000)
        if gen == self._gen:
            await asyncio.sleep(0)
            self.log.add({"type": "ended"})

    async def _read(self, transport: Transport, gen: int) -> None:
        try:
            async for raw in transport.messages():
                if gen != self._gen:
                    return
                try:
                    msg = server_adapter.validate_json(raw)
                except Exception:
                    continue
                if msg.type in ("transcript", "turn_update"):
                    t = msg.turn
                    lag = max(0, self.fed_ms - (t.end_ms)) if msg.type == "transcript" else None
                    if lag is not None:
                        self.last_lag_ms = lag
                    self.log.add({"type": "turn" if msg.type == "transcript" else "turn_update",
                                  "turn": turn_json(t, self.offset_ms), "speech_lag_ms": lag})
                elif msg.type == "status":
                    self.log.add({"type": "status", "state": msg.state, "detail": msg.detail})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.log.add({"type": "status", "state": "error", "detail": f"stream closed ({type(exc).__name__})"})

    # -- what the page polls for --

    async def poll(self) -> None:
        """One look at the stored state: new memory items, and the checker on a new turn. Called about once a second."""
        if self._conv is None:
            return
        conv, gen = self._conv, self._gen
        now = self.position_ms
        for m in self.rig.memory(conv):
            if m.id not in self._seen_memory and gen == self._gen:
                self._seen_memory.add(m.id)
                self.log.add({"type": "memory", "item": memory_json(m, now)})
        if self._flagging:
            return
        todo = [t for t in self.rig.turns(conv) if t.id not in self._checked and len(t.text.split()) >= MIN_FLAG_WORDS]
        for t in todo[:-FLAG_BACKLOG] if len(todo) > FLAG_BACKLOG else []:
            self._checked.add(t.id)  # too far behind: skip it, keep up
        todo = todo[-FLAG_BACKLOG:]
        if not todo or not GPU.acquire(blocking=False):
            return
        GPU.release()
        self._flagging = True
        task = asyncio.get_running_loop().create_task(self._flag(conv, gen, todo[0]))
        self._watchers.add(task)
        task.add_done_callback(self._watchers.discard)

    async def _flag(self, conv: str, gen: int, turn: Turn) -> None:
        try:
            self._checked.add(turn.id)
            result = await asyncio.to_thread(self._locked_check, conv, turn)
            if result and gen == self._gen:
                self.log.add({"type": "flag", "flag": {**result, "turn_id": turn.id, "at_ms": turn.end_ms + self.offset_ms}})
        except Exception as exc:
            self.log.add({"type": "status", "state": "error", "detail": f"checker failed ({type(exc).__name__})"})
        finally:
            self._flagging = False

    def _locked_check(self, conv: str, turn: Turn) -> dict | None:
        if not GPU.acquire(blocking=False):
            self._checked.discard(turn.id)  # a press has the GPU: try this turn again later
            return None
        try:
            return self.rig.check_flag(conv, turn)
        finally:
            GPU.release()

    # -- press --

    def press(self, names: list[str], question: str | None) -> dict:
        """Cards from the named assistants over what the stream API has stored so far. The seq is frozen first."""
        conv = self._conv
        turns = self.rig.turns(conv) if conv else []
        as_of = turns[-1].seq if turns else -1
        if as_of < 0:
            return {"as_of_seq": -1, "position_ms": self.position_ms, "results": [], "error": "nothing has been transcribed yet"}
        results = []
        with GPU:
            for name in names:
                card, ctx = self.rig.answer(name, conv, as_of, question)
                results.append({"assistant": name, "card": card.model_dump(mode="json"),
                                "trace": build_trace(name, card, ctx.stage_ms, ctx.drops, speech_ms=self.last_lag_ms,
                                                     speech_note="real speech service and diarizer, caption delay")})
        return {"as_of_seq": as_of, "position_ms": self.position_ms, "results": results}


async def poll_forever(session: LiveSession, every_s: float = 1.0) -> None:
    while True:
        try:
            await session.poll()
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        await asyncio.sleep(every_s)
