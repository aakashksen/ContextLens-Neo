"""Async client for the speech service (speech/proto/speech.proto), over gRPC without generated stubs.

Audio goes out as 16 kHz mono 16-bit PCM and is held only in memory.
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Protocol

from pydantic import BaseModel

METHOD = "/contextlens.speech.Speech/Transcribe"
MAX_BACKLOG_CHUNKS = 150


class SpeechUnavailable(Exception):
    """The speech service is down, stalled, or failed mid-stream."""


class Segment(BaseModel):
    text: str
    start_ms: int
    end_ms: int


class SpeechStream(Protocol):
    def send(self, pcm: bytes) -> None:
        """Queue audio without blocking. Raises SpeechUnavailable if the service is not keeping up."""
        ...

    def segments(self) -> AsyncIterator[Segment]:
        """Final segments as they arrive. Raises SpeechUnavailable if the stream fails."""
        ...

    async def close(self) -> None: ...


class SpeechService(Protocol):
    async def open(self) -> SpeechStream: ...


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _read_varint(buf: bytes, i: int) -> tuple[int, int]:
    n = shift = 0
    while True:
        b = buf[i]
        i += 1
        n |= (b & 0x7F) << shift
        if not b & 0x80:
            return n, i
        shift += 7


def encode_chunk(pcm: bytes) -> bytes:
    """AudioChunk { bytes pcm = 1; }"""
    return b"\x0a" + _varint(len(pcm)) + pcm


def decode_segment(buf: bytes) -> tuple[Segment, bool]:
    """TranscriptSegment { string text = 1; int64 start_ms = 2; int64 end_ms = 3; bool is_final = 4; }"""
    text, start, end, final = "", 0, 0, False
    i = 0
    while i < len(buf):
        tag, i = _read_varint(buf, i)
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            value, i = _read_varint(buf, i)
            if field == 2:
                start = value
            elif field == 3:
                end = value
            elif field == 4:
                final = bool(value)
        elif wire == 2:
            size, i = _read_varint(buf, i)
            if field == 1:
                text = buf[i : i + size].decode("utf-8", errors="replace")
            i += size
        else:
            raise ValueError(f"unsupported wire type {wire}")
    return Segment(text=text, start_ms=start, end_ms=end), final


class _GrpcStream:
    def __init__(self, channel, call, queue: asyncio.Queue) -> None:
        self._channel, self._call, self._queue = channel, call, queue

    def send(self, pcm: bytes) -> None:
        try:
            self._queue.put_nowait(pcm)
        except asyncio.QueueFull:
            raise SpeechUnavailable("the speech service is not keeping up") from None

    async def segments(self) -> AsyncIterator[Segment]:
        import grpc

        try:
            async for seg, final in self._call:
                if final and seg.text.strip():
                    yield seg
        except grpc.aio.AioRpcError as exc:
            raise SpeechUnavailable(f"speech service error: {exc.code().name}") from None
        raise SpeechUnavailable("the speech service closed the stream")

    async def close(self) -> None:
        self._call.cancel()
        await self._channel.close()


class GrpcSpeech:
    def __init__(self, addr: str = "localhost:50051", connect_timeout_s: float = 3.0) -> None:
        self.addr, self.connect_timeout_s = addr, connect_timeout_s

    async def open(self) -> SpeechStream:
        import grpc

        channel = grpc.aio.insecure_channel(self.addr)
        try:
            await asyncio.wait_for(channel.channel_ready(), self.connect_timeout_s)
        except BaseException as exc:  # includes being cancelled: the channel must not be left open
            await channel.close()
            if isinstance(exc, TimeoutError | asyncio.TimeoutError):
                raise SpeechUnavailable("the speech service did not answer in time") from None
            raise
        queue: asyncio.Queue = asyncio.Queue(MAX_BACKLOG_CHUNKS)

        async def chunks():
            while True:
                yield await queue.get()

        rpc = channel.stream_stream(METHOD, request_serializer=encode_chunk, response_deserializer=decode_segment)
        return _GrpcStream(channel, rpc(chunks()), queue)
