"""Contract of the C++ speech service (T8.1).

The offline test checks the proto file. The real-marked tests talk to a running service
(`docker compose up -d speech`, port 50051, or SPEECH_ADDR) with a synthetic WAV.
"""

import importlib
import os
import re
import sys
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PROTO = ROOT / "speech" / "proto" / "speech.proto"
WAV = ROOT / "tests" / "fixtures" / "speech" / "phrase.wav"
CHUNK_BYTES = 6400  # 200 ms of 16 kHz mono 16-bit audio


def test_proto_declares_the_streaming_contract():
    text = PROTO.read_text()
    assert re.search(
        r"rpc\s+Transcribe\(\s*stream\s+AudioChunk\s*\)\s*returns\s*\(\s*stream\s+TranscriptSegment\s*\)", text
    )
    body = re.search(r"message\s+TranscriptSegment\s*\{(.*?)\}", text, re.S).group(1)
    fields = re.findall(r"(\w+)\s+(\w+)\s*=\s*\d+;", body)
    assert fields == [("string", "text"), ("int64", "start_ms"), ("int64", "end_ms"), ("bool", "is_final")]


def _read_wav() -> bytes:
    with wave.open(str(WAV), "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        return w.readframes(w.getnframes())


def _stubs(tmp_path):
    from grpc_tools import protoc

    out = tmp_path / "gen"
    out.mkdir()
    rc = protoc.main(
        ["protoc", f"-I{PROTO.parent}", f"--python_out={out}", f"--grpc_python_out={out}", str(PROTO)]
    )
    assert rc == 0
    sys.path.insert(0, str(out))
    try:
        return importlib.import_module("speech_pb2"), importlib.import_module("speech_pb2_grpc")
    finally:
        sys.path.remove(str(out))


def _transcribe(tmp_path, pcm: bytes, addr: str | None = None):
    import grpc

    pb, pb_grpc = _stubs(tmp_path)
    addr = addr or os.environ.get("SPEECH_ADDR", "localhost:50051")
    with grpc.insecure_channel(addr) as channel:
        grpc.channel_ready_future(channel).result(timeout=30)
        call = pb_grpc.SpeechStub(channel).Transcribe(
            iter([pb.AudioChunk(pcm=pcm[i : i + CHUNK_BYTES]) for i in range(0, len(pcm), CHUNK_BYTES)]),
            timeout=120,
        )
        segments = list(call)
        meta = dict(call.initial_metadata())
    return segments, meta


@pytest.mark.real
def test_fixture_wav_returns_the_expected_phrase(tmp_path):
    segments, meta = _transcribe(tmp_path, _read_wav())
    assert segments, "no segments returned"
    assert all(s.is_final for s in segments)
    assert all(0 <= s.start_ms < s.end_ms for s in segments)
    text = " ".join(s.text for s in segments).lower()
    assert "budget review" in text
    assert "thursday" in text
    assert meta["x-compute-backend"] == "cpu" or meta["x-compute-backend"].startswith("gpu:")


@pytest.mark.real
def test_silence_returns_no_segments(tmp_path):
    segments, _ = _transcribe(tmp_path, bytes(16000 * 2 * 3))
    assert segments == []


@pytest.mark.real
@pytest.mark.gpu
def test_gpu_service_runs_on_the_gpu_and_transcribes(tmp_path):
    # `docker compose --profile gpu up -d speech-gpu` (host port 50052). Fails if it fell back to CPU.
    addr = os.environ.get("SPEECH_GPU_ADDR", "localhost:50052")
    segments, meta = _transcribe(tmp_path, _read_wav(), addr)
    assert meta["x-compute-backend"].startswith("gpu:"), meta["x-compute-backend"]
    text = " ".join(s.text for s in segments).lower()
    assert "budget review" in text
    assert "thursday" in text
