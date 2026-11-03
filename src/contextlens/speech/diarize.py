"""Speaker labels for transcribed segments.

Chosen option: NVIDIA Sortformer (`diar_streaming_sortformer_4spk-v2`, CC BY 4.0, ungated, streaming; approved in
docs/datasets.md) over pyannote 3.1, which is gated behind a Hugging Face token. No model adapter is included yet:
the backend is a protocol, and without one every turn gets the speaker "unknown" at confidence 0.0.
"""

import threading
from typing import Protocol

from contextlens.schemas import Turn
from contextlens.speech.client import Segment

SAMPLE_RATE = 16000
BYTES_PER_MS = SAMPLE_RATE * 2 // 1000
KEEP_MS = 120_000


class SpeakerBackend(Protocol):
    def label(self, pcm: bytes, start_ms: int, end_ms: int) -> tuple[str, float]:
        """Speaker label and confidence in [0, 1] for this span of audio (16 kHz mono 16-bit PCM)."""
        ...


class Diarizer:
    """One per conversation. Audio is kept in memory only, and only the most recent two minutes of it."""

    def __init__(self, conversation_id: str, backend: SpeakerBackend | None = None) -> None:
        self.conversation_id, self.backend = conversation_id, backend
        self.next_seq = 0
        self.last_end_ms = 0
        self._mu = threading.Lock()
        self._audio = bytearray()
        self._audio_start_ms = 0
        self._offset_ms = 0  # stream times restart at 0 on each new stream; turn times keep running

    def new_stream(self) -> None:
        with self._mu:
            self._audio.clear()
            self._audio_start_ms = 0
        self._offset_ms = self.last_end_ms

    def feed(self, pcm: bytes) -> None:
        if self.backend is None:
            return
        with self._mu:
            self._audio += pcm
            extra = len(self._audio) - KEEP_MS * BYTES_PER_MS
            if extra > 0:
                del self._audio[:extra]
                self._audio_start_ms += extra // BYTES_PER_MS

    def assign(self, segments: list[Segment]) -> list[Turn]:
        with self._mu:
            audio, audio_start = bytes(self._audio), self._audio_start_ms
        turns = []
        for seg in segments:
            speaker, confidence = "unknown", 0.0
            if self.backend is not None:
                lo = max(0, (seg.start_ms - audio_start) * BYTES_PER_MS)
                hi = max(lo, (seg.end_ms - audio_start) * BYTES_PER_MS)
                speaker, confidence = self.backend.label(audio[lo:hi], seg.start_ms, seg.end_ms)
            turn = Turn(
                id=f"{self.conversation_id}-t{self.next_seq}",
                conversation_id=self.conversation_id,
                seq=self.next_seq,
                speaker=speaker,
                speaker_confidence=min(1.0, max(0.0, confidence)),
                text=seg.text.strip(),
                start_ms=seg.start_ms + self._offset_ms,
                end_ms=seg.end_ms + self._offset_ms,
                source="asr",
            )
            turns.append(turn)
            self.next_seq += 1
            self.last_end_ms = turn.end_ms
        return turns

    def reset(self) -> None:
        self.next_seq = 0
        self.last_end_ms = 0
        self.new_stream()
