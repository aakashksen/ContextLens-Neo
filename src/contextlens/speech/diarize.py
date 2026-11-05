"""Speaker labels for transcribed segments.

Chosen option: pyannote speaker-diarization-3.1 (see docs/datasets.md), run in memory over a rolling window of
the most recent audio. It is not streaming, so a turn is first sent as "unknown" at confidence 0.0 and the speaker
is filled in later in the background (`speaker_updates`). Without a backend every turn stays "unknown".
"""

import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from contextlens.device import resolve_device
from contextlens.schemas import Turn
from contextlens.speech.client import Segment

SAMPLE_RATE = 16000
BYTES_PER_MS = SAMPLE_RATE * 2 // 1000
KEEP_MS = 120_000
PYANNOTE_NEEDS_MB = 2000  # a budget, not a measurement


class SpeakerBackend(Protocol):
    def label(self, pcm: bytes, start_ms: int, end_ms: int) -> tuple[str, float]:
        """Speaker label and confidence in [0, 1] for this span of audio (16 kHz mono 16-bit PCM)."""
        ...


class WindowBackend(Protocol):
    def spans(self, pcm: bytes) -> list[tuple[int, int, str]]:
        """Who spoke when in this audio (16 kHz mono 16-bit PCM): (start_ms, end_ms, speaker) from its start."""
        ...


def hf_token(dotenv: Path | None = None) -> str:
    """The Hugging Face token from the environment, else from `.env`. Its value is never put in a message."""
    token = os.environ.get("HF_TOKEN", "").strip()
    path = dotenv or Path(__file__).resolve().parents[3] / ".env"
    if not token and path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "HF_TOKEN":
                token = value.strip().strip("\"'")
    if not token:
        raise RuntimeError("HF_TOKEN is not set in the environment or .env; pyannote needs it to load its model")
    return token


class PyannoteBackend:
    """pyannote/speaker-diarization-3.1 on the GPU when it has room (else the CPU, with a warning), given the audio as an in-memory waveform (nothing is written)."""

    MODEL = "pyannote/speaker-diarization-3.1"

    def __init__(self) -> None:
        import torch

        from pyannote.audio import Pipeline

        self._torch = torch
        try:
            self._pipeline = Pipeline.from_pretrained(self.MODEL, token=hf_token())
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"pyannote could not load {self.MODEL} ({type(exc).__name__}). pyannote.audio 4.x also loads a PLDA "
                "from the gated repo pyannote/speaker-diarization-community-1, so the person must accept that "
                "repo's terms on Hugging Face first (see docs/datasets.md). Until then speakers stay 'unknown'."
            ) from None
        if self._pipeline is None:
            raise RuntimeError("pyannote could not load its pipeline; check that the model terms were accepted")
        self.device = resolve_device(PYANNOTE_NEEDS_MB)
        self._pipeline.to(torch.device(self.device))

    def spans(self, pcm: bytes) -> list[tuple[int, int, str]]:
        torch = self._torch
        samples = torch.frombuffer(bytearray(pcm), dtype=torch.int16).to(torch.float32) / 32768.0
        out = self._pipeline({"waveform": samples.unsqueeze(0), "sample_rate": SAMPLE_RATE})
        annotation = getattr(out, "speaker_diarization", out)  # pyannote 4 wraps the annotation
        return [
            (round(seg.start * 1000), round(seg.end * 1000), str(label))
            for seg, _, label in annotation.itertracks(yield_label=True)
        ]


class SharedSpeakerModel:
    """One speaker model for every conversation: loaded on the first call, and used by one conversation at a time.
    A load that fails is not tried again; the same error is raised to each caller."""

    def __init__(self, load: Callable[[], WindowBackend]) -> None:
        self._load, self._backend, self._error = load, None, None
        self._mu = threading.Lock()  # loading
        self._run = threading.Lock()  # running the model

    def __call__(self) -> WindowBackend:
        with self._mu:
            if self._error is not None:
                raise self._error
            if self._backend is None:
                try:
                    self._backend = _Serialized(self._load(), self._run)
                except Exception as exc:
                    self._error = exc
                    raise
            return self._backend


class _Serialized:
    def __init__(self, inner: WindowBackend, lock: threading.Lock) -> None:
        self._inner, self._lock = inner, lock

    def spans(self, pcm: bytes) -> list[tuple[int, int, str]]:
        with self._lock:
            return self._inner.spans(pcm)


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def _match_known_names(abs_spans: list[tuple[int, int, str]], inside: list[Turn], known: set[str]) -> dict[str, str]:
    """Each pipeline name mapped to the stable name already given to the turns it overlaps most."""
    score: dict[tuple[str, str], int] = {}
    for t in inside:
        if t.speaker in known:
            for a, b, who in abs_spans:
                score[(who, t.speaker)] = score.get((who, t.speaker), 0) + _overlap(a, b, t.start_ms, t.end_ms)
    stable: dict[str, str] = {}
    for (who, name), n in sorted(score.items(), key=lambda kv: -kv[1]):
        if n > 0 and who not in stable and name not in stable.values():
            stable[who] = name
    return stable


class Diarizer:
    """One per conversation. Audio is kept in memory only, and only the most recent two minutes of it."""

    def __init__(self, conversation_id: str, backend: SpeakerBackend | WindowBackend | None = None) -> None:
        self.conversation_id, self.backend = conversation_id, backend
        self.background = hasattr(backend, "spans")  # then assign() leaves speakers for speaker_updates()
        self._names = 0
        self.next_seq = 0
        self.last_end_ms = 0
        self._mu = threading.Lock()
        self._audio = bytearray()
        self._dropped = 0  # bytes of this stream no longer held; counting bytes keeps the clock exact for any chunk size
        self._offset_ms = 0  # stream times restart at 0 on each new stream; turn times keep running

    def set_backend(self, backend: SpeakerBackend | WindowBackend) -> None:
        background = hasattr(backend, "spans")  # outside the lock: a property here may call back into us
        with self._mu:
            self.backend, self.background = backend, background

    def new_stream(self) -> None:
        with self._mu:
            self._audio.clear()
            self._dropped = 0
        self._offset_ms = self.last_end_ms

    def feed(self, pcm: bytes) -> None:
        with self._mu:
            if self.backend is None:  # not loaded (yet): keep no audio, but keep the clock
                self._dropped += len(pcm)
                return
            self._audio += pcm
            extra = len(self._audio) - KEEP_MS * BYTES_PER_MS
            if extra > 0:
                del self._audio[:extra]
                self._dropped += extra

    def assign(self, segments: list[Segment]) -> list[Turn]:
        with self._mu:
            audio, audio_start = bytes(self._audio), self._dropped // BYTES_PER_MS
            backend, background = self.backend, self.background
        turns = []
        for seg in segments:
            speaker, confidence = "unknown", 0.0
            if backend is not None and not background:
                lo = max(0, (seg.start_ms - audio_start) * BYTES_PER_MS)
                hi = max(lo, (seg.end_ms - audio_start) * BYTES_PER_MS)
                speaker, confidence = backend.label(audio[lo:hi], seg.start_ms, seg.end_ms)
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

    def window(self) -> tuple[bytes, int]:
        """The retained audio and the turn-clock time (ms) at which it starts."""
        with self._mu:
            return bytes(self._audio), self._dropped // BYTES_PER_MS + self._offset_ms

    def speaker_updates(
        self, spans: list[tuple[int, int, str]], window_start_ms: int, window_end_ms: int, turns: list[Turn]
    ) -> list[tuple[int, str, float]]:
        """(seq, speaker, confidence) for each turn lying inside the window that someone spoke during.

        `spans` come from the window's audio. The pipeline names speakers afresh on every run, so each of its
        names is matched to the stable name already given to turns it overlaps (largest overlap first); a
        name matching nothing gets a new stable name. Confidence is the winner's share of the turn.
        """
        abs_spans = [(a + window_start_ms, b + window_start_ms, who) for a, b, who in spans]
        inside = [t for t in turns if t.start_ms >= window_start_ms and t.end_ms <= window_end_ms]
        known = {t.speaker for t in turns if t.speaker != "unknown" and t.speaker_confidence >= 0.5}
        stable = _match_known_names(abs_spans, inside, known)

        updates = []
        for t in inside:
            share: dict[str, int] = {}
            for a, b, who in abs_spans:
                share[who] = share.get(who, 0) + _overlap(a, b, t.start_ms, t.end_ms)
            who = max(share, key=share.get, default=None)
            if who is None or share[who] == 0:
                continue
            if who not in stable:
                while (name := f"speaker_{self._names + 1}") in known | set(stable.values()):
                    self._names += 1
                self._names += 1
                stable[who] = name
            updates.append((t.seq, stable[who], share[who] / max(1, t.end_ms - t.start_ms)))
        return updates

    def reset(self) -> None:
        self._names = 0
        self.next_seq = 0
        self.last_end_ms = 0
        self.new_stream()
