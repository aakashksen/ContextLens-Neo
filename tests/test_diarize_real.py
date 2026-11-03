"""Real pyannote on a generated two-voice recording (Windows voices David and Zira): needs HF_TOKEN and the model."""

import builtins
import json
import os
import time
import wave
from pathlib import Path

import pytest

from contextlens.speech.client import Segment
from contextlens.speech.diarize import Diarizer, PyannoteBackend

FIXTURES = Path(__file__).parent / "fixtures" / "speech"


@pytest.mark.real
@pytest.mark.skip(
    reason="pyannote.audio 4.0.7 loads a PLDA from the gated repo pyannote/speaker-diarization-community-1, whose "
    "terms the person has not accepted; 3.x does not import with torchaudio 2.11. Remove this mark once accepted."
)
def test_pyannote_tells_two_voices_apart_in_memory(tmp_path, monkeypatch):
    with wave.open(str(FIXTURES / "two_voices.wav")) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        pcm = w.readframes(w.getnframes())
    told = json.loads((FIXTURES / "two_voices.json").read_text())

    loaded = time.monotonic()
    backend = PyannoteBackend()  # may download the model on first use; that is not audio
    load_s = time.monotonic() - loaded

    d = Diarizer("c1", backend)
    d.new_stream()
    d.feed(pcm)
    turns = d.assign([Segment(text=f"turn {i}", start_ms=t["start_ms"], end_ms=t["end_ms"]) for i, t in enumerate(told)])
    assert {t.speaker for t in turns} == {"unknown"}  # nothing waits for the pipeline

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    writes, real_open, real_os_open = [], builtins.open, os.open

    def spy_open(file, mode="r", *a, **k):
        if any(c in str(mode) for c in "wax+"):
            writes.append(str(file))
        return real_open(file, mode, *a, **k)

    def spy_os_open(path, flags, *a, **k):
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
            writes.append(str(path))
        return real_os_open(path, flags, *a, **k)

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(os, "open", spy_os_open)
    started = time.monotonic()
    pcm_window, start = d.window()
    spans = backend.spans(pcm_window)
    run_s = time.monotonic() - started
    monkeypatch.undo()
    print(f"\npyannote on CPU: load {load_s:.1f} s, diarizing {len(pcm) / 32000:.0f} s of audio took {run_s:.1f} s")

    assert writes == [] and list(tmp_path.rglob("*")) == []  # no file written, none left in the temp dir
    got = d.speaker_updates(spans, start, start + len(pcm_window) // 32, turns)
    who = {seq: name for seq, name, _ in got}
    assert sorted(who) == [0, 1, 2, 3]
    assert who[0] == who[2] and who[1] == who[3] and who[0] != who[1]
