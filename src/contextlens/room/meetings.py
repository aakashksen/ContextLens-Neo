"""The AMI meetings the room can play: audio files, timed transcripts, and what AMI's own annotators wrote down.

AMI Meeting Corpus, CC BY 4.0, https://groups.inf.ed.ac.uk/ami/corpus/ . Nothing here is downloaded; it reads what
is already under datasets/raw (ami/ for annotations, ami-audio/ for the rebuilt meeting audio, and ami-video/ if
someone put room video there)."""

import re
import wave
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from contextlens.schemas import MemoryItem, Turn

ATTRIBUTION = ("AMI Meeting Corpus, CC BY 4.0, https://groups.inf.ed.ac.uk/ami/corpus/ . "
               "Audio is rebuilt from AMI's segmented headset utterances.")
SAMPLE_RATE = 16000
VIDEO_EXT = (".mp4", ".webm")
_ID = re.compile(r"^[A-Za-z]{2}\d{4}[a-z]$")
_STOP = frozenset("a an the and or but of to in on at for with is are was were be been it its this that these those "
                  "i we you he she they them our my your so do does did not no yes yeah um uh mm okay ok just like "
                  "have has had will would can could should as if then than there here what which who how when".split())


@dataclass(frozen=True)
class Meeting:
    id: str
    wav: Path
    duration_ms: int
    video: Path | None = None

    def public(self) -> dict:
        return {"id": self.id, "duration_ms": self.duration_ms, "video": self.video is not None}


def valid_id(meeting_id: str) -> bool:
    return bool(_ID.match(meeting_id))


def _duration_ms(wav: Path) -> int:
    with wave.open(str(wav), "rb") as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{wav.name} is not 16 kHz mono 16-bit")
        return round(w.getnframes() * 1000 / w.getframerate())


def find_meetings(raw: Path) -> dict[str, Meeting]:
    """Every meeting with a playable audio file. A file that is not 16 kHz mono 16-bit WAV is skipped."""
    found: dict[str, Meeting] = {}
    for wav in sorted((raw / "ami-audio").glob("*.wav")):
        if not valid_id(wav.stem):
            continue
        try:
            dur = _duration_ms(wav)
        except (ValueError, wave.Error, OSError, EOFError):
            continue
        video = next((p for ext in VIDEO_EXT if (p := raw / "ami-video" / f"{wav.stem}{ext}").is_file()), None)
        found[wav.stem] = Meeting(wav.stem, wav, dur, video)
    return found


def wav_chunks(wav: Path, start_ms: int, chunk_ms: int = 500) -> Iterator[bytes]:
    """16-bit mono PCM from `start_ms` to the end, `chunk_ms` at a time."""
    with wave.open(str(wav), "rb") as w:
        w.setpos(min(max(start_ms, 0) * SAMPLE_RATE // 1000, w.getnframes()))
        frames = chunk_ms * SAMPLE_RATE // 1000
        while pcm := w.readframes(frames):
            yield pcm


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """One `bytes=a-b` range as inclusive (start, end), or None for no (or an unusable) header. Raises ValueError
    for a range outside the file."""
    if not header or not header.startswith("bytes=") or "," in header:
        return None
    first, _, last = header[6:].partition("-")
    try:
        if first == "":
            start, end = max(size - int(last), 0), size - 1
        else:
            start, end = int(first), (int(last) if last else size - 1)
    except ValueError:
        return None
    end = min(end, size - 1)
    if start < 0 or start >= size or end < start:
        raise ValueError("range not satisfiable")
    return start, end


def timed_turns(raw: Path, meeting_id: str) -> list[Turn]:
    """AMI's own transcript with word timings, one turn per annotated segment, in time order."""
    from contextlens.data.loaders import _load_ami

    convs = _load_ami(raw, {meeting_id})
    return convs[0].turns if convs else []


def summary_items(raw: Path, meeting_id: str) -> list[tuple[str, str]]:
    """What AMI's annotators wrote as decisions and actions for a meeting: (kind, sentence). Kinds follow the memory
    extractor's: `decision` and `commitment`."""
    path = raw / "ami" / "abstractive" / f"{meeting_id}.abssumm.xml"
    if not path.is_file():
        return []
    out: list[tuple[str, str]] = []
    for node in ET.parse(path).getroot():
        kind = {"decisions": "decision", "actions": "commitment"}.get(node.tag)
        if kind:
            out += [(kind, " ".join("".join(s.itertext()).split())) for s in node.iter("sentence")]
    return [(k, s) for k, s in out if s]


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in _STOP and len(w) > 2}


def align_items(meeting_id: str, items: list[tuple[str, str]], turns: list[Turn], min_shared: int = 2) -> list[MemoryItem]:
    """Places each annotated sentence at the turn that shares the most content
    words with it, as an item that exists from that turn's seq on. A sentence with fewer than `min_shared` shared
    words is left out, not guessed. This stands in for the memory extractor when no model runs, and the page says so."""
    words = [content_words(t.text) for t in turns]
    placed: list[MemoryItem] = []
    for n, (kind, sentence) in enumerate(items):
        want = content_words(sentence)
        best, best_i = 0, -1
        for i, w in enumerate(words):
            score = len(want & w)
            if score >= max(best, 1):  # a tie goes to the later turn: by then more of the sentence has been said
                best, best_i = score, i
        if best < min_shared or best_i < 0:
            continue
        t = turns[best_i]
        subject = " ".join(sorted(want & content_words(t.text))[:3]) or kind
        placed.append(MemoryItem(
            id=f"{meeting_id}-ann{n}", conversation_id=meeting_id, kind=kind, subject=subject,
            statement=sentence, speaker=t.speaker, source_turn_ids=[t.id], supersedes=None, as_of_seq=t.seq,
        ))
    return sorted(placed, key=lambda m: m.as_of_seq)
