"""QMSum and DECODE may be used locally but never committed: no tracked or committable file may hold their text."""
import json
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW_DIRS = [ROOT / "datasets/raw/qmsum", ROOT / "datasets/raw/decode"]
WINDOW = 8  # a run of this many words in a row counts as copied text
SAMPLE = 4  # keep one window in SAMPLE, chosen by content, so a copied passage of 30 words is still found


def _strings(x: object) -> Iterator[str]:
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from _strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from _strings(v)


def _texts(path: Path) -> Iterator[str]:
    try:
        raw = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return
    if path.suffix in (".json", ".jsonl"):
        lines = raw.splitlines() if path.suffix == ".jsonl" else [raw]
        for line in lines:
            try:
                yield from _strings(json.loads(line))
            except ValueError:
                yield line
    else:
        yield from raw.splitlines()


def _windows(text: str) -> Iterator[int]:
    words = re.findall(r"\w+", text.lower())
    for i in range(len(words) - WINDOW + 1):
        h = hash(" ".join(words[i : i + WINDOW]))
        if h % SAMPLE == 0:
            yield h


def leaked_files(raw_dirs: list[Path], files: list[Path]) -> list[str]:
    """Files among `files` that share a run of WINDOW words with any file under `raw_dirs`."""
    seen = {h for d in raw_dirs if d.is_dir() for f in d.rglob("*") if f.is_file() for t in _texts(f) for h in _windows(t)}
    return [f.as_posix() for f in files if any(h in seen for t in _texts(f) for h in _windows(t))]


def test_detector_flags_copied_text_and_nothing_else(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    passage = " ".join(f"invented{i}" for i in range(40))
    (raw / "a.jsonl").write_text(json.dumps({"meeting_transcripts": [{"content": passage}]}) + "\n", encoding="utf-8")
    copied = tmp_path / "copied.md"
    copied.write_text("notes: " + " ".join(passage.split()[5:35]), encoding="utf-8")
    as_json = tmp_path / "rows.json"
    as_json.write_text(json.dumps({"quote": passage.upper()}), encoding="utf-8")
    clean = tmp_path / "clean.md"
    clean.write_text("ids only: invented1 invented2 invented3, and some other words entirely unrelated here", encoding="utf-8")
    found = leaked_files([raw], [copied, as_json, clean])
    assert found == [copied.as_posix(), as_json.as_posix()]


def _committable_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                         capture_output=True, text=True, check=True)
    return [ROOT / p for p in out.stdout.splitlines() if (ROOT / p).is_file()]


def test_no_tracked_file_holds_qmsum_or_decode_text() -> None:
    present = [d for d in RAW_DIRS if d.is_dir()]
    if not present:
        pytest.skip("datasets/raw/qmsum and datasets/raw/decode are not on this machine, so there is nothing to compare")
    assert leaked_files(present, _committable_files()) == []
