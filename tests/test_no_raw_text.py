"""QMSum and DECODE may be used locally but never committed: no tracked or committable file may hold their text."""
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW_DIRS = [ROOT / "datasets/raw/qmsum", ROOT / "datasets/raw/decode"]
NL = chr(10)
WINDOW = 8  # a run of this many words in a row counts as copied text

# Everyday 8-word phrases that appear in both our scenarios and DECODE. Exact match on the normalized window only.
ALLOWED_WINDOWS = {
    "that s a lot of work for one": "stock complaint, not DECODE-specific",
    "s a lot of work for one person": "same idiom, shifted one word",
    "that makes a lot of sense i ve": "common agreement phrase",
    "can you give me an example of a": "common request phrase",
    "i ll believe it when i see it": "proverb",
}


def _strings(x: object) -> Iterator[str]:
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for k, v in x.items():  # a key can hold copied text as well as a value
            yield from _strings(k)
            yield from _strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from _strings(v)


def _texts(path: Path, joined: bool = False) -> Iterator[str]:
    """Strings of a file. `joined` reads a plain file as one text, so a quote hard-wrapped across its lines is found;
    the raw side is read line by line, because its own line breaks are not part of the copied text."""
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
        yield from [raw] if joined else raw.splitlines()


def _windows(text: str, allowed: frozenset[str] | set[str] = frozenset()) -> Iterator[int]:
    words = re.findall(r"\w+", text.lower())
    for i in range(len(words) - WINDOW + 1):
        window = " ".join(words[i : i + WINDOW])
        if window not in allowed:
            yield int.from_bytes(hashlib.blake2b(window.encode(), digest_size=8).digest(), "big")


def leaked_files(raw_dirs: list[Path], files: list[Path], allowed: set[str] | frozenset[str] = frozenset(ALLOWED_WINDOWS)) -> list[str]:
    """Files among `files` that share a run of WINDOW words with any file under `raw_dirs`, ignoring `allowed` windows."""
    seen = {h for d in raw_dirs if d.is_dir() for f in d.rglob("*") if f.is_file() for t in _texts(f) for h in _windows(t)}
    return [f.as_posix() for f in files if any(h in seen for t in _texts(f, joined=True) for h in _windows(t, allowed))]


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


def test_a_quote_hard_wrapped_across_lines_in_a_plain_file_is_flagged(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    words = [f"wrapped{i}" for i in range(40)]
    (raw / "a.txt").write_text(" ".join(words), encoding="utf-8")
    wrapped = tmp_path / "wrapped.md"
    wrapped.write_text(NL.join(" ".join(words[i : i + 4]) for i in range(0, 40, 4)), encoding="utf-8")
    short = tmp_path / "short_lines.md"  # 7 words in a row, then a different line: below the window
    short.write_text(NL.join(" ".join(words[i : i + 7]) + " elsewhere" for i in range(0, 28, 7)), encoding="utf-8")
    assert leaked_files([raw], [wrapped, short]) == [wrapped.as_posix()]


def test_a_quote_used_as_a_json_dict_key_is_flagged(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    passage = " ".join(f"keyed{i}" for i in range(30))
    (raw / "a.jsonl").write_text(json.dumps({"content": passage}) + NL, encoding="utf-8")
    top = tmp_path / "top.json"
    top.write_text(json.dumps({passage: 1}), encoding="utf-8")
    nested = tmp_path / "nested.json"
    nested.write_text(json.dumps({"rows": [{"ids": {passage: ["x"]}}]}), encoding="utf-8")
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"a": 1}) + NL + json.dumps({passage: 2}) + NL, encoding="utf-8")
    clean = tmp_path / "clean.json"
    clean.write_text(json.dumps({"keyed1 keyed2": passage.split()[0]}), encoding="utf-8")
    assert leaked_files([raw], [top, nested, rows, clean]) == [top.as_posix(), nested.as_posix(), rows.as_posix()]


def test_allowlist_is_exact_and_nothing_else_is_forgiven(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.txt").write_text(
        "\n".join(list(ALLOWED_WINDOWS) + ["that s a lot of work for two", "zorp blat quux flim flam wibble wobble snork plonk"]), encoding="utf-8")
    allowed_only = tmp_path / "allowed_only.md"
    allowed_only.write_text("Well, " + "\n".join(p.upper() for p in ALLOWED_WINDOWS), encoding="utf-8")
    one_word_off = tmp_path / "one_word_off.md"
    one_word_off.write_text("that s a lot of work for two", encoding="utf-8")
    new_phrase = tmp_path / "new_phrase.md"
    new_phrase.write_text("intro. zorp blat quux flim flam wibble wobble snork. end", encoding="utf-8")
    longer = tmp_path / "longer.md"  # contains an allowed phrase plus other copied words
    longer.write_text("zorp blat quux flim flam wibble wobble snork plonk that s a lot of work for one", encoding="utf-8")
    files = [allowed_only, one_word_off, new_phrase, longer]
    assert leaked_files([raw], files) == [one_word_off.as_posix(), new_phrase.as_posix(), longer.as_posix()]


def _committable_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                         capture_output=True, text=True, check=True)
    return [ROOT / p for p in out.stdout.splitlines() if (ROOT / p).is_file()]


def test_no_tracked_file_holds_qmsum_or_decode_text() -> None:
    present = [d for d in RAW_DIRS if d.is_dir()]
    if not present:
        pytest.skip("datasets/raw/qmsum and datasets/raw/decode are not on this machine, so there is nothing to compare")
    assert leaked_files(present, _committable_files()) == []


@pytest.mark.parametrize("seed", ["0", "1", "2", "3", "5", "8", "13", "21"])
def test_detection_does_not_depend_on_the_hash_seed(tmp_path: Path, seed: str) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    words = [f"zq{i}x" for i in range(60)]
    (raw / "a.txt").write_text(" ".join(words), encoding="utf-8")
    exact8 = tmp_path / "exact8.md"
    exact8.write_text("pre " + " ".join(words[20:28]) + " post", encoding="utf-8")
    seven = tmp_path / "seven.md"
    seven.write_text(" ".join(words[20:27]) + " other words here " + " ".join(words[40:47]), encoding="utf-8")
    script = (
        "import importlib.util, sys, pathlib\n"
        f"spec = importlib.util.spec_from_file_location('m', {str(Path(__file__))!r})\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        f"raw = pathlib.Path({str(raw)!r})\n"
        f"print(m.leaked_files([raw], [pathlib.Path({str(exact8)!r}), pathlib.Path({str(seven)!r})]))\n"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONHASHSEED": seed})
    assert out.stdout.strip() == str([exact8.as_posix()])
