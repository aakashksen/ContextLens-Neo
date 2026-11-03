"""Rebuild whole-meeting audio from a Hugging Face audio dataset that stores segmented utterances in parquet.

Reads each shard's footer and the id column over HTTP range requests, picks the row groups that hold the wanted
meetings, fetches only those row groups' columns (no shard is saved to disk), and places every utterance at its
timestamp in a continuous 16-bit mono WAV, one per meeting. Nothing here is specific to AMI: the id, time, speaker and
audio columns are arguments, and the sample rate is read from the data.

  python tools/hf_audio.py plan  --dataset edinburghcstr/ami --config ihm --revision <sha> --meetings ES2003a ...
  python tools/hf_audio.py build --dataset edinburghcstr/ami --config ihm --revision <sha> --meetings ES2003a ... --out datasets/raw/ami-audio

How the audio is rebuilt (also stated in each manifest and in the E7 record):
- Each speaker (or channel) gets its own track. An utterance is written at round(begin_time * rate) samples; a later
  utterance of the same speaker replaces an overlapping earlier one, since both are the same microphone.
- Tracks are summed in 32 bits. If the sum ever passes the 16-bit range, the whole meeting is scaled by one gain
  (32767 / peak), so nothing is clipped and relative levels are kept; the gain is recorded (1.0 when none was needed).
- Time between utterances is digital silence. The meeting ends at its last utterance.
- Every utterance must be mono 16-bit and at the expected sample rate; anything else stops the run (no resampling).

Limits: stops when one fetch would be over 40 GB or free disk would fall under 40 GB. A 403 or 429 is retried 3 times
(30 s then 120 s, or the server's Retry-After if longer); if it persists the run stops and says so. The request headers
are urllib's defaults, with no cookies or proxy.
"""
import argparse
import io
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pyarrow.parquet as pq

API = "https://huggingface.co/api/datasets"
RESOLVE = "https://huggingface.co/datasets/{dataset}/resolve/{revision}/{path}"
MAX_SINGLE_BYTES = 40 * 10**9
MIN_FREE_BYTES = 40 * 10**9
BACKOFF_S = (30.0, 120.0)
PAUSE_S = 0.2
MANIFEST = "manifest.json"
REBUILT = "rebuilt from segmented utterances"


class Refused(Exception):
    """The server answered 403 or 429 on every try."""


class Stop(Exception):
    """A limit or a missing piece of data; the run does nothing more."""


def _request(url: str, headers: dict[str, str] | None, sleep: Callable[[float], None]) -> tuple[bytes, Any]:
    for wait in (*BACKOFF_S, None):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=120) as resp:
                return resp.read(), resp.headers
        except urllib.error.HTTPError as exc:
            if exc.code not in (403, 429):
                raise
            if wait is None:
                raise Refused(f"HTTP {exc.code} for {url} after {len(BACKOFF_S) + 1} tries") from None
            after = exc.headers.get("Retry-After") if exc.headers else None
            sleep(max(wait, float(after) if after and after.strip().isdigit() else 0.0))
    raise AssertionError("unreachable")


class HttpRangeFile(io.RawIOBase):
    """A read-only file over HTTP range requests, for pyarrow."""

    def __init__(self, url: str, sleep: Callable[[float], None] = time.sleep) -> None:
        self.url, self._sleep, self._pos, self.requests = url, sleep, 0, 0
        _, headers = _request(url, {"Range": "bytes=0-0"}, sleep)
        self.size = int(headers["Content-Range"].split("/")[1])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self._pos = offset if whence == 0 else self._pos + offset if whence == 1 else self.size + offset
        return self._pos

    def readinto(self, b: Any) -> int:
        n = min(len(b), self.size - self._pos)
        if n <= 0:
            return 0
        data, _ = _request(self.url, {"Range": f"bytes={self._pos}-{self._pos + n - 1}"}, self._sleep)
        self.requests += 1
        time.sleep(PAUSE_S)
        b[: len(data)] = data
        self._pos += len(data)
        return len(data)


@dataclass
class Columns:
    meeting: str = "meeting_id"
    begin: str = "begin_time"
    speaker: str = "speaker_id"
    audio: str = "audio"


@dataclass
class ShardPlan:
    path: str
    size: int
    row_groups: dict[int, int] = field(default_factory=dict)  # row group -> bytes to fetch for it
    meetings: set[str] = field(default_factory=set)


def list_shards(dataset: str, config: str, revision: str, sleep: Callable[[float], None] = time.sleep) -> list[tuple[str, int]]:
    body, _ = _request(f"{API}/{dataset}/revision/{revision}?blobs=true", None, sleep)
    files = json.loads(body)["siblings"]
    return sorted((f["rfilename"], f["size"]) for f in files
                  if f["rfilename"].startswith(config + "/") and f["rfilename"].endswith(".parquet"))


def resolve_revision(dataset: str, sleep: Callable[[float], None] = time.sleep) -> str:
    body, _ = _request(f"{API}/{dataset}", None, sleep)
    return json.loads(body)["sha"]


def plan_shard(f: BinaryIO, path: str, size: int, wanted: set[str], cols: Columns) -> ShardPlan:
    """Which row groups of one shard hold the wanted meetings, and how many bytes they cost. Column statistics
    narrow the row groups; the id column of the remaining ones is read to confirm."""
    pf = pq.ParquetFile(f)
    names = [pf.metadata.schema.column(i).path for i in range(pf.metadata.num_columns)]
    id_index = names.index(cols.meeting)
    out = ShardPlan(path, size)
    for g in range(pf.metadata.num_row_groups):
        stats = pf.metadata.row_group(g).column(id_index).statistics
        if stats is not None and stats.has_min_max and not any(stats.min <= m <= stats.max for m in wanted):
            continue
        found = wanted & set(pf.read_row_group(g, columns=[cols.meeting]).column(0).to_pylist())
        if found:
            rg = pf.metadata.row_group(g)
            keep = {cols.meeting, cols.begin, cols.speaker}
            out.row_groups[g] = sum(rg.column(i).total_compressed_size for i, n in enumerate(names)
                                    if n.split(".")[0] in keep | {cols.audio})
            out.meetings |= found
    return out


def make_plan(shards: list[tuple[str, int]], wanted: list[str], open_shard: Callable[[str], BinaryIO],
              cols: Columns = Columns(), say: Callable[[str], None] = print) -> dict[str, Any]:
    plans = []
    for path, size in shards:
        p = plan_shard(open_shard(path), path, size, set(wanted), cols)
        say(f"  {path}: {size / 1e6:.0f} MB, {len(p.row_groups)} row groups needed" if p.row_groups
            else f"  {path}: {size / 1e6:.0f} MB, none needed")
        if p.row_groups:
            plans.append(p)
    found = set().union(*(p.meetings for p in plans)) if plans else set()
    return {"shards": plans, "missing": [m for m in wanted if m not in found],
            "shard_bytes": sum(p.size for p in plans), "fetch_bytes": sum(sum(p.row_groups.values()) for p in plans)}


# -- rebuilding ------------------------------------------------------------------------------------------------------

@dataclass
class Utterance:
    begin_s: float
    speaker: str
    wav: bytes  # a complete WAV file


def rebuild_meeting(utterances: list[Utterance], rate: int = 16000) -> tuple[np.ndarray, dict[str, Any]]:
    """The meeting's samples (int16) and what was done. See the module docstring for the rules."""
    tracks: dict[str, np.ndarray] = {}
    placed = []
    for u in utterances:
        with wave.open(io.BytesIO(u.wav), "rb") as w:
            if (w.getnchannels(), w.getsampwidth()) != (1, 2):
                raise Stop("an utterance is not mono 16-bit")
            if w.getframerate() != rate:
                raise Stop(f"an utterance is at {w.getframerate()} Hz, not {rate}")
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        start = round(u.begin_s * rate)
        if start < 0:
            raise Stop("an utterance begins before 0 s")
        placed.append((u.speaker, start, pcm))
    length = max((s + len(p) for _, s, p in placed), default=0)
    for speaker, start, pcm in placed:
        track = tracks.setdefault(speaker, np.zeros(length, dtype=np.int16))
        track[start : start + len(pcm)] = pcm  # the same microphone: a later utterance replaces an overlap
    total = np.zeros(length, dtype=np.int32)
    for track in tracks.values():
        total += track
    peak = int(np.abs(total).max()) if length else 0
    gain = 32767 / peak if peak > 32767 else 1.0
    out = np.rint(total * gain).astype(np.int16) if gain != 1.0 else total.astype(np.int16)
    overlap = np.zeros(length, dtype=np.int8)
    for track in tracks.values():
        overlap += (track != 0)
    return out, {"utterances": len(placed), "speakers": len(tracks), "samples": length, "duration_s": length / rate,
                 "peak_before_scaling": peak, "gain": gain, "overlapped_samples": int((overlap > 1).sum())}


def write_wav_atomic(path: Path, samples: np.ndarray, rate: int) -> int:
    """16-bit mono WAV to a hidden temp file beside `path`, checked for size, then renamed. Returns its size."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with wave.open(str(tmp), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(samples.astype("<i2").tobytes())
        want = 44 + 2 * len(samples)
        if tmp.stat().st_size != want:
            raise OSError(f"{path.name}: wrote {tmp.stat().st_size} bytes, expected {want}")
        os.replace(tmp, path)
        return want
    finally:
        tmp.unlink(missing_ok=True)


def fetch_utterances(f: BinaryIO, shard: ShardPlan, meeting: str, cols: Columns) -> list[Utterance]:
    pf = pq.ParquetFile(f)
    out = []
    for g in shard.row_groups:
        table = pf.read_row_group(g, columns=[cols.meeting, cols.begin, cols.speaker, cols.audio]).to_pylist()
        out += [Utterance(float(r[cols.begin]), str(r[cols.speaker]), r[cols.audio]["bytes"])
                for r in table if r[cols.meeting] == meeting]
    return out


def check_limits(plan: dict[str, Any], free_bytes: int, written_bytes: int) -> None:
    for p in plan["shards"]:
        if sum(p.row_groups.values()) > MAX_SINGLE_BYTES:
            raise Stop(f"{p.path}: one fetch over 40 GB")
    if free_bytes - written_bytes < MIN_FREE_BYTES:
        raise Stop(f"{(free_bytes - written_bytes) / 1e9:.1f} GB would be free after writing, under 40 GB")


def build(dataset: str, config: str, revision: str, meetings: list[str], out_dir: Path, *,
          open_shard: Callable[[str], BinaryIO], shards: list[tuple[str, int]], cols: Columns = Columns(),
          rate: int = 16000, free: Callable[[], int] | None = None, say: Callable[[str], None] = print) -> dict[str, Any]:
    """Plan, check the limits, then rebuild each meeting one at a time. A meeting the data lacks stops the run before
    anything is fetched. Returns the manifest, also written to `out_dir/manifest.json`."""
    plan = make_plan(shards, meetings, open_shard, cols, say)
    if plan["missing"]:
        raise Stop(f"not in {dataset} {config}: {plan['missing']}")
    estimate = int(plan["fetch_bytes"])  # compressed audio is never larger than the WAV it becomes, so this is low
    out_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = (free or (lambda: shutil.disk_usage(out_dir).free))()
    check_limits(plan, free_bytes, estimate)
    manifest_path = out_dir / MANIFEST
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if manifest.get("revision") not in (None, revision) or manifest.get("config") not in (None, config):
        raise Stop("the existing manifest is for another revision or config; use another --out")
    manifest.update(dataset=dataset, config=config, revision=revision, sample_rate=rate, method=REBUILT,
                    rule=__doc__.split("How the audio is rebuilt")[1].split("Limits")[0].strip(), meetings=manifest.get("meetings", {}))
    for m in meetings:
        dest, done = out_dir / f"{m}.wav", manifest["meetings"].get(m)
        if done and dest.is_file() and dest.stat().st_size == done["bytes"]:
            say(f"{m}: already complete")
            continue
        utterances = []
        for shard in (p for p in plan["shards"] if m in p.meetings):
            utterances += fetch_utterances(open_shard(shard.path), shard, m, cols)
        samples, info = rebuild_meeting(utterances, rate)
        info["bytes"] = write_wav_atomic(dest, samples, rate)
        manifest["meetings"][m] = info
        write_text_atomic(manifest_path, json.dumps(manifest, indent=1) + "\n")
        say(f"{m}: {info['duration_s']:.0f} s, {info['utterances']} utterances, gain {info['gain']:.3f}")
    return manifest


def write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hf_audio")
    ap.add_argument("cmd", choices=["plan", "build"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--revision", help="commit sha to pin; default is the dataset's current one, which is printed")
    ap.add_argument("--meetings", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("datasets/raw/ami-audio"))
    ap.add_argument("--sample-rate", type=int, default=16000)
    for name in ("meeting", "begin", "speaker", "audio"):
        ap.add_argument(f"--{name}-column", default=getattr(Columns(), name))
    a = ap.parse_args(argv)
    cols = Columns(a.meeting_column, a.begin_column, a.speaker_column, a.audio_column)
    revision = a.revision or resolve_revision(a.dataset)
    print(f"{a.dataset} {a.config} at {revision}")
    shards = list_shards(a.dataset, a.config, revision)
    print(f"{len(shards)} shards, {sum(s for _, s in shards) / 1e9:.1f} GB in all")

    def open_shard(path: str) -> BinaryIO:
        return HttpRangeFile(RESOLVE.format(dataset=a.dataset, revision=revision, path=path))  # type: ignore[return-value]

    try:
        if a.cmd == "plan":
            plan = make_plan(shards, a.meetings, open_shard, cols)
            print(f"found {len(a.meetings) - len(plan['missing'])} of {len(a.meetings)} meetings; missing {plan['missing']}")
            print(f"shards holding them: {len(plan['shards'])}, {plan['shard_bytes'] / 1e9:.2f} GB whole; "
                  f"row groups to fetch: {plan['fetch_bytes'] / 1e9:.2f} GB")
            return 2 if plan["missing"] else 0
        build(a.dataset, a.config, revision, a.meetings, a.out, open_shard=open_shard, shards=shards, cols=cols,
              rate=a.sample_rate)
    except (Stop, Refused) as exc:
        print("STOP", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
