"""tools/hf_audio.py: rebuilding whole meetings from segmented utterances, on small local parquet fixtures."""
import io
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import hf_audio as H  # noqa: E402

RATE = 16000


def wav(value: int, seconds: float, rate: int = RATE, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.full(int(seconds * rate) * channels, value, dtype="<i2").tobytes())
    return buf.getvalue()


def utt(begin: float, speaker: str, value: int, seconds: float, **kw) -> H.Utterance:
    return H.Utterance(begin, speaker, wav(value, seconds, **kw))


def test_utterances_land_at_their_timestamps_with_silence_between() -> None:
    samples, info = H.rebuild_meeting([utt(1.0, "A", 1000, 0.5), utt(3.0, "A", 500, 0.25)])
    assert len(samples) == int(3.25 * RATE) and info["duration_s"] == 3.25 and info["samples"] == 52000
    assert not samples[:16000].any() and (samples[16000:24000] == 1000).all()
    assert not samples[24000:48000].any() and (samples[48000:] == 500).all()  # the gap is silence
    assert info["gain"] == 1.0 and info["overlapped_samples"] == 0


def test_overlap_between_speakers_is_summed_and_the_same_speaker_is_replaced() -> None:
    samples, info = H.rebuild_meeting([utt(1.0, "A", 1000, 0.5), utt(1.25, "B", 2000, 0.5)])
    assert (samples[16000:20000] == 1000).all() and (samples[20000:24000] == 3000).all()
    assert (samples[24000:28000] == 2000).all() and info["overlapped_samples"] == 4000
    same, _ = H.rebuild_meeting([utt(0.0, "A", 1000, 0.5), utt(0.25, "A", 2000, 0.5)])
    assert (same[:4000] == 1000).all() and (same[4000:12000] == 2000).all()  # not 3000: one microphone, one signal


def test_a_sum_past_16_bits_is_scaled_not_clipped() -> None:
    samples, info = H.rebuild_meeting([utt(0.0, "A", 30000, 0.5), utt(0.0, "B", 30000, 0.5), utt(1.0, "A", 10000, 0.1)])
    assert info["peak_before_scaling"] == 60000 and info["gain"] == pytest.approx(32767 / 60000)
    assert samples.max() == 32767 and (samples[:8000] == 32767).all()  # flat, because both were equal: no wrap-around
    assert samples[16000] == round(10000 * 32767 / 60000)  # one gain for the whole meeting keeps relative levels


def test_sample_rate_and_format_are_checked_not_converted() -> None:
    with pytest.raises(H.Stop, match="8000 Hz"):
        H.rebuild_meeting([utt(0.0, "A", 1, 0.1, rate=8000)])
    with pytest.raises(H.Stop, match="not mono"):
        H.rebuild_meeting([utt(0.0, "A", 1, 0.1, channels=2)])
    samples, _ = H.rebuild_meeting([utt(0.0, "A", 1, 0.1, rate=8000)], rate=8000)
    assert len(samples) == 800


def shard(path: Path, rows: list[tuple[str, float, str, bytes]], group: int = 2) -> None:
    table = pa.table({
        "meeting_id": [r[0] for r in rows], "begin_time": pa.array([r[1] for r in rows], pa.float32()),
        "speaker_id": [r[2] for r in rows],
        "audio": pa.array([{"bytes": r[3], "path": "x.wav"} for r in rows],
                          pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    })
    pq.write_table(table, path, row_group_size=group)


@pytest.fixture
def shards(tmp_path: Path):
    d = tmp_path / "hf"
    d.mkdir()
    shard(d / "s0.parquet", [("M1", 1.0, "A", wav(1000, 0.5)), ("M1", 1.25, "B", wav(2000, 0.5)),
                             ("M1", 3.0, "A", wav(500, 0.25)), ("M1", 3.5, "B", wav(7, 0.1))])
    shard(d / "s1.parquet", [("M2", 0.0, "C", wav(9, 0.1)), ("M2", 0.5, "D", wav(9, 0.1)),
                             ("M3", 0.0, "C", wav(4, 0.2)), ("M3", 1.0, "C", wav(4, 0.2))])
    listing = [("s0.parquet", (d / "s0.parquet").stat().st_size), ("s1.parquet", (d / "s1.parquet").stat().st_size)]
    opened: list[str] = []

    def open_shard(path: str):
        opened.append(path)
        return open(d / path, "rb")

    return listing, open_shard, opened


def test_the_plan_names_only_the_row_groups_that_hold_the_meetings(shards) -> None:
    listing, open_shard, _ = shards
    plan = H.make_plan(listing, ["M1"], open_shard, say=lambda m: None)
    assert [p.path for p in plan["shards"]] == ["s0.parquet"] and sorted(plan["shards"][0].row_groups) == [0, 1]
    both = H.make_plan(listing, ["M1", "M3"], open_shard, say=lambda m: None)
    assert [p.path for p in both["shards"]] == ["s0.parquet", "s1.parquet"] and sorted(both["shards"][1].row_groups) == [1]
    assert 0 < both["fetch_bytes"] < both["shard_bytes"] and both["missing"] == []


def test_build_writes_the_meeting_wav_and_a_manifest_and_resumes(shards, tmp_path: Path) -> None:
    listing, open_shard, opened = shards
    out = tmp_path / "audio"
    manifest = H.build("d/x", "ihm", "abc123", ["M1", "M3"], out, open_shard=open_shard, shards=listing,
                       free=lambda: 900 * 10**9, say=lambda m: None)
    with wave.open(str(out / "M1.wav")) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (RATE, 1, 2)
        assert w.getnframes() == int(3.6 * RATE)  # ends at its last utterance: 3.5 s + 0.1 s
    with wave.open(str(out / "M3.wav")) as w:
        assert w.getnframes() == int(1.2 * RATE)
    assert sorted(p.name for p in out.iterdir()) == ["M1.wav", "M3.wav", "manifest.json"]  # no temp files left
    saved = json.loads((out / "manifest.json").read_text())
    assert saved["revision"] == "abc123" and saved["method"] == H.REBUILT and "silence" in saved["rule"]
    assert saved["meetings"]["M1"]["duration_s"] == pytest.approx(3.6) and manifest == saved
    opened.clear()
    H.build("d/x", "ihm", "abc123", ["M1", "M3"], out, open_shard=open_shard, shards=listing,
            free=lambda: 900 * 10**9, say=lambda m: None)
    assert len(opened) == len(listing)  # only the planning reads; no shard is read again for audio


def test_a_meeting_missing_from_the_data_stops_before_anything_is_written(shards, tmp_path: Path) -> None:
    listing, open_shard, _ = shards
    out = tmp_path / "audio"
    with pytest.raises(H.Stop, match=r"not in d/x ihm: \['NOPE'\]"):
        H.build("d/x", "ihm", "abc", ["M1", "NOPE"], out, open_shard=open_shard, shards=listing,
                free=lambda: 900 * 10**9, say=lambda m: None)
    assert not any(out.glob("*.wav")) if out.exists() else True


def test_the_disk_rule_stops_the_build(shards, tmp_path: Path) -> None:
    listing, open_shard, _ = shards
    plan = H.make_plan(listing, ["M1"], open_shard, say=lambda m: None)
    H.check_limits(plan, 41 * 10**9, 10**9)
    with pytest.raises(H.Stop, match="under 40 GB"):
        H.check_limits(plan, 41 * 10**9, 2 * 10**9)
    plan["shards"][0].row_groups[0] = 41 * 10**9
    with pytest.raises(H.Stop, match="over 40 GB"):
        H.check_limits(plan, 900 * 10**9, 0)
    with pytest.raises(H.Stop, match="under 40 GB"):
        H.build("d/x", "ihm", "abc", ["M1"], tmp_path / "a2", open_shard=open_shard, shards=listing,
                free=lambda: 10**9, say=lambda m: None)


def test_a_persistent_403_is_reported_not_worked_around(monkeypatch) -> None:
    import urllib.error

    calls: list[dict] = []
    waits: list[float] = []

    def urlopen(req, timeout):
        calls.append(dict(req.header_items()))
        raise urllib.error.HTTPError(req.full_url, 403, "x", {"Retry-After": "50"}, io.BytesIO())

    monkeypatch.setattr(H.urllib.request, "urlopen", urlopen)
    with pytest.raises(H.Refused, match="403"):
        H._request("https://huggingface.co/x", None, waits.append)
    assert len(calls) == 3 and waits == [50.0, 120.0]  # 30 s raised to the server's 50 s, then 120 s
    assert all(set(c) <= {"Range"} for c in calls)  # no extra headers, no cookies, no changed agent


def float_wav(value: float, seconds: float) -> bytes:
    n = int(seconds * RATE)
    body = np.full(n, value, dtype="<f4").tobytes()
    fmt = (3).to_bytes(2, "little") + (1).to_bytes(2, "little") + RATE.to_bytes(4, "little")         + (RATE * 4).to_bytes(4, "little") + (4).to_bytes(2, "little") + (32).to_bytes(2, "little")
    chunks = b"fmt " + len(fmt).to_bytes(4, "little") + fmt + b"data" + len(body).to_bytes(4, "little") + body
    return b"RIFF" + (4 + len(chunks)).to_bytes(4, "little") + b"WAVE" + chunks


def test_float_wav_utterances_are_converted_and_counted() -> None:
    samples, info = H.rebuild_meeting([H.Utterance(0.0, "A", float_wav(0.5, 0.1)), utt(1.0, "A", 100, 0.1)])
    assert (samples[:1600] == round(0.5 * 32767)).all() and (samples[16000:17600] == 100).all()
    assert info["float_utterances_converted"] == 1
    with pytest.raises(H.Stop, match="format"):
        bad = bytearray(wav(1, 0.01))
        bad[34] = 8  # 8 bits per sample
        H.decode_wav(bytes(bad))
