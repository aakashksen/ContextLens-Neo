"""A diagnostic, not an E7 record: is E7's word error rate about Whisper `tiny.en`, the rebuilt audio, or 4x streaming?

    python -m contextlens.experiments.e7_wer_control --record E7-<run id> --meetings 5 --models tiny.en base.en

For the shortest N meetings, the audio up to the meeting's last question (the range E7 streamed) is transcribed offline,
whole, by Whisper and compared with the same reference text E7 uses. The streamed figure for the same range is read from
the saved `asr_pipeline` row of that question (no row is regenerated). Reading the arms:

- streamed vs offline `tiny.en`: what streaming, the service's speech gate and whisper.cpp cost (or save) against an
  offline run of the same model size on the same audio;
- offline `tiny.en` vs `base.en`: what the model size costs. `base.en` is reported as its own measurement and is never
  E7's pipeline (E7's settings are not changed to chase a number);
- `overlap_share`: the share of reference words spoken while another speaker also speaks. The rebuilt audio sums
  different speakers' headset tracks, so overlap is mixed into one signal.

The output goes to results/local/ (git-ignored), labelled a diagnostic. Offline Whisper here is the Hugging Face
transformers implementation of OpenAI's weights (safetensors, no remote code), not the whisper.cpp the service runs."""

import argparse
import json
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from contextlens.eval import metrics as M
from contextlens.eval.records import ROOT
from contextlens.eval.series import reference_text
from contextlens.experiments.e7_realtime import AUDIO_REL, _load_dev, cut_ms, meetings_needed, read_wav, shortest_meetings
from contextlens.replay.runner import Prediction
from contextlens.schemas import Scenario, Turn
from contextlens.speech.diarize import BYTES_PER_MS

CACHE_REL = Path("models/whisper-control")  # git-ignored
REVISIONS = {"tiny.en": ("openai/whisper-tiny.en", "87c7102498dcde7456f24cfd30239ca606ed9063"),
             "base.en": ("openai/whisper-base.en", "911407f4214e0e1d82085af863093ec0b66f9cd6")}
LIMITS = ("Offline here is the Hugging Face transformers implementation of OpenAI's weights, not whisper.cpp (ggml) as the "
          "speech service runs; differences between the streamed and offline figures include that implementation.",
          "The original AMI recordings are not available, so the rebuild itself (per-speaker headset tracks summed, gaps as "
          "silence) cannot be separated from the audio's difficulty; overlap_share only shows how much overlap there is.",
          "Five meetings, one pass; no interval is computed.")


class Offline(Protocol):
    def __call__(self, audio: np.ndarray) -> str: ...


def _overlap_share(turns: list[Turn]) -> float:
    words = overlapped = 0
    for t in turns:
        n = len(t.text.split())
        words += n
        overlapped += n * any(o.speaker != t.speaker and o.start_ms < t.end_ms and t.start_ms < o.end_ms for o in turns)
    return overlapped / words if words else 0.0


def control(scenarios: list[Scenario], rows: dict[str, Prediction], audio_dir: Path, meetings: list[str],
            models: dict[str, Callable[[], Offline]], say: Callable[[str], None] = print) -> dict[str, Any]:
    """Per meeting, the streamed WER (from the saved row of its last question) and each model's offline WER."""
    last = {m: max((s for s in scenarios if s.conversation.id == m), key=lambda s: s.request.as_of_seq) for m in meetings}
    refs = {m: reference_text(s) for m, s in last.items()}
    per: dict[str, dict[str, Any]] = {}
    for m, sc in last.items():
        row = rows[sc.id]
        if row.asr_text is None:
            raise ValueError(f"{sc.id} has no asr_text; use an E7 record's asr_pipeline rows")
        per[m] = {"scenario": sc.id, "cut_s": cut_ms(sc) / 1000, "reference_words": len(refs[m].split()),
                  "streamed_wer": M.word_error_rate([refs[m]], [row.asr_text]), "offline_wer": {}}
    offline_text: dict[str, dict[str, str]] = {}
    for name, factory in models.items():
        say(f"loading {name}")
        model = factory()
        try:
            offline_text[name] = {}
            for m, sc in last.items():
                pcm = read_wav(audio_dir / f"{m}.wav")[: cut_ms(sc) * BYTES_PER_MS]
                audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                offline_text[name][m] = model(audio)
                per[m]["offline_wer"][name] = M.word_error_rate([refs[m]], [offline_text[name][m]])
                say(f"  {name} {m}: streamed {per[m]['streamed_wer']:.3f}, offline {per[m]['offline_wer'][name]:.3f}")
        finally:
            close = getattr(model, "close", None)
            if close:
                close()
    ms = list(last)
    totals = {"streamed_wer": M.word_error_rate([refs[m] for m in ms], [rows[last[m].id].asr_text or "" for m in ms]),
              "offline_wer": {n: M.word_error_rate([refs[m] for m in ms], [offline_text[n][m] for m in ms]) for n in models}}
    return {"meetings": ms, "per_meeting": per, "totals": totals,
            "overlap_share": {m: _overlap_share([t for t in s.conversation.turns if t.seq <= s.request.as_of_seq])
                              for m, s in last.items()}}


def write_diagnostic(result: dict[str, Any], root: Path, record_id: str) -> Path:
    doc = {"diagnostic": True, "not_an_e7_record": True, "e7_record": record_id,
           "note": "A diagnostic of E7's word error rate. base.en is its own measurement, never E7's pipeline; E7's settings "
                   "(tiny.en) are unchanged by it. Not a run record and not for claims.json.",
           "limits": list(LIMITS), "models": {k: {"repo": v[0], "revision": v[1]} for k, v in REVISIONS.items()},
           "written_at": datetime.now(timezone.utc).isoformat(), **result}
    path = root / "results/local" / f"e7-wer-control-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return path


class HFWhisper:
    """Offline Whisper from the pinned, locally cached safetensors; unloaded on close."""

    def __init__(self, name: str, device: str, root: Path) -> None:
        import torch
        from transformers import pipeline

        repo, revision = REVISIONS[name]
        self._torch = torch
        self.pipe = pipeline("automatic-speech-recognition", model=repo, revision=revision, device=device,
                             dtype=torch.float16 if device.startswith("cuda") else torch.float32, chunk_length_s=30,
                             batch_size=8, model_kwargs={"use_safetensors": True, "cache_dir": str(root / CACHE_REL)})

    def __call__(self, audio: np.ndarray) -> str:
        return self.pipe({"raw": audio, "sampling_rate": 16000})["text"]

    def close(self) -> None:
        del self.pipe
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="WER control: offline Whisper on E7's rebuilt audio (a diagnostic)")
    ap.add_argument("--record", required=True, help="the E7 record whose asr_pipeline rows give the streamed WER")
    ap.add_argument("--meetings", type=int, default=5, help="the N shortest meetings")
    ap.add_argument("--models", nargs="+", default=["tiny.en", "base.en"], choices=sorted(REVISIONS))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    rec = json.loads((ROOT / "results/runs" / f"{a.record}.json").read_text(encoding="utf-8"))
    path = ROOT / rec["predictions"]["asr_pipeline"]
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = {p.scenario_id: p for p in (Prediction.model_validate_json(x) for x in lines if x.strip())}
    scenarios, _ = _load_dev(ROOT)
    audio = ROOT / AUDIO_REL
    meetings = shortest_meetings(audio, meetings_needed(), a.meetings)
    factories = {n: (lambda n=n: HFWhisper(n, a.device, ROOT)) for n in a.models}
    result = control(scenarios, rows, audio, meetings, factories)
    out = write_diagnostic(result, ROOT, a.record)
    print(out.as_posix())
    print("streamed", result["totals"]["streamed_wer"], "offline", result["totals"]["offline_wer"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
