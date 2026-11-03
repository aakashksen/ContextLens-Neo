"""When pyannote cannot load, the backend says what the person must do, and never shows the token."""

import pytest

from contextlens.speech.diarize import PyannoteBackend


def test_load_failure_names_the_gated_repo_requirement(monkeypatch):
    pipeline = pytest.importorskip("pyannote.audio").Pipeline
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")

    def refuse(*args, **kwargs):
        raise OSError(f"403 for community-1 with {kwargs}")

    monkeypatch.setattr(pipeline, "from_pretrained", refuse)
    with pytest.raises(RuntimeError, match="speaker-diarization-community-1") as err:
        PyannoteBackend()
    assert "hf_secret_value" not in str(err.value) and "hf_secret_value" not in repr(err.value.__cause__)
