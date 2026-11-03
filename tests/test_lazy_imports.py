"""Light modules must not pull in the heavy ML libraries: checks, tests and the API stay quick to start."""
import subprocess
import sys

import pytest

HEAVY = ("torch", "transformers", "sentence_transformers", "faster_whisper", "pyannote")
LIGHT = [
    "contextlens.eval.check_run",
    "contextlens.schemas",
    "contextlens.llm.client",
    "contextlens.cards.train_lora",
    "contextlens.cards.dataset",
    "contextlens.consistency.model",
    "contextlens.consistency.train",
]


@pytest.mark.parametrize("module", LIGHT)
def test_importing_does_not_load_heavy_libraries(module):
    code = (
        f"import sys, {module}\n"
        f"bad = sorted(m for m in {HEAVY!r} if m in sys.modules)\n"
        "print(','.join(bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-500:]
    assert out.stdout.strip() == "", f"{module} loaded {out.stdout.strip()}"
