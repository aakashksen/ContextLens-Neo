"""Where a model runs. Models use the GPU when one is there and has room, and the CPU otherwise, saying so.

`CONTEXTLENS_DEVICE` (cpu, cuda, cuda:1) overrides the default of cuda; tests set it to cpu.
"""

import logging
import os
from collections.abc import Callable
from typing import Any

from contextlens.llm.ollama_ctl import full_name

log = logging.getLogger(__name__)

ENV = "CONTEXTLENS_DEVICE"


def pick_device(wanted: str, cuda_available: bool, say: Callable[[str], None]) -> tuple[str, str | None]:
    """(device used, fallback note). A request for cuda that cannot be met warns and runs on cpu."""
    if wanted.startswith("cuda") and not cuda_available:
        note = f"WARNING: {wanted} was requested but is not available; the model runs on CPU"
        say(note)
        return "cpu", note
    return wanted, None


def resolve_device(need_mb: int, say: Callable[[str], None] = log.warning) -> str:
    """The device a model needing about `need_mb` of GPU memory should load on; a fallback to cpu is warned about."""
    wanted = os.environ.get(ENV, "cuda")
    if not wanted.startswith("cuda"):
        return wanted
    import torch

    device, _ = pick_device(wanted, torch.cuda.is_available(), say)
    if device == "cpu":
        return device
    free_mb = torch.cuda.mem_get_info(torch.device(device))[0] / 2**20
    if free_mb < need_mb:
        say(f"WARNING: {device} has {free_mb:.0f} MB free and the model needs about {need_mb} MB; it runs on CPU")
        return "cpu"
    return device


def loaded_device(model: Any) -> str:
    """The device a loaded model reports, or "unknown"."""
    device = getattr(model, "device", None)
    return str(device) if device is not None else "unknown"


def ollama_device(ctl: Any, model: str) -> str:
    """Where Ollama holds `model` right now: gpu, cpu, partial (split), or unknown when it does not say."""
    try:
        found = ctl._call("GET", "/api/ps", None, 10.0).get("models", [])
    except Exception:
        return "unknown"
    for m in found:
        if full_name(m.get("name", "")) == full_name(model):
            size, vram = m.get("size"), m.get("size_vram")
            if not size or vram is None:
                return "unknown"
            return "gpu" if vram >= size else "cpu" if vram == 0 else "partial"
    return "unknown"
