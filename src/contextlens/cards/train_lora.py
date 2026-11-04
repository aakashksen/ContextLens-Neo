"""QLoRA training for the card model. Nothing here runs at import; the real path loads a model only in `train`."""
from __future__ import annotations

import errno
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests

from contextlens.cards.dataset import CARDS_DIR, Example, load_examples

if TYPE_CHECKING:
    import torch

PRIMARY = "Qwen/Qwen3.5-4B"
FALLBACK = "Qwen/Qwen3-4B"
VRAM_LIMIT_GB = 7.0
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def tokenize_example(tok: Any, ex: Example, max_len: int) -> dict[str, list[int]] | None:
    """Prompt tokens get label -100 so loss is only on the card. None when it does not fit in max_len."""
    prompt = tok.apply_chat_template(ex.messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    c = tok(ex.completion + tok.eos_token, add_special_tokens=False)["input_ids"]
    if len(p) + len(c) > max_len:
        return None
    return {"input_ids": p + c, "labels": [-100] * len(p) + c}


def collate(rows: list[dict[str, list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    import torch

    n = max(len(r["input_ids"]) for r in rows)
    pad = [n - len(r["input_ids"]) for r in rows]
    return {
        "input_ids": torch.tensor([r["input_ids"] + [pad_id] * k for r, k in zip(rows, pad, strict=True)]),
        "labels": torch.tensor([r["labels"] + [-100] * k for r, k in zip(rows, pad, strict=True)]),
        "attention_mask": torch.tensor([[1] * len(r["input_ids"]) + [0] * k for r, k in zip(rows, pad, strict=True)]),
    }


def attach_lora(model: Any, r: int = 16, alpha: int = 32, dropout: float = 0.05) -> Any:
    from peft import LoraConfig, get_peft_model

    return get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=TARGETS,
                                            task_type="CAUSAL_LM"))


def check_vram(limit_gb: float = VRAM_LIMIT_GB) -> float:
    """Peak GPU memory so far in GB; raises when over the limit. 0.0 without CUDA."""
    import torch

    if not torch.cuda.is_available():
        return 0.0
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    if peak_gb > limit_gb:
        raise RuntimeError(f"blocked: peak {peak_gb:.2f} GB is over the {limit_gb} GB limit")
    return peak_gb


def is_architecture_error(e: BaseException) -> bool:
    """True for failures a model class PEFT or the loader cannot handle. Out-of-memory, a missing package, a network
    or permission problem, or a full disk are not architecture problems, so they are never answered by switching model."""
    if isinstance(e, requests.exceptions.RequestException):  # a requests error is an OSError, but a network one
        return False
    if isinstance(e, OSError):
        return not isinstance(e, (ConnectionError, TimeoutError, PermissionError)) and e.errno != errno.ENOSPC
    return isinstance(e, (ValueError, KeyError, TypeError, AttributeError))


def run_steps(model: Any, tok: Any, examples: list[Example], max_len: int, steps: int, batch_size: int = 2,
              lr: float = 2e-4, vram_limit_gb: float | None = None) -> list[float]:
    """Plain loop: AdamW over the LoRA parameters only. Returns the loss at each step. With `vram_limit_gb`, the
    peak memory is checked after every step, so a run over the limit stops at the first step."""
    import torch

    rows = [r for e in examples if (r := tokenize_example(tok, e, max_len)) is not None]
    if not rows:
        raise ValueError("no example fits in max_len")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    model.train()
    losses = []
    for step in range(steps):
        batch = collate([rows[(step * batch_size + k) % len(rows)] for k in range(batch_size)], tok.pad_token_id)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        loss = model(**batch).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
        if vram_limit_gb is not None:
            check_vram(vram_limit_gb)
    return losses


def _load_4bit(model_id: str) -> Any:
    import torch
    from peft import prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant, device_map={"": 0})
    return prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)


def train(out_dir: Path, steps: int, max_len: int = 2048, cards_dir: Path = CARDS_DIR) -> dict[str, Any]:
    """Needs a GPU turn. Tries PRIMARY; if loading or PEFT fails with an architecture error (see
    is_architecture_error), uses FALLBACK and records the exact error in train_meta.json; any other failure is
    raised. Exceeding the VRAM limit raises at the first step over it: the model class is not changed silently."""
    from transformers import AutoTokenizer

    import torch

    base, reason = PRIMARY, None
    try:
        model = attach_lora(_load_4bit(base))
    except Exception as e:
        if not is_architecture_error(e):
            raise
        base, reason = FALLBACK, f"{type(e).__name__}: {e}"
        torch.cuda.empty_cache()
        model = attach_lora(_load_4bit(base))
    tok = AutoTokenizer.from_pretrained(base)
    losses = run_steps(model, tok, load_examples("train", cards_dir), max_len, steps,
                       vram_limit_gb=VRAM_LIMIT_GB)
    peak_gb = check_vram()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    meta = {"base_model": base, "fallback_used": base == FALLBACK, "fallback_reason": reason,
            "peak_vram_gb": round(peak_gb, 3), "steps": steps, "final_loss": losses[-1]}
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return meta
