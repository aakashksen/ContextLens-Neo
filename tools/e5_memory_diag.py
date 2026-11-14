"""Measures peak GPU memory and seconds per step for card-LoRA training at several input lengths, with and without the
two memory savers (batch 1 with accumulation; loss on the card's tokens only). Needs a GPU turn. Padding and truncation
happen only here, on copies of real training rows, to reach an exact length; training never shortens an example.

    python tools/e5_memory_diag.py [out.json [length,length,...]]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from contextlens.cards.dataset import load_examples
from contextlens.cards.train_lora import PRIMARY, _load_4bit, attach_lora, tokenize_example

LENGTHS = (512, 1024, 1536, 2048)
STEPS = 3  # the first is warm-up and is not timed
BATCH_TOTAL = 2  # what E5 trains with now: batch 2, no accumulation


def row_of_length(rows: list[dict], n: int) -> dict:
    """The real row closest in length to n, with its prompt part repeated or cut (from the front) to make exactly n."""
    r = min(rows, key=lambda x: abs(len(x["input_ids"]) - n))
    k = sum(1 for lab in r["labels"] if lab != -100)
    p = r["input_ids"][:-k]
    p = (p * (n // len(p) + 1))[-(n - k):] if len(p) != n - k else p
    return {"input_ids": p + r["input_ids"][-k:], "labels": [-100] * len(p) + r["input_ids"][-k:]}


def step_loss(model, ids: torch.Tensor, labels: torch.Tensor, slice_logits: bool) -> torch.Tensor:
    if not slice_logits:
        return model(input_ids=ids, labels=labels).loss
    keep = int((labels != -100).sum(dim=1).max()) + 1
    logits = model(input_ids=ids, logits_to_keep=keep).logits[:, :-1].float()
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels[:, -keep + 1:].reshape(-1), ignore_index=-100)


def measure(model, row: dict, n_per_micro: int, accum: int, slice_logits: bool) -> dict:
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4)
    ids = torch.tensor([row["input_ids"]] * n_per_micro).cuda()
    labels = torch.tensor([row["labels"]] * n_per_micro).cuda()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for s in range(STEPS):
        t = time.perf_counter()
        for _ in range(accum):
            (step_loss(model, ids, labels, slice_logits) / accum).backward()
        opt.step()
        opt.zero_grad()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    peak = torch.cuda.max_memory_allocated() / 1024**3
    del opt
    torch.cuda.empty_cache()
    return {"peak_gb": round(peak, 2), "s_per_step": round(sum(times[1:]) / (STEPS - 1), 2)}


def main() -> None:
    tok = AutoTokenizer.from_pretrained(PRIMARY)
    examples = load_examples("train", Path(__file__).resolve().parents[1] / "datasets" / "cards", include_scenarios=True)
    rows = [r for e in examples if (r := tokenize_example(tok, e, 10**6)) is not None]
    model = attach_lora(_load_4bit(PRIMARY))
    model.train()
    out = []
    for n in [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else LENGTHS:
        row = row_of_length(rows, n)
        for name, micro, accum, sl in (("baseline b2", 2, 1, False), ("A b1 accum2", 1, 2, False),
                                       ("B slice b2", 2, 1, True), ("A+B b1 accum2 slice", 1, 2, True)):
            try:
                res = measure(model, row, micro, accum, sl)
            except torch.OutOfMemoryError:
                res = {"peak_gb": "OOM", "s_per_step": None}
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            out.append({"length": n, "savers": name, **res})
            print(out[-1], flush=True)
            if len(sys.argv) > 1:
                Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
