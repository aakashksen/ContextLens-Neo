"""Fine-tuning for the consistency model: class weights, early stopping on dev macro-F1, MLflow logging."""
import contextlib
import random
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from contextlens.consistency import model as cm
from contextlens.eval.metrics import macro_f1
from contextlens.train_guard import require_finite

Log = Callable[[dict[str, float], int], None]


def class_weights(labels: list[str], order: tuple[str, ...] = cm.LABELS) -> list[float]:
    """n / (classes present * count) per label in `order`; a label that never occurs gets 0."""
    present = [x for x in order if x in labels]
    return [len(labels) / (len(present) * labels.count(x)) if x in present else 0.0 for x in order]


def collate(tokenizer: Any, examples: list[cm.Example], order: tuple[str, ...] = cm.LABELS) -> dict[str, Any]:
    import torch

    first, second = zip(*(cm.render(e) for e in examples), strict=True)
    enc = tokenizer(list(first), list(second), truncation=True, max_length=cm.MAX_LENGTH, padding=True,
                    return_tensors="pt")
    enc["labels"] = torch.tensor([order.index(e.label) for e in examples])  # type: ignore[arg-type]
    return dict(enc)


def batch_loss(model: Any, batch: dict[str, Any], weights: Any) -> Any:
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}
    labels = batch.pop("labels")
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
    with ctx:
        logits = model(**batch).logits
    return F.cross_entropy(logits.float(), labels, weight=weights.to(device))


def train_step(model: Any, batch: dict[str, Any], optimizer: Any, weights: Any, scheduler: Any = None) -> float:
    import torch

    loss = batch_loss(model, batch, weights)
    value = require_finite(float(loss.item()), "this training step")
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return value


@dataclass
class FitResult:
    best_epoch: int
    best_macro_f1: float
    history: list[dict[str, float]] = field(default_factory=list)
    seconds_per_epoch: float | None = None  # mean over the epochs run, training plus the dev scoring
    peak_gpu_memory_mb: float | None = None  # None when the model was not on a GPU: not measured, not zero


class Collator:
    """Tokenises one batch. A module-level class (not a lambda) so a spawned worker on Windows can unpickle it."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, examples: list[cm.Example]) -> dict[str, Any]:
        return collate(self.tokenizer, examples)


class EpochOrder:
    """The shuffled sample order for the epoch; replaced each epoch so persistent workers can stay alive."""

    def __init__(self) -> None:
        self.indices: list[int] = []

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def dev_macro_f1(model: Any, tokenizer: Any, dev: list[cm.Example], batch_size: int) -> float:
    probs = cm.predict_proba(model, tokenizer, dev, batch_size)
    pred = [cm.LABELS[max(range(len(p)), key=p.__getitem__)] for p in probs]
    return macro_f1([e.label for e in dev], pred) or 0.0  # type: ignore[misc]


def _run_epoch(model: Any, tokenizer: Any, loader: Any, sampler: EpochOrder, rng: random.Random, n_train: int,
               step_state: tuple[Any, Any, Any], dev: list[cm.Example], batch_size: int) -> dict[str, float]:
    """One shuffled pass of training, then the dev score; the row `fit` records for the epoch (without its number)."""
    optimizer, weights, scheduler = step_state
    started = time.perf_counter()
    model.train()
    sampler.indices = list(range(n_train))
    rng.shuffle(sampler.indices)
    losses = [train_step(model, batch, optimizer, weights, scheduler) for batch in loader]
    score = dev_macro_f1(model, tokenizer, dev, batch_size)
    return {"train_loss": sum(losses) / len(losses), "dev_macro_f1": score,
            "epoch_seconds": time.perf_counter() - started}


def _freeze_embeddings(model: Any) -> None:
    """The embedding matrix is a large share of the weights; frozen, it needs no gradient or optimizer state."""
    for p in model.get_input_embeddings().parameters():
        p.requires_grad = False


def fit(model: Any, tokenizer: Any, train: list[cm.Example], dev: list[cm.Example], *, lr: float, epochs: int,
        batch_size: int = 16, patience: int = 1, seed: int = 13, out_dir: Path | None = None,
        freeze_embeddings: bool = False, log: Log | None = None, meta: dict[str, Any] | None = None,
        num_workers: int = 0) -> FitResult:
    """Train for up to `epochs`, scoring macro-F1 on `dev` after each. The best epoch is saved to `out_dir`, and
    training stops after `patience` epochs without a better score. `dev` is used for stopping and nothing else.
    `num_workers` worker processes tokenise the next batches while the model trains. The shuffle stays in this process
    and uses the same seeded generator, so the batches, and so the result, are the same for any worker count."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    rng = random.Random(seed)
    torch.manual_seed(seed)
    if freeze_embeddings:
        _freeze_embeddings(model)
    weights = torch.tensor(class_weights([e.label for e in train]))  # type: ignore[misc]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps = epochs * ((len(train) + batch_size - 1) // batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    result = FitResult(best_epoch=0, best_macro_f1=-1.0)
    on_gpu = next(model.parameters()).device.type == "cuda"
    if on_gpu:
        torch.cuda.reset_peak_memory_stats()
    sampler = EpochOrder()
    loader = DataLoader(train, batch_size=batch_size, sampler=sampler, collate_fn=Collator(tokenizer),  # type: ignore[arg-type]
                        num_workers=num_workers, persistent_workers=num_workers > 0)
    for epoch in range(1, epochs + 1):
        row = _run_epoch(model, tokenizer, loader, sampler, rng, len(train), (optimizer, weights, scheduler), dev,
                         batch_size)
        row = {"epoch": epoch, **row}
        score = row["dev_macro_f1"]
        result.history.append(row)
        if log:
            log({k: v for k, v in row.items() if k != "epoch"}, epoch)
        if score > result.best_macro_f1:
            result.best_epoch, result.best_macro_f1 = epoch, score
            if out_dir is not None:
                cm.save_checkpoint(model, tokenizer, out_dir, {**(meta or {}), "epoch": epoch, "dev_macro_f1": score})
        elif epoch - result.best_epoch >= patience:
            break
    result.seconds_per_epoch = sum(h["epoch_seconds"] for h in result.history) / len(result.history)
    if on_gpu:
        result.peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / 2**20
        if log:
            log({"peak_gpu_memory_mb": result.peak_gpu_memory_mb}, len(result.history))
    return result


@contextlib.contextmanager
def mlflow_run(store: Path, name: str, params: dict[str, Any]) -> Iterator[Log]:
    """An MLflow run in a local file store (git-ignored mlruns/); yields a logger for metrics by step."""
    import os

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # a local file store is the point: no server, git-ignored
    import mlflow

    client = mlflow.MlflowClient(tracking_uri=store.as_uri())
    exp = client.get_experiment_by_name(name)
    exp_id = exp.experiment_id if exp else client.create_experiment(name)
    run = client.create_run(exp_id)
    for k, v in params.items():
        client.log_param(run.info.run_id, k, v)

    def log(metrics: dict[str, float], step: int) -> None:
        for k, v in metrics.items():
            client.log_metric(run.info.run_id, k, v, step=step)

    status = "FINISHED"
    try:
        yield log
    except BaseException:
        status = "FAILED"
        raise
    finally:
        client.set_terminated(run.info.run_id, status)
