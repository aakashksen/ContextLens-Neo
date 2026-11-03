"""Fine-tuning for the consistency model: class weights, early stopping on dev macro-F1, MLflow logging."""
import contextlib
import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from contextlens.consistency import model as cm
from contextlens.eval.metrics import macro_f1

Log = Callable[[dict[str, float], int], None]


def class_weights(labels: list[str], order: tuple[str, ...] = cm.LABELS) -> list[float]:
    """n / (classes present * count) per label in `order`; a label that never occurs gets 0."""
    present = [x for x in order if x in labels]
    return [len(labels) / (len(present) * labels.count(x)) if x in present else 0.0 for x in order]


def collate(tokenizer: Any, examples: list[cm.Example], order: tuple[str, ...] = cm.LABELS) -> dict[str, Any]:
    import torch

    first, second = zip(*(cm.render(e) for e in examples))
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
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return float(loss.item())


@dataclass
class FitResult:
    best_epoch: int
    best_macro_f1: float
    history: list[dict[str, float]] = field(default_factory=list)


def dev_macro_f1(model: Any, tokenizer: Any, dev: list[cm.Example], batch_size: int) -> float:
    probs = cm.predict_proba(model, tokenizer, dev, batch_size)
    pred = [cm.LABELS[max(range(len(p)), key=p.__getitem__)] for p in probs]
    return macro_f1([e.label for e in dev], pred) or 0.0  # type: ignore[misc]


def fit(model: Any, tokenizer: Any, train: list[cm.Example], dev: list[cm.Example], *, lr: float, epochs: int,
        batch_size: int = 16, patience: int = 1, seed: int = 13, out_dir: Path | None = None,
        freeze_embeddings: bool = False, log: Log | None = None, meta: dict[str, Any] | None = None) -> FitResult:
    """Train for up to `epochs`, scoring macro-F1 on `dev` after each. The best epoch is saved to `out_dir`, and
    training stops after `patience` epochs without a better score. `dev` is used for stopping and nothing else."""
    import torch
    from transformers import get_linear_schedule_with_warmup

    rng = random.Random(seed)
    torch.manual_seed(seed)
    if freeze_embeddings:  # the embedding matrix is a large share of the weights; frozen, it needs no gradient or optimizer state
        for p in model.get_input_embeddings().parameters():
            p.requires_grad = False
    weights = torch.tensor(class_weights([e.label for e in train]))  # type: ignore[misc]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps = epochs * ((len(train) + batch_size - 1) // batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    result = FitResult(best_epoch=0, best_macro_f1=-1.0)
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(train)
        rng.shuffle(order)
        losses = [train_step(model, collate(tokenizer, order[i : i + batch_size]), optimizer, weights, scheduler)
                  for i in range(0, len(order), batch_size)]
        score = dev_macro_f1(model, tokenizer, dev, batch_size)
        row = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "dev_macro_f1": score}
        result.history.append(row)
        if log:
            log({k: v for k, v in row.items() if k != "epoch"}, epoch)
        if score > result.best_macro_f1:
            result.best_epoch, result.best_macro_f1 = epoch, score
            if out_dir is not None:
                cm.save_checkpoint(model, tokenizer, out_dir, {**(meta or {}), "epoch": epoch, "dev_macro_f1": score})
        elif epoch - result.best_epoch >= patience:
            break
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
