"""E4: does a trained consistency model beat TF-IDF, zero-shot and prompted baselines?

Four systems on the dev split only, in two records: our dev scenarios (four labels, cards, with the model's
probability so calibration is scored) and a fixed sample of DECODE's dev conversations (contradiction or not,
ids only, no DECODE text). The trained model is the best of a small search over learning rate and epochs, picked by a
rule fixed here (`pick_winner`), never by a test split, which this module does not read.

Training data: DECODE's train split (a fixed sample) plus the train-only scenario set at datasets/scenarios_train/.
That set is in our own writing style, so scores on our scenarios could look better than real use. Dev scenarios are used
for early stopping and for picking the winner, so they are selection data, not a held-out score.

Run: .venv/Scripts/python -m contextlens.experiments.e4_consistency --prepare   (once; then commit the manifest)
     .venv/Scripts/python -m contextlens.experiments.e4_consistency
"""
import argparse
import gc
import hashlib
import json
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contextlens.agents.consistency_check import ConsistencyChecker, PromptedChecker
from contextlens.consistency import model as cm
from contextlens.consistency import train as ct
from contextlens.consistency.baselines import ZERO_SHOT_MODEL, TfidfChecker, ZeroShotChecker
from contextlens.consistency.data import assert_trainable, build_pairs
from contextlens.data import train_scenarios as ts
from contextlens.data.loaders import Pair
from contextlens.data.scenarios import load_scenarios
from contextlens.eval.check_run import committed_unchanged, recompute_comparison, recompute_intervals, recompute_metrics
from contextlens.eval.records import load_verified, ROOT, SCENARIO_SOURCES, head_commit, write_local_manifest, write_run
from contextlens.llm.client import unload_ollama
from contextlens.eval.series import F1_LABELS, base_name, predicted_label
from contextlens.replay.runner import Prediction
from contextlens.schemas import ButtonRequest, Card, Conversation, Finding, Scenario, Turn

SPLIT = "dev"
SOURCES = ("scenarios", "decode")
GRID = tuple((lr, epochs) for lr in (1e-5, 2e-5, 3e-5) for epochs in (2, 3))  # 6 runs, the most the plan allows
BATCH_SIZE = 8
MIN_BATCH_SIZE = 2  # out of memory halves the batch size down to this; below it the setting is recorded as failed
NUM_WORKERS = 2  # tokenising worker processes; the model step is the slow part, so two keep the GPU fed without crowding the CPU
PATIENCE = 1
DECODE_TRAIN_N = 8000  # DECODE train pairs used, a fixed sample
DECODE_DEV_N = 1000  # DECODE dev pairs scored, a fixed sample
SCENARIO_REPEAT = 8  # each train-only scenario pair is seen this many times per epoch
PROMPTED_MODEL = "qwen3.5:4b-q4_K_M"  # docs/model-choice.md
RAW_DIR = Path("datasets/raw")
CHECKPOINTS = Path("models/e4")  # git-ignored
MLRUNS = Path("mlruns")  # git-ignored
ERROR_ANALYSIS_REL = Path("results/error-analysis-e4.md")
ERRORS_SHOWN = 50
SYSTEMS = ("trained", "tfidf", "zeroshot", "prompted")
SAVED = {"macro_f1", "false_alert_rate", "label_accuracy", "ece", "latency_p50_ms", "latency_p95_ms",
         "abstain_when_insufficient", "answer_when_answerable"}
WINNER_RULE = ("highest dev macro-F1 over the four labels on our dev scenarios (the best epoch of the run); ties broken "
               "by fewer epochs, then by lower learning rate")
SAMPLE_RULE = ("pairs ordered by sha256 of conversation_id|label|statement, the first N taken; the same rule picks the "
               "DECODE train sample and the DECODE dev sample")
SELECTION_NOTE = "dev scenarios (early stopping and winner), so dev scores are not held-out"
STYLE_NOTE = ("training used our own writing style (the train-only scenario set was written for this project, like the "
              "dev and test scenarios), so scores on our scenarios could look better than real use")
TRAIN_SET_REVIEW_NOTE = ("an independent read of the invented train-only set, before training, caught real near-copies of "
                         "held-out scenarios; the 3-word overlap guard scored them only 0.13 to 0.18, so it could not "
                         "have caught them; those cases were rewritten and read again")
BATCH_NOTE = ("on out of memory a setting is retried at half the batch size (floor MIN_BATCH_SIZE). A smaller batch means more "
              "optimizer steps per epoch at the same learning rate, so it can change the dev score; the winner rule does not "
              "adjust for that, and `settings_at_lower_batch_size` lists the settings whose score is not like for like")
RERUN_REASON = ("the first run's training loss was NaN in every epoch of all six settings, so its trained model was "
                "untrained and its records are void; the weights had loaded as float16 and AdamW's epsilon underflowed")
RERUN_NOTE = ("rerun after a bug: pretrained weights were loaded in float16 (the stored dtype), the first AdamW step made "
              "them NaN, and every run was recorded ok. Fix: weights load as float32 (bf16 autocast on GPU is unchanged), a "
              "non-finite loss now fails its setting, and check_run rejects a winner with a non-finite training loss. "
              "The grid, winner rule, data and manifests are unchanged; the fix changes the weight precision of every "
              "trained setting, so the first run's numbers cannot be compared with this one's")
SMOKE_STEPS = 24
SMOKE_BATCHES = 4  # distinct real batches, cycled, so a falling loss means the model learns them
SMOKE_LR = 3e-5
DECODE_NOTE = ("DECODE is binary: a predicted label other than contradiction counts as non-contradiction. Its score "
               "covers contradiction vs not only and is not four-label performance. Metrics are accuracy, false alert "
               "rate and the F1 of contradiction and of non-contradiction; the four-label macro-F1 is not saved for it. "
               "DECODE dev was used for no training; DECODE test is untouched.")
DECODE_CAVEAT = ("DECODE's score covers contradiction vs not only and is not four-label performance; "
                 "its cases appear here by id only.")
BALANCE_NOTE = ("class weights change how much a label counts in the loss but add no new examples: `update` and "
                "`insufficient` rest on very few distinct train-only cases (each repeated `repeats` times), so scores on "
                "them should be read with that in mind")


class SystemFailed(RuntimeError):
    pass


def _rank(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def sample_pairs(pairs: list[Pair], n: int) -> list[Pair]:
    return sorted(pairs, key=lambda p: _rank(p.conversation_id, p.label, p.statement))[:n]


def _speaker(m: int, j: int) -> str:
    return "A" if (m - j) % 2 == 0 else "B"


def decode_scenario(p: Pair) -> Scenario:
    """A DECODE dev pair as a scenario: the statement is the last turn. The text stays in git-ignored datasets/raw/."""
    sid = f"decode-dev-{_rank(p.conversation_id, p.label, p.statement)[:12]}"
    texts = [*p.earlier, p.statement]
    m = len(p.earlier)
    turns = [Turn(id=f"{sid}:t{i}", conversation_id=p.conversation_id, seq=i, speaker=_speaker(m, i), text=t,
                  start_ms=i * 1000, end_ms=i * 1000 + 900, source="dataset") for i, t in enumerate(texts)]
    return Scenario(
        id=sid, conversation=Conversation(id=p.conversation_id, turns=turns, source="decode", split="dev"),
        request=ButtonRequest(conversation_id=p.conversation_id, as_of_seq=m), category="decode",
        gold_evidence_turn_ids=[], gold_label=p.label, gold_answer=None, origin="dataset")  # type: ignore[arg-type]


def prepare_decode_dev(raw_root: Path = RAW_DIR, root: Path = ROOT, n: int = DECODE_DEV_N) -> int:
    """Write the DECODE dev sample (git-ignored) and its manifest of hash, count and ids (commit it before the run)."""
    pairs = [p for p in build_pairs(SPLIT, raw_root, root / "datasets/none") if p.source == "decode"]
    scs = [decode_scenario(p) for p in sample_pairs(pairs, n)]
    write_local_manifest("decode", {SPLIT: scs}, root)
    return len(scs)


@dataclass
class TrainingData:
    examples: list[cm.Example]
    pairs: list[Pair]  # the same data as pairs, for the TF-IDF baseline
    counts: dict[str, int]
    balance: dict[str, Any]


def training_data(raw_root: Path, root: Path, dev_scenarios: list[Scenario]) -> TrainingData:
    """DECODE train sample plus the train-only scenarios. Stops if anything of dev reaches training."""
    mine = ts.load_train_scenarios(root / ts.TRAIN_DIR)
    if mine != ts.build_train_scenarios():
        raise RuntimeError("datasets/scenarios_train/train.jsonl is not what the code builds; rebuild and commit it")
    clash = ts.overlap_problems(mine, dev_scenarios + ts.held_out_present(root / SCENARIO_SOURCES["scenarios"][1]))
    if clash:
        raise RuntimeError("the train-only scenarios share text with dev or test: " + "; ".join(clash))
    got = build_pairs("train", raw_root, root / ts.TRAIN_DIR)
    if any(p.split != "train" for p in got):
        raise RuntimeError("a non-train pair reached training")
    decode = sample_pairs([p for p in got if p.source == "decode"], DECODE_TRAIN_N)
    own = [p for p in got if p.source == "scenarios"]
    assert_trainable(decode + own, raw_root, root / SCENARIO_SOURCES["scenarios"][1])
    pairs = decode + own * SCENARIO_REPEAT
    examples = cm.examples_from_pairs(decode) + cm.examples_from_scenarios(mine, "scenarios_train") * SCENARIO_REPEAT
    labels = [e.label for e in examples]
    weights = dict(zip(cm.LABELS, ct.class_weights(labels), strict=True))
    positive = [w for w in weights.values() if w > 0]
    balance = {"distinct_cases_per_label": {x: sum(s.gold_label == x for s in mine) for x in cm.LABELS},
               "repeats": SCENARIO_REPEAT, "class_weights": weights, "weight_ratio": max(positive) / min(positive),
               "note": BALANCE_NOTE}
    return TrainingData(examples, pairs, {
        "decode_train_pairs": len(decode), "scenarios_train_pairs": len(own), "scenario_repeat": SCENARIO_REPEAT,
        "examples": len(examples), **{f"label_{k}": v for k, v in sorted(Counter(e.label for e in examples).items())}}, balance)


def pick_winner(attempts: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The finished attempt that WINNER_RULE picks; None when none finished."""
    done = [(i, a) for i, a in enumerate(attempts) if a["ok"]]
    return min(done, key=lambda ia: (-ia[1]["dev_macro_f1"], ia[1]["epochs"], ia[1]["lr"], ia[0]))[1] if done else None


def _free() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _say(text: str) -> None:
    print(text, flush=True)


def is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"


def _batch_sizes() -> list[int]:
    sizes, n = [], BATCH_SIZE
    while n >= MIN_BATCH_SIZE:
        sizes.append(n)
        n //= 2
    return sizes


def search(base_model: str, data: TrainingData, dev: list[cm.Example], *, device: str, root: Path, grid=GRID,
           say=_say, num_workers: int = 0) -> list[dict[str, Any]]:
    """Every setting of the grid, in order. A run that errors is listed with its error and the search goes on. Out of
    memory is not a failure at first: the setting is retried from the pretrained weights at half the batch size, down to
    MIN_BATCH_SIZE, and each try is listed in `oom_retries`. `batch_size` is the one the result came from."""
    attempts = []
    reported = False
    for lr, epochs in grid:
        name = f"lr{lr:g}_ep{epochs}"
        out = root / CHECKPOINTS / name
        row: dict[str, Any] = {"setting": name, "lr": lr, "epochs": epochs, "ok": False, "error": None,
                               "best_epoch": None, "dev_macro_f1": None, "history": [], "checkpoint": out.relative_to(root).as_posix(),
                               "batch_size": BATCH_SIZE, "lowered_batch_size": False, "oom_retries": [],
                               "device": device, "seconds_per_epoch": None, "peak_gpu_memory_mb": None,
                               "peak_gpu_memory_note": None}
        say(f"training {name}")
        for bs in _batch_sizes():
            row["batch_size"] = bs
            row["lowered_batch_size"] = bs != BATCH_SIZE
            model = tokenizer = None
            oom = None
            try:
                model, tokenizer = cm.load_pretrained(base_model)
                model.to(device)
                params = {"lr": lr, "epochs": epochs, "batch_size": bs, "base_model": base_model, "device": device,
                          "num_workers": num_workers, **data.counts}
                with ct.mlflow_run(root / MLRUNS, "E4", {"setting": name, **params}) as log:
                    res = ct.fit(model, tokenizer, data.examples, dev, lr=lr, epochs=epochs, batch_size=bs,
                                 patience=PATIENCE, out_dir=out, freeze_embeddings=True, log=log, meta=params,
                                 num_workers=num_workers)
                row.update(ok=True, error=None, best_epoch=res.best_epoch, dev_macro_f1=res.best_macro_f1,
                           history=res.history, seconds_per_epoch=res.seconds_per_epoch,
                           peak_gpu_memory_mb=res.peak_gpu_memory_mb,
                           peak_gpu_memory_note=None if res.peak_gpu_memory_mb is not None else "not measured: no GPU")
                say(f"  best epoch {res.best_epoch}, dev macro-F1 {res.best_macro_f1:.4f}, batch size {bs}")
            except Exception as exc:  # the setting is recorded as failed and the search goes on
                row["error"] = f"{type(exc).__name__}: {exc}"
                if is_oom(exc):
                    oom = row["error"]
                    row["oom_retries"].append({"batch_size": bs, "error": oom})
                say(f"  failed at batch size {bs}: {row['error']}")
            finally:
                del model, tokenizer  # after the except block, so the exception no longer holds the model's frames
                _free()
            if row["ok"] or oom is None:
                break
            if bs // 2 >= MIN_BATCH_SIZE:
                say(f"  out of memory: retrying at batch size {bs // 2}")
            else:
                row["error"] = f"{oom} (batch size {bs} is the floor of {MIN_BATCH_SIZE})"
        attempts.append(row)
        if row["ok"] and not reported:
            reported = True
            peak = row["peak_gpu_memory_note"] or f"peak GPU memory {row['peak_gpu_memory_mb']:.0f} MB"
            say(f"FIRST SEARCH RUN DONE {name}: {row['seconds_per_epoch']:.1f} s/epoch, {peak}, batch size {row['batch_size']}")
    return attempts


def _card(label: str, confidence: float, statement: str) -> Card:
    if label == "insufficient":
        return Card(mode="clarify", status="abstained", findings=[], missing=["not enough information"])
    return Card(mode="clarify", status="answered", findings=[Finding(
        agent="consistency", kind="consistency", statement=statement, evidence=[], label=label,  # type: ignore[arg-type]
        confidence=confidence)])


def _judged(s: Scenario) -> tuple[Turn, list[Turn]]:
    turns = sorted((t for t in s.conversation.turns if t.seq <= s.request.as_of_seq), key=lambda t: t.seq)
    return turns[-1], turns[:-1]


def predict_scenarios(checker: ConsistencyChecker, scenarios: list[Scenario]) -> list[Prediction]:
    out = []
    for s in scenarios:
        statement, earlier = _judged(s)
        start = time.perf_counter()
        label, conf = checker.check(statement, earlier)
        ms = int((time.perf_counter() - start) * 1000)
        out.append(Prediction(scenario_id=s.id, card=_card(label, conf, statement.text), probability=min(1.0, max(0.0, conf)),
                              stage_latency_ms={"total": ms}))
    return out


def predict_decode(checker: ConsistencyChecker, scenarios: list[Scenario]) -> list[Prediction]:
    """Ids and a bare label only; any label but contradiction is non-contradiction (`consistent`)."""
    out = []
    for s in scenarios:
        statement, earlier = _judged(s)
        start = time.perf_counter()
        label, _ = checker.check(statement, earlier)
        ms = int((time.perf_counter() - start) * 1000)
        out.append(Prediction(scenario_id=s.id, label="contradiction" if label == "contradiction" else "consistent",
                              stage_latency_ms={"total": ms}))
    return out


def evaluate_system(checker: ConsistencyChecker, data: dict[str, list[Scenario]]) -> dict[str, list[Prediction]]:
    """Predictions on every source, or SystemFailed: a system that fails anywhere is out of both records."""
    try:
        return {"scenarios": predict_scenarios(checker, data["scenarios"]),
                "decode": predict_decode(checker, data["decode"])}
    except Exception as exc:
        raise SystemFailed(f"{type(exc).__name__}: {exc}") from exc


def warm_up(checker: ConsistencyChecker, scenario: Scenario, tries: int = 3) -> None:
    """The first call to a local server loads its model and can time out; later calls are the ones that count."""
    statement, earlier = _judged(scenario)
    for _ in range(tries):
        try:
            checker.check(statement, earlier)
            return
        except Exception:
            time.sleep(2)


def _saved(fresh: dict[str, float | None], src: str) -> dict[str, float | None]:
    """The metrics a record keeps. Our scenarios keep the four-label set and accuracy per category; DECODE is binary,
    so it keeps accuracy, false alert rate and latency only."""
    if src == "decode":
        return {k: v for k, v in fresh.items() if k in {"false_alert_rate", "label_accuracy", "latency_p50_ms", "latency_p95_ms",
                                                         "f1[label=contradiction]", "f1[label=consistent]"}}
    return {k: v for k, v in fresh.items()
            if (base_name(k) in SAVED and "[" not in k) or k in F1_LABELS or k.startswith("label_accuracy[category=")}


def error_analysis(scenarios: dict[str, list[Scenario]], preds: dict[str, list[Prediction]], limit: int = ERRORS_SHOWN) -> str:
    """Markdown on the trained model's mistakes. Which mistakes are shown is fixed: take categories in alphabetical order,
    one mistake at a time from each (by scenario id), until `limit`. Our own scenario text is quoted; DECODE
    mistakes are shown by id only."""
    wrong: dict[str, list[tuple[Scenario, Prediction, str]]] = {}
    scored = 0
    for source, scs in scenarios.items():
        by_id = {p.scenario_id: p for p in preds[source]}
        for s in scs:
            p = by_id[s.id]
            if not s.gold_label:
                continue
            scored += 1
            got = predicted_label(p)
            if got != s.gold_label:
                wrong.setdefault(s.category, []).append((s, p, source))
    queues = {c: sorted(v, key=lambda t: t[0].id) for c, v in sorted(wrong.items())}
    shown: list[tuple[Scenario, Prediction, str]] = []
    while len(shown) < limit and any(queues.values()):
        for q in queues.values():
            if q and len(shown) < limit:
                shown.append(q.pop(0))
    total = sum(len(v) for v in wrong.values())
    lines = ["# E4 error analysis: the trained model's mistakes on dev", "",
             f"{total} mistakes among {scored} scored dev cases; {len(shown)} are listed, chosen by a fixed rule "
             "(categories in alphabetical order, one at a time from each, by id). " + DECODE_CAVEAT + " Caveats: " + STYLE_NOTE + "; " + SELECTION_NOTE + ".", "",
             "## Mistakes by category", ""]
    lines += [f"- {c}: {len(v)}" for c, v in sorted(wrong.items())] or ["- none"]
    lines += ["", "## Listed mistakes", ""]
    for s, p, source in shown:
        got = predicted_label(p)
        prob = f", probability {p.probability:.2f}" if p.probability is not None else ""
        lines.append(f"### {s.id} ({s.category})")
        lines.append(f"- gold: {s.gold_label}; predicted: {got}{prob}")
        if source == "scenarios":
            judged, earlier = _judged(s)
            lines += [f"- {t.speaker}: {t.text}" for t in earlier[-3:]]
            lines.append(f"- statement, {judged.speaker}: {judged.text}")
        else:
            lines.append("- text not shown (DECODE); look the id up in the local file")
        lines.append("")
    return "\n".join(lines)


def _source_of(root: Path, run_id: str) -> str | None:
    path = root / "results/runs" / f"{run_id}.json"
    return json.loads(path.read_text(encoding="utf-8")).get("scenario_source") if path.is_file() else None


def _score_systems(
    scenarios: dict[str, list[Scenario]], train: Any, ckpt: Path, device: str,
    make_zeroshot: Callable[[], ConsistencyChecker], make_prompted: Callable[[], ConsistencyChecker], say: Callable[[str], None],
) -> tuple[dict[str, dict[str, list[Prediction]]], dict[str, str]]:
    """Predictions of every system on every source, and the reason each system that errored was left out."""
    systems: dict[str, dict[str, list[Prediction]]] = {}
    failed: dict[str, str] = {}

    def score(name: str, make: Callable[[], ConsistencyChecker], warm: bool = False) -> None:
        say(f"scoring {name}")
        try:
            checker = make()
            if warm:
                warm_up(checker, scenarios["scenarios"][0])
            systems[name] = evaluate_system(checker, scenarios)
        except Exception as exc:
            failed[name] = f"{type(exc).__name__}: {exc}"
            say(f"  {name} failed: {failed[name]}")
        finally:
            _free()

    score("trained", lambda: cm.TrainedChecker.from_checkpoint(ckpt, device=device, strip_instructions=False))
    score("tfidf", lambda: TfidfChecker().fit(train.pairs))
    score("zeroshot", make_zeroshot)
    score("prompted", make_prompted, warm=True)
    return systems, failed


def _intervals_and_comparisons(
    scs: list[Scenario], preds: dict[str, list[Prediction]], metrics: dict[str, Any], names: list[str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    intervals = {}
    for n, p in preds.items():
        got = {m: iv for m, iv in recompute_intervals(scs, p, [m for m in names if metrics[n].get(m) is not None]).items() if iv}
        if got:
            intervals[n] = got
    comparisons = []
    for other in preds:
        for m in [x for x in names if x != "ece"]:
            if other != "trained" and (c := recompute_comparison(scs, preds["trained"], preds[other], m)):
                comparisons.append({"a": "trained", "b": other, **c})
    return intervals, comparisons


def run_e4(*, root: Path = ROOT, raw_root: Path | None = None, base_model: str = cm.BASE_MODEL, device: str = "cpu",
           grid=GRID, make_zeroshot: Callable[[], ConsistencyChecker], make_prompted: Callable[[], ConsistencyChecker],
           prompted_model: str = PROMPTED_MODEL, say=_say, num_workers: int = 0,
           supersedes: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """Search, train the winner, score every system on both sources, and write two records and the error analysis.
    A search run or a baseline that errors is listed in `attempts` and left out. With no finished training run
    nothing is written and the run raises. Any left-out system makes the result incomplete. `supersedes` lists earlier
    records (run_id and reason); each new record lists the ones of its own scenario source, and the earlier files stay
    on disk."""
    raw_root = raw_root or root / RAW_DIR
    if not committed_unchanged({"git_commit": head_commit(root)}, root, SCENARIO_SOURCES["decode"][0].as_posix()):
        raise RuntimeError("the decode manifest must be committed before the run (run with --prepare)")
    started = datetime.now(timezone.utc).isoformat()
    data = {src: load_verified(src, SPLIT, root) for src in SOURCES}
    scenarios = {src: scs for src, (scs, _) in data.items()}
    train = training_data(raw_root, root, scenarios["scenarios"])
    dev_examples = cm.examples_from_scenarios(scenarios["scenarios"])
    say(f"training data: {train.counts}; dev macro-F1 is scored on {len(dev_examples)} labelled dev scenarios")

    attempts = search(base_model, train, dev_examples, device=device, root=root, grid=grid, say=say, num_workers=num_workers)
    winner = pick_winner(attempts)
    if winner is None:
        raise RuntimeError("no training run finished; nothing is written: " + "; ".join(a["error"] for a in attempts))
    say(f"winner: {winner['setting']}")
    lowered = [a["setting"] for a in attempts if a["lowered_batch_size"]]
    if lowered:
        say(f"WARNING: {lowered} ran at a lower batch size than {BATCH_SIZE}, so the settings are not trained alike")

    ckpt = root / CHECKPOINTS / winner["setting"]
    systems, failed = _score_systems(scenarios, train, ckpt, device, make_zeroshot, make_prompted, say)
    if "trained" not in systems:
        raise RuntimeError(f"the winning model could not be scored: {failed['trained']}")

    error_md = error_analysis(scenarios, {src: systems["trained"][src] for src in SOURCES})
    ckpt_hash = hashlib.sha256((ckpt / "model.safetensors").read_bytes()).hexdigest() if (ckpt / "model.safetensors").is_file() else None
    paths = {}
    for src in SOURCES:
        scs, digest = data[src]
        preds = {n: systems[n][src] for n in SYSTEMS if n in systems}
        metrics = {n: _saved(recompute_metrics(scs, p), src) for n, p in preds.items()}
        names = ["macro_f1", "label_accuracy", "ece"] if src == "scenarios" else ["label_accuracy", "false_alert_rate"]
        intervals, comparisons = _intervals_and_comparisons(scs, preds, metrics, names)
        attempt_rows = [{**a, "kind": "training"} for a in attempts] + [
            {"setting": n, "kind": "system", "ok": n in systems, "error": failed.get(n)} for n in SYSTEMS if n != "trained"]
        own_supersedes = [e for e in supersedes or [] if _source_of(root, e["run_id"]) == src]
        config = {
            "experiment": "E4", "source": src, "winner_rule": WINNER_RULE, "winner": winner["setting"],
            "grid": [list(g) for g in grid], "batch_size": BATCH_SIZE, "min_batch_size": MIN_BATCH_SIZE,
            "settings_at_lower_batch_size": lowered, "winner_batch_size": winner["batch_size"],
            "batch_size_note": BATCH_NOTE, "num_workers": num_workers, "patience": PATIENCE,
            "threshold": cm.THRESHOLD, "freeze_word_embeddings": True, "training": train.counts,
            "train_only_balance": train.balance,
            "sample_rule": SAMPLE_RULE, "decode_dev_sample": DECODE_DEV_N, "decode_train_sample": DECODE_TRAIN_N,
            "selection_data": SELECTION_NOTE,
            "style_note": STYLE_NOTE, "train_set_review_note": TRAIN_SET_REVIEW_NOTE, "decode_note": DECODE_NOTE, "checkpoint_sha256": ckpt_hash,
            "scored_cases": sum(s.gold_label is not None for s in scs), "started_at": started,
            "failed_systems": failed,
            **({"rerun_note": RERUN_NOTE} if supersedes else {}),
        }
        paths[src] = write_run(
            "E4", SPLIT, config,
            {"trained": f"{base_model} fine-tuned", "tfidf": "tfidf-logreg", "zeroshot": ZERO_SHOT_MODEL,
             "prompted": prompted_model}, {"consistency": "v1"}, metrics, intervals, attempt_rows, preds, digest,
            comparisons=comparisons, scenario_source=src, root=root, supersedes=own_supersedes)
    out = root / ERROR_ANALYSIS_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(error_md, encoding="utf-8")
    return {"winner": winner["setting"], "complete": not failed, "records": paths, "failed": failed}


def run_and_unload(**kw: Any) -> dict[str, Any]:
    """`run_e4`, then unload the prompted Ollama model whether or not the run succeeded."""
    try:
        return run_e4(**kw)
    finally:
        unload_ollama(kw.get("prompted_model", PROMPTED_MODEL))


def smoke(*, root: Path = ROOT, raw_root: Path | None = None, base_model: str = cm.BASE_MODEL, device: str = "cuda",
          lr: float = SMOKE_LR, steps: int = SMOKE_STEPS, say=_say) -> dict[str, Any]:
    """A few minutes' check at the real settings (batch size, frozen embeddings, class weights, bf16 on a GPU): train
    on a few real batches, cycled, and require every loss finite and the last cycle's mean below the first's.
    Prints the peak GPU memory. Writes nothing under results/ or models/."""
    import random

    import torch
    from transformers import get_linear_schedule_with_warmup

    from contextlens.train_guard import require_finite

    raw_root = raw_root or root / RAW_DIR
    dev = load_scenarios(SPLIT, root / SCENARIO_SOURCES["scenarios"][1])
    data = training_data(raw_root, root, dev)
    say(f"smoke: training data {data.counts}")
    model, tokenizer = cm.load_pretrained(base_model)
    model.to(device)
    for p in model.get_input_embeddings().parameters():
        p.requires_grad = False
    order = list(range(len(data.examples)))
    random.Random(13).shuffle(order)
    batches = [ct.collate(tokenizer, [data.examples[i] for i in order[k * BATCH_SIZE:(k + 1) * BATCH_SIZE]])
               for k in range(SMOKE_BATCHES)]
    weights = torch.tensor(ct.class_weights([e.label for e in data.examples]))  # type: ignore[misc]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, steps)
    on_gpu = device.startswith("cuda")
    if on_gpu:
        torch.cuda.reset_peak_memory_stats()
    model.train()
    losses = []
    for i in range(steps):
        loss = ct.train_step(model, dict(batches[i % SMOKE_BATCHES]), optimizer, weights, scheduler)
        require_finite(loss, f"smoke step {i + 1}")
        losses.append(loss)
        say(f"  step {i + 1}/{steps} loss {loss:.4f}")
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise RuntimeError("smoke: the weights are no longer finite after training")
    first, last = (sum(losses[:SMOKE_BATCHES]) / SMOKE_BATCHES, sum(losses[-SMOKE_BATCHES:]) / SMOKE_BATCHES)
    peak = torch.cuda.max_memory_allocated() / 2**20 if on_gpu else None
    say(f"smoke: mean loss {first:.4f} -> {last:.4f}; peak GPU memory " + (f"{peak:.0f} MB" if peak is not None else "not measured: no GPU"))
    if not last < first:
        raise RuntimeError(f"smoke: the loss did not fall ({first:.4f} -> {last:.4f})")
    return {"losses": losses, "first": first, "last": last, "peak_gpu_memory_mb": peak}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prepare", action="store_true", help="write the DECODE dev sample and its manifest, then stop")
    ap.add_argument("--smoke", action="store_true",
                    help="train a few real batches on the GPU at the real settings, check the loss, write nothing")
    ap.add_argument("--supersedes", nargs="+", metavar="RUN_ID", default=[],
                    help="earlier E4 records this run replaces (RERUN_REASON is the stated reason)")
    args = ap.parse_args()
    if args.prepare:
        print(prepare_decode_dev())
        return 0
    import torch

    if not torch.cuda.is_available():
        print("no GPU: this run trains DeBERTa and needs a GPU turn", file=sys.stderr)
        return 3
    if args.smoke:
        smoke()
        print("smoke ok")
        return 0
    from contextlens.llm.client import OllamaClient

    result = run_and_unload(device="cuda", make_zeroshot=ZeroShotChecker,
                    make_prompted=lambda: PromptedChecker(OllamaClient(PROMPTED_MODEL)), num_workers=NUM_WORKERS,
                    supersedes=[{"run_id": r, "reason": RERUN_REASON} for r in args.supersedes] or None)
    print(json.dumps({"winner": result["winner"], "complete": result["complete"], "failed": result["failed"],
                      "records": [p.as_posix() for p in result["records"].values()]}, indent=1))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    sys.exit(main())
