"""E5: a fine-tuned card writer against the same model with the best E6 prompt, both through `HFClient`.

Stages (each its own command; only `--probe`, `--train` and `--write` hold the GPU):
  --make-subset                     CPU: fix the QMSum dev subset (ids only) before anything runs.
  --probe --prompt-version V        GPU: a short timed run of HFClient and of training; prints the projected time.
  --train [--small-ok]              GPU: build the training set, train, save the adapter under models/, hash it.
  --write --prompt-version V        GPU: write the dev cards, base model with prompt V, then the adapter. Judges nothing.
  --judge --prompt-version V        no GPU: Haiku through `claude -p` judges every claim, one per call; then the record.

The prompted side is the product's `RetrievalAssistant` with prompt V; the adapter side shows the model the same retrieved
turns in the format it was trained on (`cards.dataset.SYSTEM` and `user_prompt`). Retrieval runs once, before any model is
loaded, and both sides read the saved selection. Dev text of QMSum never goes to a cloud model: its rows are scored on
the measures that need no judge and are kept under the git-ignored results/partial/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from contextlens.agents.single import RetrievalAssistant, load_retrieval_config
from contextlens.cards.dataset import MAX_GOLD, N_RECENT, SYSTEM, _trim, build_examples, user_prompt
from contextlens.eval import series
from contextlens.eval.check_run import recompute_metrics
from contextlens.eval.records import ROOT, load_verified, write_atomic, write_run
from contextlens.experiments.e2_context import (
    EMBED_MODEL,
    RUBRIC_VERSION,
    STREAK_LOCAL,
    build_store,
    choose_subset,
    run_system,
)
from contextlens.llm.client import REPO_ROOT, Embedder, shared_root
from contextlens.llm.hf import HFClient
from contextlens.replay.runner import Context, Prediction
from contextlens.retrieval.hybrid import Retriever, ScoredTurn
from contextlens.retrieval.query import query_for
from contextlens.schemas import ButtonRequest, Scenario, Turn

EXPERIMENT, SPLIT = "E5", "dev"
BASE_MODEL = "Qwen/Qwen3.5-4B"
ADAPTER_REL = Path("models/e5-adapter")  # git-ignored
PARTIAL_REL = Path("results/partial/e5")  # git-ignored
QMSUM_SUBSET, QMSUM_N, SUBSET_SEED = "e5-qmsum-100", 100, "e5-v1"
ABSTAIN_CAP, CAP_SEED, MIN_EXAMPLES = 0.25, 13, 300
EPOCHS, BATCH, MAX_LEN, PROBE_STEPS, PROBE_CARDS = 2, 2, 2048, 4, 3
DEADLINE_S, REPLY_TOKENS = 300.0, 700  # a local 4-bit model is slow; both sides get the same generous limits
MODES = ("recall", "clarify", "respond")
HEADLINE = ("grounded_rate", "unsupported_claim_rate", "schema_valid_rate", "abstain_when_insufficient",
            "answer_when_answerable")
LOWER_IS_BETTER = {"unsupported_claim_rate"}

WINNER_RULE = (
    "Written before the run. The adapter is chosen only if, on the dev cases of `scenarios`, the interval of its grounded_rate "
    "minus the prompted version's lies wholly above zero, and its unsupported_claim_rate, schema_valid_rate, "
    "abstain_when_insufficient and answer_when_answerable are each no worse than the prompted version's. Otherwise the prompted "
    "version stays in the product and the record says the tuning did not beat it."
)
SYNTHETIC_STATEMENT = ("The training mix includes synthetic data: the invented train-only scenario set (datasets/scenarios_train), "
                       "marked synthetic, besides QMSum train conversations.")
OVERLAP_STATEMENT = ("Limit of T6.1's overlap guard: it refuses a training scenario at 50% shared 3-word windows with a dev or "
                     "test conversation. It guards against accidental reuse in our hand-written set, not against a deliberate "
                     "paraphrase.")
READ_STATEMENT = ("An independent read of the invented train-only set, before training, is what caught real near-copies of "
                  "held-out scenarios; the 3-word overlap guard scored them only 0.13 to 0.18, so it could not have caught "
                  "them. Those cases were rewritten and read again.")
JUDGE_STATEMENT = ("Judge: Haiku through `claude -p`, one claim per call, rubric e2-v1. Haiku failed the written calibration "
                   "against Sonnet (docs/DECISIONS.md entry 56: agreement 0.813 against the local judge's 0.870, difference "
                   "-0.057 with an interval wholly below zero). The explanation there, that the local judge says supported to "
                   "nearly everything and so gains from the base rate, is a reading and not proof. Sonnet did not grade E5. "
                   "No person reviewed these judgements (no config.judge_review), so they are provisional, not claims.")
QMSUM_STATEMENT = ("QMSum dev cards are scored without a judge: their text is not sent to a cloud model and not committed. "
                   "Their figures are counts from git-ignored rows, so check_run cannot recompute them.")
SPLIT_NOTE = "Retrieval for both sides is the product's (configs/retrieval.json), computed once and shared."


def data_root() -> Path:
    """The main folder, where git-ignored data (QMSum, trained models, partial rows) lives even from a worktree."""
    return shared_root(REPO_ROOT)


# ---------------------------------------------------------------- training set

def training_summary(stats: Counter) -> dict[str, Any]:
    n = stats["train_final_size"]
    by_mode_status = {k.split("_", 1)[1]: v for k, v in stats.items()
                      if k.split("_", 1)[0] in ("qmsum", "scenarios") and k.endswith(("answered", "abstained", "partial"))}
    return {
        "size": n, "answered": stats["train_class_answered"], "abstained": stats["train_class_abstained"],
        "abstained_share": stats["train_class_abstained"] / n if n else None,
        "by_source": {"qmsum": stats["train_source_qmsum"], "scenarios_train": stats["train_source_scenarios_train"]},
        "synthetic_share": stats["train_synthetic_share"],
        "converted_consistent": stats["converted_consistent"],
        "abstained_dropped_by_cap": stats["train_abstained_dropped_by_cap"],
        "dropped": {k: v for k, v in stats.items() if k.startswith("dropped_")},
        "built_by_source_mode_status": by_mode_status,
        "max_gold": MAX_GOLD, "abstain_cap": ABSTAIN_CAP, "cap_seed": CAP_SEED,
        "under_minimum": n < MIN_EXAMPLES, "minimum": MIN_EXAMPLES,
    }


def build_training(root: Path, cards_dir: Path) -> dict[str, Any]:
    """QMSum train plus the invented train-only set, consistent cases made into clarify cards, 25% abstained cap."""
    if not (root / "datasets/scenarios_train/train.jsonl").is_file():
        raise RuntimeError("the invented train-only set (datasets/scenarios_train) does not exist; E5 needs it")
    stats: Counter = Counter()
    build_examples(raw_root=root / "datasets/raw", splits_dir=root / "datasets/splits", out_dir=cards_dir,
                   train_scenarios=root / "datasets/scenarios_train", held_out_dir=root / "datasets/scenarios",
                   stats=stats, abstain_cap=ABSTAIN_CAP, cap_seed=CAP_SEED, convert_consistent=True)
    return training_summary(stats)


def adapter_hash(path: Path) -> str:
    """SHA-256 over the adapter's config and weights files (names and bytes), so any change to either changes it."""
    h = hashlib.sha256()
    names = sorted(n for n in ("adapter_config.json", "adapter_model.safetensors") if (path / n).is_file())
    if "adapter_model.safetensors" not in names:
        raise FileNotFoundError(f"no adapter_model.safetensors in {path}")
    for n in names:
        h.update(n.encode() + b"\0" + (path / n).read_bytes())
    return h.hexdigest()


def steps_for(n_examples: int) -> int:
    return math.ceil(EPOCHS * n_examples / BATCH)


def run_train(root: Path, *, small_ok: bool, say: Callable[[str], None] = print) -> Path:
    from contextlens.cards.train_lora import train

    cards_dir = root / "datasets/cards"
    summary = build_training(root, cards_dir)
    say(f"training set: {summary['size']} examples, {summary['by_source']}, synthetic share "
        f"{summary['synthetic_share']:.2f}, abstained share {summary['abstained_share']:.2f}")
    if summary["under_minimum"] and not small_ok:
        raise SystemExit(f"refused: {summary['size']} examples is under {MIN_EXAMPLES}; tell the coordinator, then rerun with --small-ok")
    out = root / ADAPTER_REL
    meta = train(out, steps_for(summary["size"]), MAX_LEN, cards_dir, include_scenarios=True)
    info = {"training_set": summary, "adapter_hash": adapter_hash(out), "epochs": EPOCHS, "batch_size": BATCH,
            "max_len": MAX_LEN, "train_meta": meta}
    write_atomic(out / "e5_training.json", json.dumps(info, indent=1) + "\n")
    say(f"adapter saved to {out} (sha256 {info['adapter_hash'][:16]}...), final loss {meta['final_loss']:.4f}")
    return out


# ---------------------------------------------------------------- retrieval shared by both sides

class ReplayRetriever:
    """Hands back the turns the real retriever chose for a conversation, so both sides see the same evidence."""

    def __init__(self, chosen: dict[str, list[ScoredTurn]]) -> None:
        self.chosen = chosen

    def search(self, conv_id: str, query: str, as_of_seq: int, k: int = 8, **kw: Any) -> list[ScoredTurn]:
        return self.chosen.get(conv_id, [])


def select_evidence(scenarios: list[Scenario], embedder: Any, config: dict[str, Any]) -> dict[str, list[tuple[str, float]]]:
    """Scenario id -> (turn id, score) of the retrieved turns, by the product's rule. Needs the embedder."""
    out = {}
    for sc in scenarios:
        query, last_seq = query_for(sc)
        hits = Retriever(build_store(sc), embedder).search(sc.conversation.id, query, last_seq, k=config["k"], mode=config["mode"])
        out[sc.id] = [(h.turn.id, h.score) for h in hits]
    return out


def replay_retriever(sc: Scenario, selection: dict[str, list[tuple[str, float]]]) -> ReplayRetriever:
    by_id = {t.id: t for t in sc.conversation.turns}
    return ReplayRetriever({sc.conversation.id: [ScoredTurn(turn=by_id[i], score=s, parts={}) for i, s in selection[sc.id]]})


# ---------------------------------------------------------------- the two writers

class _Local:
    reply_tokens = REPLY_TOKENS
    call_timeout_s = DEADLINE_S


class PromptedAssistant(_Local, RetrievalAssistant):
    """The product's retrieval assistant with the chosen prompt version."""


class AdapterAssistant(_Local, RetrievalAssistant):
    """Same retrieved turns, shown in the format the adapter was trained on: the newest turns as the recent exchange,
    the rest as evidence."""

    def __init__(self, llm: Any, retriever: Any, prompt_version: str, **kw: Any) -> None:
        super().__init__(llm, retriever, prompt_version, **kw)
        self.system = SYSTEM

    def _messages(self, turns: list[Turn], req: ButtonRequest, ctx: Context) -> list[dict]:
        trimmed = [_trim(t) for t in turns]
        recent, evidence = trimmed[-N_RECENT:], trimmed[:-N_RECENT]
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt(recent, evidence, req.question)}]


class RecordingHF(HFClient):
    """Keeps every raw reply, so a card's first reply can be checked against the schema without the retry's help."""

    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        self.raws: list[str] = []

    def _raw(self, *a: Any, **kw: Any):
        res = super()._raw(*a, **kw)
        self.raws.append(res.text)
        return res


def write_side(name: str, scenarios: list[Scenario], client: RecordingHF, selection: dict[str, list[tuple[str, float]]],
               prompt_version: str, rows_path: Path, say: Callable[[str], None] = print) -> list[Prediction]:
    """Cards for every scenario, one at a time, saved as they come so a stopped run resumes. Calls no judge."""
    have = {p.scenario_id: p for p in _read_rows(rows_path)}
    cls = AdapterAssistant if name == "adapter" else PromptedAssistant

    def make(sc: Scenario):
        client.raws.clear()
        return cls(client, replay_retriever(sc, selection), prompt_version, deadline_s=DEADLINE_S)

    def keep(row: Prediction) -> None:
        row.raw_output = client.raws[0] if client.raws else None
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        with rows_path.open("a", encoding="utf-8") as f:
            f.write(row.model_dump_json() + "\n")

    return run_system(name, scenarios, make, STREAK_LOCAL, say=say, have=have, on_row=keep)


def _read_rows(path: Path) -> list[Prediction]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(Prediction.model_validate_json(line))
        except ValueError:
            continue  # a line cut short by a crash
    return out


def run_dir(root: Path, prompt_version: str, adapter_sha: str) -> Path:
    return root / PARTIAL_REL / f"{prompt_version}-{adapter_sha[:12]}"


def eval_scenarios(root: Path) -> dict[str, list[Scenario]]:
    """Dev cases per source: all of `scenarios`, and the fixed QMSum subset (read from the main folder's raw data)."""
    own, _ = load_verified("scenarios", SPLIT, root)
    q_all, _ = load_verified("qmsum", SPLIT, data_root())
    ids = set(json.loads((root / "datasets/splits/subsets" / f"{QMSUM_SUBSET}.json").read_text(encoding="utf-8"))["ids"])
    return {"scenarios": own, "qmsum": [s for s in q_all if s.id in ids]}


def make_subset(root: Path) -> Path:
    q_all, _ = load_verified("qmsum", SPLIT, data_root())
    ids = choose_subset(q_all, QMSUM_N, SUBSET_SEED)
    manifest = {"name": QMSUM_SUBSET, "split": SPLIT, "ids": ids, "source": "qmsum",
                "rule": f"{QMSUM_N} of the QMSum dev cases, proportional by (category, origin), hash order with seed {SUBSET_SEED}"}
    path = root / "datasets/splits/subsets" / f"{QMSUM_SUBSET}.json"
    write_atomic(path, json.dumps(manifest, indent=1) + "\n")
    return path


def run_write(root: Path, prompt_version: str, say: Callable[[str], None] = print) -> Path:
    from contextlens.llm.client import unload_ollama

    info = json.loads((root_models(root) / "e5_training.json").read_text(encoding="utf-8"))
    base = info["train_meta"]["base_model"]
    folder = run_dir(root, prompt_version, info["adapter_hash"])
    sets = eval_scenarios(root)
    config = load_retrieval_config()
    sel_path = folder / "selection.json"
    if sel_path.is_file():
        selection = json.loads(sel_path.read_text(encoding="utf-8"))
    else:
        say("retrieving once for both sides")
        selection = select_evidence([s for v in sets.values() for s in v], Embedder(EMBED_MODEL), config)
        unload_ollama(EMBED_MODEL)
        write_atomic(sel_path, json.dumps(selection))
    for side, adapter in (("prompted", None), ("adapter", str(root_models(root)))):
        client = RecordingHF(base, adapter)
        try:
            client.load()
            for source, scs in sets.items():
                write_side(side, scs, client, selection, prompt_version, folder / f"{side}.{source}.rows.jsonl", say)
        finally:
            client.unload()
    return folder


def root_models(root: Path) -> Path:
    return data_root() / ADAPTER_REL


# ---------------------------------------------------------------- the probe

def run_probe(root: Path, prompt_version: str, say: Callable[[str], None] = print) -> dict[str, float]:
    """A short timed run: load the base model, write a few cards, a few training steps. Prints the projected time."""
    from contextlens.cards.train_lora import train

    sets = eval_scenarios(root)
    scs = sets["scenarios"][:PROBE_CARDS]
    config = load_retrieval_config()
    selection = select_evidence(scs, Embedder(EMBED_MODEL), config)
    from contextlens.llm.client import unload_ollama

    unload_ollama(EMBED_MODEL)
    client = RecordingHF(BASE_MODEL)
    t0 = time.perf_counter()
    client.load()
    load_s = time.perf_counter() - t0
    rows = write_side("prompted", scs, client, selection, prompt_version, root / PARTIAL_REL / "probe.rows.jsonl", say)
    card_s = sum(p.stage_latency_ms.get("total", 0) for p in rows) / len(rows) / 1000
    client.unload()
    shared = data_root()
    summary = build_training(shared, shared / "datasets/cards")
    meta = train(shared / "models/e5-probe", PROBE_STEPS, MAX_LEN, shared / "datasets/cards", include_scenarios=True)
    step_s = meta["train_s"] / PROBE_STEPS
    full_n = len(sets["scenarios"]) + len(sets["qmsum"])
    train_total = meta["load_s"] + steps_for(summary["size"]) * step_s
    write_total = 2 * (load_s + full_n * card_s)
    proj = {"load_s": load_s, "card_s": card_s, "step_s": step_s, "train_load_s": meta["load_s"],
            "peak_vram_gb": meta["peak_vram_gb"], "train_min": train_total / 60, "write_min": write_total / 60,
            "training_examples": summary["size"], "cases": full_n}
    say(f"probe: model load {load_s:.0f}s, {card_s:.1f}s per card, {step_s:.1f}s per training step, peak {meta['peak_vram_gb']} GB")
    say(f"projected: training {steps_for(summary['size'])} steps = {train_total / 60:.0f} min; writing {full_n} cases x 2 sides = "
        f"{write_total / 60:.0f} min; judging after release: about one Haiku call per claim, 8 in parallel")
    return proj


# ---------------------------------------------------------------- measures

def gold_mode(s: Scenario) -> str:
    """The mode the case asks for: a pair to compare is clarify, a question is recall, anything else respond."""
    if s.gold_label in ("contradiction", "update") or (s.gold_label == "consistent" and not s.request.question):
        return "clarify"
    return "recall" if s.request.question else "respond"


def false_alarm_rate(rows: list[tuple[Scenario, Prediction]]) -> dict[str, Any]:
    """Among cases whose statements agree (gold consistent), the share whose card calls a contradiction or an update."""
    agree = [(s, p) for s, p in rows if s.gold_label == "consistent" and p.card is not None]
    alarms = sum(any(f.label in ("contradiction", "update") for f in p.card.findings) for _, p in agree)  # type: ignore[union-attr]
    return {"n": len(agree), "false_alarms": alarms, "rate": alarms / len(agree) if agree else None}


def per_mode(rows: list[tuple[Scenario, Prediction]]) -> dict[str, Any]:
    out = {}
    for mode in MODES:
        sub = [(s, p) for s, p in rows if gold_mode(s) == mode]
        out[mode] = {"n": len(sub), **{m: series.metric_value(sub, m) if sub else None for m in HEADLINE}}
    return out


def pair_rows(scenarios: list[Scenario], preds: list[Prediction]) -> list[tuple[Scenario, Prediction]]:
    by_id = {p.scenario_id: p for p in preds}
    return [(s, by_id[s.id]) for s in scenarios]


def unsupported_count(preds: list[Prediction]) -> dict[str, int]:
    flags = [x for p in preds for x in (p.claim_supported or [])]
    return {"claims": len(flags), "unsupported": flags.count(False)}


def choose_winner(comparisons: list[dict[str, Any]], metrics: dict[str, dict[str, float | None]]) -> tuple[str, str]:
    """The winner by WINNER_RULE, and the sentence that says why."""
    grounded = next((c for c in comparisons if c["metric"] == "grounded_rate"), None)
    if grounded is None or grounded["ci"][0] <= 0:
        return "prompted", "the adapter's grounded_rate is not shown above the prompted version's"
    a, b = metrics["adapter"], metrics["prompted"]
    worse = [m for m in HEADLINE[1:] if a.get(m) is not None and b.get(m) is not None
             and ((a[m] > b[m]) if m in LOWER_IS_BETTER else (a[m] < b[m]))]
    if worse:
        return "prompted", f"the adapter is worse on {', '.join(worse)}"
    return "adapter", "the adapter's grounded_rate is above the prompted version's and nothing in the rule is worse"


def build_record(*, root: Path, scenarios: list[Scenario], dataset_hash: str, preds: dict[str, list[Prediction]],
                 extra_config: dict[str, Any], models: dict[str, str], prompt_version: str, training: dict[str, Any],
                 qmsum: dict[str, Any]) -> Path:
    """The E5 record for the `scenarios` dev source. `preds` have the judge's `claim_supported` filled in."""
    metrics = {n: recompute_metrics(scenarios, p) for n, p in preds.items()}
    rows = {n: pair_rows(scenarios, p) for n, p in preds.items()}
    intervals = {n: {m: iv for m in HEADLINE if (iv := series.interval(rows[n], m)) is not None} for n in preds}
    comparisons = [{"a": "adapter", "b": "prompted", **c} for m in HEADLINE
                   if (c := series.comparison(rows["adapter"], rows["prompted"], m)) is not None]
    winner, why = choose_winner(comparisons, metrics)
    history = [{"step": i, "train_loss": x} for i, x in enumerate(training["train_meta"]["losses"], 1)]
    config = {
        "labels_reviewed": False, "winner": winner, "winner_rule": WINNER_RULE, "winner_reason": why,
        "tuning_beat_prompted": winner == "adapter",
        "prompt_version": prompt_version, "base_model": training["train_meta"]["base_model"],
        "adapter_hash": training["adapter_hash"], "adapter_path": ADAPTER_REL.as_posix(),
        "fallback_used": training["train_meta"]["fallback_used"], "fallback_reason": training["train_meta"]["fallback_reason"],
        "training_set": training["training_set"], "epochs": training["epochs"], "batch_size": training["batch_size"],
        "synthetic_statement": SYNTHETIC_STATEMENT, "overlap_guard_statement": OVERLAP_STATEMENT,
        "independent_read_statement": READ_STATEMENT, "qmsum_statement": QMSUM_STATEMENT, "retrieval_note": SPLIT_NOTE,
        "judge": {"model": models["judge"], "rubric_version": RUBRIC_VERSION, "same_model_and_prompt_as_writer": False,
                  "statement": JUDGE_STATEMENT},
        "per_mode": {n: per_mode(rows[n]) for n in rows},
        "false_alarm_on_consistent": {n: false_alarm_rate(rows[n]) for n in rows},
        "unsupported_claims": {n: unsupported_count(preds[n]) for n in preds},
        "per_source": {"scenarios": {"n": len(scenarios)}, "qmsum": qmsum},
        "schema_valid_note": "schema_valid_rate is the share of cases whose first model reply parsed as a Card, before the retry.",
        "peak_vram_gb": training["train_meta"]["peak_vram_gb"],
        **extra_config,
    }
    attempts = [{"setting": "adapter", "kind": "training", "history": history, "steps": len(history)},
                {"setting": "prompted", "kind": "writing", "prompt_version": prompt_version}]
    return write_run(EXPERIMENT, SPLIT, config, models, {"prompted": prompt_version, "adapter": "cards.dataset.SYSTEM"},
                     metrics, intervals, attempts, preds, dataset_hash, comparisons=comparisons, root=root)


# ---------------------------------------------------------------- judging

def run_judge(root: Path, prompt_version: str, say: Callable[[str], None] = print, *, llm: Any = None,
              guard: Any = None, concurrency: int | None = None) -> Path:
    from contextlens.experiments import parallel_grader as pg
    from contextlens.experiments.judge_check import preflight
    from contextlens.llm.client import CLAUDE_CALL_CAP, CLAUDE_CALLS_PATH

    info = json.loads((root_models(root) / "e5_training.json").read_text(encoding="utf-8"))
    folder = run_dir(root, prompt_version, info["adapter_hash"])
    sets = eval_scenarios(root)
    own = sets["scenarios"]
    preds = {side: _complete_rows(own, _read_rows(folder / f"{side}.scenarios.rows.jsonl"), side) for side in ("prompted", "adapter")}
    items = [pg.Item(f"{side}|{p.scenario_id}|{k}", f.statement, tuple((e.speaker, e.quote) for e in f.evidence))
             for side, rows in preds.items() for p in rows if p.card for k, f in enumerate(p.card.findings)]
    llm = llm or pg.CountedClaude()
    if guard is None:
        from contextlens.experiments.judge_calibrate import UsageGuard

        guard = UsageGuard(90.0)
    guard.start()
    store = pg.RowStore(folder / "judge.rows.jsonl", {"kind": "e5-judge-rows", "model": llm.model, "rubric": RUBRIC_VERSION})
    todo = pg.pending(items, store)
    say(f"judging {len(todo)} of {len(items)} claims with {llm.model}, one per call")
    preflight(len(todo), CLAUDE_CALLS_PATH, CLAUDE_CALL_CAP)
    before = getattr(llm, "calls_made", 0)
    out = pg.grade_single(items, llm, store, concurrency=concurrency or pg.DEFAULT_CONCURRENCY, should_stop=guard.check)
    if out.stop_reason or pg.ungraded(items, store):
        raise RuntimeError(f"judging stopped ({out.stop_reason}); {len(pg.ungraded(items, store))} claims ungraded; rerun to resume")
    judged = {side: [_with_verdicts(side, p, store) for p in rows] for side, rows in preds.items()}
    _, dhash = load_verified("scenarios", SPLIT, root)
    qmsum = qmsum_figures(sets["qmsum"], {s: _read_rows(folder / f"{s}.qmsum.rows.jsonl") for s in ("prompted", "adapter")})
    models = {"writer": BASE_MODEL, "adapter": ADAPTER_REL.as_posix(), "judge": llm.model, "embedder": EMBED_MODEL}
    return build_record(root=root, scenarios=own, dataset_hash=dhash, preds=judged,
                        extra_config={"claude_calls": getattr(llm, "calls_made", 0) - before}, models=models,
                        prompt_version=prompt_version, training=info, qmsum=qmsum)


def _complete_rows(scs: list[Scenario], rows: list[Prediction], side: str) -> list[Prediction]:
    by_id = {p.scenario_id: p for p in rows}
    missing = [s.id for s in scs if s.id not in by_id]
    if missing:
        raise RuntimeError(f"the {side} side has no card for {len(missing)} cases; run --write first")
    return [by_id[s.id] for s in scs]


def _with_verdicts(side: str, p: Prediction, store: Any) -> Prediction:
    if p.card is None or not p.card.findings:
        return p
    flags = [bool(store.rows[f"{side}|{p.scenario_id}|{k}"]["cloud"]) for k in range(len(p.card.findings))]
    return p.model_copy(update={"claim_supported": flags})


def qmsum_figures(scs: list[Scenario], rows: dict[str, list[Prediction]]) -> dict[str, Any]:
    """Counts and rates that need no judge, from the QMSum rows: no text."""
    out: dict[str, Any] = {"n": len(scs), "subset": QMSUM_SUBSET, "judged": False}
    for side, preds in rows.items():
        paired = pair_rows(scs, _complete_rows(scs, preds, side))
        m = recompute_metrics(scs, [p for _, p in paired])
        out[side] = {k: m.get(k) for k in ("schema_valid_rate", "abstain_when_insufficient", "answer_when_answerable",
                                           "latency_p50_ms", "latency_p95_ms")}
        out[side]["per_mode"] = {mode: {k: v for k, v in d.items() if k not in ("grounded_rate", "unsupported_claim_rate")}
                                 for mode, d in per_mode(paired).items()}
        out[side]["status"] = dict(Counter(p.card.status for _, p in paired if p.card))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stage = ap.add_mutually_exclusive_group(required=True)
    for flag in ("make-subset", "probe", "train", "write", "judge"):
        stage.add_argument(f"--{flag}", action="store_true")
    ap.add_argument("--prompt-version", help="the best E6 prompt version; required by probe, write and judge")
    ap.add_argument("--small-ok", action="store_true", help="train even though the set is under 300 examples")
    a = ap.parse_args(argv)
    if (a.probe or a.write or a.judge) and not a.prompt_version:
        ap.error("--prompt-version is required")
    if a.make_subset:
        print(make_subset(ROOT))
    elif a.probe:
        run_probe(ROOT, a.prompt_version)
    elif a.train:
        run_train(data_root(), small_ok=a.small_ok)
    elif a.write:
        print(run_write(ROOT, a.prompt_version))
    else:
        print(run_judge(ROOT, a.prompt_version))
    return 0


if __name__ == "__main__":
    sys.exit(main())
