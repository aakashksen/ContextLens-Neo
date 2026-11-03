"""E6: which prompt choices move groundedness and abstention?

One strategy (`retrieval_memory`, the one E2 carried forward), one writer model, the whole dev split, and the same
cases for every prompt. The baseline is prompt version v1. Every other version changes exactly one factor of it, and
says which in its `factor.json` (see `declared_versions`). Nothing else is varied: the memory notes are built once per
case and shared by all versions, and the judge is the one E2 used.

The selection rule is `SELECTION_RULE`; it is fixed here, in a commit before any run. The test split is never read.

Run: .venv/Scripts/python -m contextlens.experiments.e6_prompts
"""

import difflib
import hashlib
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contextlens.data.scenarios import load_scenarios
from contextlens.eval.check_run import recompute_comparison, recompute_intervals
from contextlens.eval.records import ROOT, SCENARIO_SOURCES, head_commit, is_dirty, split_hash, write_run
from contextlens.experiments import e2_context as e2
from contextlens.llm.client import LLMClient, LLMError
from contextlens.llm.ollama_ctl import OllamaCtl, full_name
from contextlens.memory.extract import MemoryStore
from contextlens.memory.store import InMemoryTranscriptStore
from contextlens.prompts.registry import get_prompt
from contextlens.replay.runner import Assistant, Prediction
from contextlens.retrieval.hybrid import Retriever
from contextlens.retrieval.query import query_for
from contextlens.schemas import Scenario

BASELINE = "v1"
FACTORS = ("few_shot", "evidence_format", "abstention", "output_format", "length_limit", "role_split")
VERSIONS = ("v1", "v2", "v3", "v4", "v5", "v6", "v7")  # the order they run in, and the tie-break order
PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
PROMPT_NAME = "recall"

# Not a factor here: a prompt that defines the card modes (recall, clarify, respond). T3.3 tried one and reverted it
# (record E2-20261103T155026435232Z); T4.2 sets the respond mode in code. The record still reports respond mode and
# injected claims, in findings and in suggestions, for every version.
GATE_METRICS = ("grounded_rate", "abstain_when_insufficient", "answer_when_answerable")
GAIN_METRICS = ("grounded_rate", "abstain_when_insufficient")
SELECTION_RULE = (
    "Each candidate version is compared with the baseline v1 on the same cases. score(v) is the mean of grounded_rate "
    "and the lower of abstain_when_insufficient and answer_when_answerable (the weaker abstention side); a version "
    "whose three metrics are not all defined is not eligible. A candidate qualifies only if (1) it has a measurable gain: "
    "the 95% interval of its difference from v1 on grounded_rate or on abstain_when_insufficient lies above zero, "
    "(2) it has no measurable loss: the interval of its difference from v1 on none of grounded_rate, "
    "abstain_when_insufficient and answer_when_answerable lies below zero, and (3) its score is above v1's. The chosen "
    "version is the qualifier with the highest score, ties broken by lower median latency, then by the order of VERSIONS. "
    "With no qualifier the baseline v1 is kept and the record says there was no measurable improvement. A version whose "
    "output was invalid or timed out too often to be measured is listed in attempts and cannot qualify. Respond mode and "
    "injected claims are reported for every version and do not enter the score. No correction is made for comparing six "
    "candidates, so a qualifier is a candidate for a confirming run, not a proven improvement."
)


@dataclass(frozen=True)
class Version:
    version: str
    factor: str
    summary: str
    added: tuple[str, ...]
    removed: tuple[str, ...]
    text: str

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def _lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.strip()]


def line_diff(baseline: str, other: str) -> tuple[list[str], list[str]]:
    """(added, removed) non-blank lines of `other` against `baseline`, each in text order."""
    d = list(difflib.ndiff(_lines(baseline), _lines(other)))
    return [x[2:] for x in d if x.startswith("+ ")], [x[2:] for x in d if x.startswith("- ")]


def declared_versions(prompts_dir: Path = PROMPTS_DIR, names: tuple[str, ...] = VERSIONS) -> list[Version]:
    """Every version, read through the registry, with the factor its `factor.json` declares. Raises ValueError when a
    version does not differ from v1 by exactly its one declared factor: an unknown or repeated factor, a declared
    change that is not the real line difference, or a text another version already has."""
    out: list[Version] = []
    for name in names:
        meta = json.loads((prompts_dir / name / "factor.json").read_text(encoding="utf-8"))
        text = get_prompt(PROMPT_NAME, name).text
        out.append(Version(name, meta["factor"], meta["summary"], tuple(meta["added"]), tuple(meta["removed"]), text))
    base = out[0]
    if base.version != BASELINE or base.factor != "baseline" or base.added or base.removed:
        raise ValueError(f"{BASELINE} must be the baseline and declare no change")
    seen_factors: set[str] = set()
    seen_hashes = {base.text_sha256: base.version}
    for v in out[1:]:
        if v.factor not in FACTORS:
            raise ValueError(f"{v.version}: factor {v.factor!r} is not one of {FACTORS}")
        if v.factor in seen_factors:
            raise ValueError(f"{v.version}: factor {v.factor} is already tried by another version")
        seen_factors.add(v.factor)
        added, removed = line_diff(base.text, v.text)
        if (list(v.added), list(v.removed)) != (added, removed) or not (added or removed):
            raise ValueError(f"{v.version}: the declared change is not its difference from {BASELINE}")
        if v.text_sha256 in seen_hashes:
            raise ValueError(f"{v.version} has the same text as {seen_hashes[v.text_sha256]}")
        seen_hashes[v.text_sha256] = v.version
    return out


def score(m: dict[str, float | None]) -> float | None:
    """Mean of grounded_rate and the weaker abstention side; None unless all three metrics are defined."""
    g, a, r = (m.get(k) for k in ("grounded_rate", "abstain_when_insufficient", "answer_when_answerable"))
    if g is None or a is None or r is None:
        return None
    return (g + min(a, r)) / 2


def _interval(comparisons: list[dict[str, Any]], candidate: str, metric: str) -> list[float] | None:
    for c in comparisons:
        if (c["a"], c["b"], c["metric"]) == (candidate, BASELINE, metric):
            return c["ci"]
    return None


def pick_best(metrics: dict[str, dict[str, float | None]], comparisons: list[dict[str, Any]],
              order: tuple[str, ...] = VERSIONS) -> dict[str, Any]:
    """Apply SELECTION_RULE. `comparisons` holds candidate-minus-baseline entries (`a` the candidate, `b` v1).
    Returns {"selected", "scores", "qualifies"} where qualifies maps each candidate to the reason it did or did not."""
    scores = {v: score(metrics[v]) for v in order if v in metrics}
    base = scores.get(BASELINE)
    if base is None:
        raise ValueError(f"{BASELINE} has an undefined metric, so nothing can be compared with it")
    qualifies: dict[str, str] = {}
    for v in order:
        if v == BASELINE or v not in metrics:
            continue
        ci = {m: _interval(comparisons, v, m) for m in GATE_METRICS}
        if scores[v] is None:
            qualifies[v] = "not eligible: a metric is undefined"
        elif any(ci[m] is not None and ci[m][1] < 0 for m in GATE_METRICS):
            qualifies[v] = "no: a measurable loss on " + ", ".join(m for m in GATE_METRICS if ci[m] and ci[m][1] < 0)
        elif not any(ci[m] is not None and ci[m][0] > 0 for m in GAIN_METRICS):
            qualifies[v] = "no: no measurable gain on grounded_rate or abstain_when_insufficient"
        elif scores[v] <= base:  # type: ignore[operator]
            qualifies[v] = "no: score not above the baseline's"
        else:
            qualifies[v] = "yes"
    eligible = [v for v, why in qualifies.items() if why == "yes"]
    if not eligible:
        return {"selected": BASELINE, "scores": scores, "qualifies": qualifies}
    best = min(eligible, key=lambda v: (-scores[v], metrics[v].get("latency_p50_ms") or 0.0, order.index(v)))  # type: ignore[operator]
    return {"selected": best, "scores": scores, "qualifies": qualifies}


# ---------------------------------------------------------------- the run

SPLIT = "dev"
SYSTEM = e2.TRIED_PROMPT_SYSTEM  # retrieval_memory: the strategy E2 carried forward
PARTIAL_REL = Path("results/partial/e6-rows.jsonl")  # git-ignored; one line per finished card
CALLS_PER_CARD_WORST = 4  # the card and its schema retry, then the suggestion and its schema retry (respond mode only)


def call_counts(n_cases: int, n_versions: int) -> dict[str, int]:
    """Model calls a run makes, from the loops in `_run`. The judge makes one call per finding in the cards, so that
    count is known only after generation; E2's retrieval_memory cards held about 0.86 findings each."""
    cards = n_cases * n_versions
    return {
        "judge_canary": len(e2.CANARY),
        "writer_and_embedder_ping": 2,
        "memory_extraction": n_cases,
        "writer_cards_min": cards,
        "writer_cards_worst": cards * CALLS_PER_CARD_WORST,
        "embedder_index": n_cases,  # one index per case, shared by every version
    }


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_assistant(v: str, writer: LLMClient, retriever: Retriever, memory: MemoryStore,
                     retrieval_config: dict[str, Any]) -> e2.RetrievalMemoryAssistant:
    return e2.RetrievalMemoryAssistant(writer, retriever, memory, v, config=retrieval_config, deadline_s=e2.DEADLINE_S)


def _retriever(sc: Scenario, embedder: Any) -> Retriever:
    r = Retriever(e2.build_store(sc), embedder)
    query, last = query_for(sc)
    if last >= 0:
        r.search(sc.conversation.id, query, last, k=1, mode="dense")  # index the turns outside the timing
    return r


def _saved_rows(path: Path, commit: str, ids: set[str]) -> dict[str, dict[str, Prediction]]:
    """Rows of an interrupted run at this commit, by version and case. A row that failed outright is not reused,
    and a line cut short by a crash is skipped."""
    have: dict[str, dict[str, Prediction]] = {}
    if not path.exists():
        return have
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
            row = Prediction.model_validate(e["row"])
            if e["commit"] != commit or row.scenario_id not in ids or e2.row_state(row) == "broken":
                continue
            have.setdefault(e["version"], {}).setdefault(row.scenario_id, row)
        except (ValueError, KeyError, TypeError):
            continue
    return have


def _health(version: str, rows: list[Prediction]) -> tuple[dict[str, Any], str | None]:
    """(counts, why the version cannot be measured or None). A prompt that makes the output invalid or slow is a
    result and is listed in attempts; calls that fail outright mean the setup is broken, so that raises. A version
    whose cards all abstain is measured (it answers nothing), except the baseline, which must work."""
    n = len(rows)
    states = [e2.row_state(p) for p in rows]
    counts: dict[str, Any] = {s: states.count(s) for s in ("broken", "invalid", "timeout", "ok")}
    counts["abstained"] = sum(p.card is not None and p.card.status == "abstained" for p in rows)
    counts["n"] = n
    if counts["broken"] / n > e2.BROKEN_MAX:
        sample = next((p.card.missing[0][:200] for p in rows if p.card and e2.row_state(p) == "broken"), "")
        raise e2.RunBroken(f"version {version}: {counts['broken']} of {n} cases failed outright; {sample}")
    if version == BASELINE and counts["abstained"] == n:
        raise e2.RunBroken("the baseline abstained on every case, so nothing was measured")
    for what, key, limit in (("ran out of time", "timeout", e2.TIMEOUT_MAX), ("gave invalid output twice", "invalid", e2.INVALID_MAX)):
        if counts[key] / n > limit:
            return counts, f"{counts[key]} of {n} cases {what} (limit {limit:.0%})"
    return counts, None


def run_e6(*, root: Path = ROOT, writer: LLMClient, judge_llm: LLMClient, embedder: Any, ctl: OllamaCtl,
           names: e2.Models | None = None, retrieval_config: dict[str, Any],
           gpu_mb: Callable[[], int | None] = e2.gpu_used_mb, say: Callable[[str], None] = print) -> dict[str, Any]:
    """Run every prompt version on dev with the writer, judge the cards, write one record. The prompts and the agent
    code are taken as they are at the run's commit. Raises RunBroken, before anything is written, when the setup is
    broken. Models are unloaded at the end whatever happens."""
    names = names or e2.Models()
    mine = [(names.writer, False), (names.judge, False), (names.embedder, True)]
    try:
        return _run(root, writer, judge_llm, embedder, ctl, names, retrieval_config, gpu_mb, say)
    except LLMError as exc:
        raise e2.RunBroken(f"a model server call failed: {exc}") from exc
    finally:
        e2._unload_models(ctl, mine, say)


def _run(root: Path, writer: LLMClient, judge_llm: LLMClient, embedder: Any, ctl: OllamaCtl, names: e2.Models,
         retrieval_config: dict[str, Any], gpu_mb: Callable[[], int | None], say: Callable[[str], None]) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    if is_dirty(root):
        raise e2.RunBroken("the working tree has uncommitted changes; commit first, or the record cannot be written")
    if retrieval_config.get("rerank") or retrieval_config.get("chunk_turns", 1) != 1:
        raise e2.RunBroken("the retrieval config needs a one-turn chunk and no reranker")
    try:
        versions = declared_versions()
    except (ValueError, KeyError, OSError) as exc:
        raise e2.RunBroken(f"the prompt versions are not valid: {exc}") from exc
    commit = head_commit(root)
    manifest_rel = SCENARIO_SOURCES[e2.SCENARIO_SOURCE][0]
    entry = json.loads((root / manifest_rel).read_text(encoding="utf-8"))["splits"][SPLIT]
    digest = split_hash(SPLIT, root, e2.SCENARIO_SOURCE)
    if digest != entry["hash"]:
        raise e2.RunBroken("the dev scenario file does not match its manifest")
    scenarios = load_scenarios(SPLIT, root / SCENARIO_SOURCES[e2.SCENARIO_SOURCE][1])
    missing = [m for m in (names.writer, names.judge, names.embedder) if full_name(m) not in ctl.installed()]
    if missing:
        raise e2.RunBroken(f"models not installed in Ollama: {missing}")

    # What each version really sends: the registry text plus whatever the agent code adds now. Two versions that end
    # up with the same system message would be one version under two labels.
    probe = Retriever(InMemoryTranscriptStore(), embedder)
    effective = {v.version: _digest(_build_assistant(v.version, writer, probe, MemoryStore(), retrieval_config).system)
                 for v in versions}
    if len(set(effective.values())) != len(effective):
        raise e2.RunBroken("two prompt versions send the same system message to the model")

    cold: dict[str, float] = {}
    gpu: dict[str, int | None] = {}
    say("checking the judge")
    cold["judge_check"] = ctl.load(names.judge)
    gpu["judge_loaded"] = gpu_mb()
    e2.Judge(judge_llm).canary()
    ctl.unload(names.judge)

    say("loading the writer and the embedder")
    cold["writer"] = ctl.load(names.writer)
    cold["embedder"] = ctl.load(names.embedder, embedding=True)
    t0 = time.perf_counter()
    try:
        writer.complete([{"role": "user", "content": 'Reply with {"ok": true}'}], schema=e2._Ping, max_tokens=20,
                        timeout_s=120.0)
        if not embedder.embed(["ping"])[0]:
            raise LLMError("the embedder returned an empty vector")
    except LLMError as exc:
        raise e2.RunBroken(f"the writer or the embedder is not answering: {exc}") from exc
    cold["first_calls"] = time.perf_counter() - t0
    gpu["writer_and_embedder_loaded"] = gpu_mb()

    memories: dict[str, MemoryStore] = {}
    memory_cost: list[dict[str, float]] = []
    memory_failed = 0
    say("building memory notes (shared by every version)")
    for sc in scenarios:
        memories[sc.id], cost = e2.build_memory(sc, writer)
        if cost is None:
            memory_failed += 1
        else:
            memory_cost.append(cost)
    if memory_failed / len(scenarios) > e2.BROKEN_MAX:
        raise e2.RunBroken(f"memory extraction failed on {memory_failed} of {len(scenarios)} cases")
    retrievers: dict[str, Retriever] = {}

    saved = _saved_rows(root / PARTIAL_REL, commit, {s.id for s in scenarios})
    if saved:
        say("reusing saved rows: " + ", ".join(f"{v} {len(r)}" for v, r in saved.items()))
    systems: dict[str, list[Prediction]] = {}
    health: dict[str, dict[str, Any]] = {}
    failed: dict[str, str] = {}
    for v in versions:
        say(f"running {v.version} ({v.factor})")

        def make(sc: Scenario, version: str = v.version) -> Assistant:
            if sc.id not in retrievers:
                retrievers[sc.id] = _retriever(sc, embedder)
            return _build_assistant(version, writer, retrievers[sc.id], memories[sc.id], retrieval_config)

        def keep(row: Prediction, version: str = v.version) -> None:
            line = json.dumps({"commit": commit, "version": version, "row": json.loads(row.model_dump_json())})
            e2._append_durable(root / PARTIAL_REL, line)

        rows = e2.run_system(v.version, scenarios, make, e2.STREAK_LOCAL, say=say, have=saved.get(v.version), on_row=keep)
        health[v.version], why = _health(v.version, rows)
        if why and v.version == BASELINE:
            raise e2.RunBroken(f"the baseline cannot be measured: {why}")
        if why:
            failed[v.version] = why
        else:
            systems[v.version] = rows
    gpu["after_generate"] = gpu_mb()
    ctl.unload(names.writer)

    say("judging")
    cold["judge"] = ctl.load(names.judge)
    gpu["judge_and_embedder_loaded"] = gpu_mb()
    judge = e2.Judge(judge_llm)
    systems = {n: e2.judge_rows(judge, p) for n, p in systems.items()}
    if judge.calls == 0:
        raise e2.JudgeFailed("no finding was judged, so no card made a claim")

    metrics = {n: e2._saved(scenarios, p) for n, p in systems.items()}
    intervals = {}
    for n, p in systems.items():
        names_ = [m for m in e2.INTERVAL_METRICS if metrics[n].get(m) is not None]
        got = {m: iv for m, iv in recompute_intervals(scenarios, p, names_).items() if iv}
        if got:
            intervals[n] = got
    comparisons = [{"a": n, "b": BASELINE, **c} for n in systems if n != BASELINE for m in e2.COMPARE_METRICS
                   if (c := recompute_comparison(scenarios, systems[n], systems[BASELINE], m))]
    choice = pick_best(metrics, comparisons, tuple(n for n in VERSIONS if n in systems))
    say(f"selected: {choice['selected']}")

    config: dict[str, Any] = {
        "experiment": "E6", "system": SYSTEM, "baseline": BASELINE, "factors": list(FACTORS),
        "versions": [{"version": v.version, "factor": v.factor, "summary": v.summary, "added": list(v.added),
                      "removed": list(v.removed), "text_sha256": v.text_sha256,
                      "system_message_sha256": effective[v.version]} for v in versions],
        "selection_rule": SELECTION_RULE, "selection_rule_sha256": _digest(SELECTION_RULE), "selection": choice,
        "no_measurable_improvement": choice["selected"] == BASELINE,
        "mode_factor": "not a factor: T4.2 sets respond mode in code; T3.3's mode-defining prompt was reverted",
        "deadline_s": e2.DEADLINE_S,
        "retrieval": {k: retrieval_config.get(k) for k in ("winner", "mode", "k", "rerank", "chunk_turns", "run_ids")},
        "judge": {"model": names.judge, "rubric_version": e2.RUBRIC_VERSION, "same_model_and_prompt_as_writer": False,
                  "calls": judge.calls, "share_supported": judge.yes / judge.calls},
        "health": health, "health_limits": {"broken": e2.BROKEN_MAX, "timeout": e2.TIMEOUT_MAX, "invalid": e2.INVALID_MAX},
        "unmeasured_versions": failed,
        "modes": {n: e2.mode_report(scenarios, p) for n, p in systems.items()},
        "injection_findings": {n: e2.injection_findings_report(scenarios, p) for n, p in systems.items()},
        "cold_start_s": cold, "gpu_memory_used_mb": gpu,
        "memory_build": {"cases": len(scenarios), "failed": memory_failed,
                         "mean_ms": sum(c["ms"] for c in memory_cost) / len(memory_cost) if memory_cost else None,
                         "note": "built once per case before the press and shared by every version"},
        "calls": call_counts(len(scenarios), len(versions)),
        "notes": [
            "grounded_rate and unsupported_claim_rate come from the local judge alone; no person has reviewed a sample "
            "(config.judge_review is absent), so they are not claims.",
            "The injected-claim counts are a word-overlap proxy, not a reader's judgement.",
            "Six candidates are compared with the baseline without a multiple-comparison correction.",
        ],
        "claude_calls": 0, "labels_reviewed": False, "started_at": started,
    }
    models = {"writer": names.writer, "judge": names.judge, "embedder": names.embedder}
    attempts = [{"setting": v.version, "kind": "prompt version", "factor": v.factor, "ok": v.version not in failed,
                 "error": failed.get(v.version), "health": health[v.version], "text_sha256": v.text_sha256}
                for v in versions]
    path = write_run("E6", SPLIT, config, models, {v.version: v.text_sha256 for v in versions}, metrics, intervals,
                     attempts, systems, digest, comparisons=comparisons, scenario_source=e2.SCENARIO_SOURCE, root=root)
    (root / PARTIAL_REL).unlink(missing_ok=True)  # the record holds them now
    say(f"wrote {path.name}")
    return {"selected": choice["selected"], "record": path, "choice": choice}


def main() -> int:
    from contextlens.llm.client import Embedder, OllamaClient

    try:
        result = run_e6(writer=OllamaClient(e2.WRITER_MODEL), judge_llm=OllamaClient(e2.JUDGE_MODEL),
                        embedder=Embedder(e2.EMBED_MODEL), ctl=OllamaCtl(),
                        retrieval_config=e2.load_retrieval_config(ROOT / e2.CONFIG_REL))
    except e2.RunBroken as exc:
        print(f"E6 stopped, nothing written: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"selected": result["selected"], "record": result["record"].as_posix(),
                      "qualifies": result["choice"]["qualifies"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
