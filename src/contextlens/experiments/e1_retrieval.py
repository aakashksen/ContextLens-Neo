"""E1: which retrieval setup finds evidence best?

Eight systems on the dev split only: dense, bm25, hybrid and hybrid + rerank, each over one-turn chunks (t1)
and 3-turn windows (w3). Our scenarios and QMSum are scored in separate records. The winner comes from a rule
fixed here (see `pick_winner`), never from the test split, which this module does not read.

Run: .venv/Scripts/python -m contextlens.experiments.e1_retrieval
"""
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contextlens.data.scenarios import load_scenarios
from contextlens.eval.check_run import committed_unchanged, recompute_comparison, recompute_intervals, recompute_metrics
from contextlens.eval.records import ROOT, SCENARIO_SOURCES, head_commit, split_hash, write_run
from contextlens.eval.series import base_name
from contextlens.memory.store import InMemoryTranscriptStore
from contextlens.replay.runner import Prediction
from contextlens.retrieval.hybrid import Retriever, ScoredTurn
from contextlens.schemas import Scenario, Turn

SPLIT = "dev"
KINDS = ("dense", "bm25", "hybrid", "hybrid_rerank")
CHUNKS = {"t1": 1, "w3": 3}  # one turn, or a window of the turn and the two before it
SYSTEMS = [f"{kind}_{chunk}" for chunk in CHUNKS for kind in KINDS]  # also the tie-break order
SOURCES = ("scenarios", "qmsum")
K = 10  # turns kept in each ranking; the largest k in the metrics
POOL = 30  # candidates the reranker reorders
KS = (1, 3, 5, 10)
SAVED = {"recall@1", "recall@3", "recall@5", "recall@10", "mrr", "latency_p50_ms", "latency_p95_ms"}
WINNER_RULE = ("highest mean over the sources of recall@5; ties broken by higher mean mrr, then by the order of "
               "SYSTEMS (one-turn chunks before windows, then dense, bm25, hybrid, hybrid_rerank)")
QUERY_RULE = ("the question when there is one; otherwise the text of the last turn, searched over the turns "
              "before it")
CONFIG_REL = Path("configs/retrieval.json")


class RerankerFallback(RuntimeError):
    """The reranker did not run, so a '+ rerank' result would be unranked."""


class BatchedEmbedder:
    """Splits big embedding calls into requests the server answers in time, and remembers one-text calls."""

    def __init__(self, inner: Any, batch: int = 32) -> None:
        self.inner, self.batch, self.model = inner, batch, getattr(inner, "model", "unknown")
        self._single: dict[str, list[float]] = {}

    def embed(self, texts: list[str]) -> list[list[float]]:
        if len(texts) == 1 and texts[0] in self._single:
            return [self._single[texts[0]]]
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            out += self.inner.embed(texts[i : i + self.batch])
        if len(texts) == 1:
            self._single[texts[0]] = out[0]
        return out


def chunks(turns: list[Turn], size: int) -> tuple[list[Turn], dict[str, list[str]]]:
    """Search units and, for each window, the turn ids it holds. A window gets the seq of its last turn, so
    filtering on as_of_seq never lets a later turn in."""
    turns = sorted(turns, key=lambda t: t.seq)
    if size == 1:
        return turns, {}
    units, members = [], {}
    for i, last in enumerate(turns):
        held = turns[max(0, i - size + 1) : i + 1]
        unit = last.model_copy(update={
            "id": f"{last.conversation_id}:w{last.seq}", "start_ms": held[0].start_ms,
            "text": "\n".join(f"{t.speaker}: {t.text}" for t in held),
        })
        units.append(unit)
        members[unit.id] = [t.id for t in held]
    return units, members


def query_for(sc: Scenario) -> tuple[str, int]:
    """The query text and the last seq that may be searched."""
    if sc.request.question:
        return sc.request.question, sc.request.as_of_seq
    last = max((t for t in sc.conversation.turns if t.seq <= sc.request.as_of_seq), key=lambda t: t.seq)
    return last.text, sc.request.as_of_seq - 1


def ranked_turns(hits: list[ScoredTurn], members: dict[str, list[str]], k: int = K) -> list[str]:
    """Turn ids in the order their windows were ranked, each once."""
    seen: dict[str, None] = {}
    for h in hits:
        for tid in members.get(h.turn.id, [h.turn.id]):
            seen.setdefault(tid)
    return list(seen)[:k]


def _search(retriever: Retriever, reranker: Any, kind: str, conv_id: str, query: str, last: int) -> list[ScoredTurn]:
    if kind == "hybrid_rerank":
        pool = retriever.search(conv_id, query, last, k=POOL, mode="hybrid")
        hits = reranker.rerank(query, pool, K)
        if getattr(reranker, "fallback_reason", None):
            raise RerankerFallback(f"the reranker did not run: {reranker.fallback_reason}")
        return hits
    return retriever.search(conv_id, query, last, k=K, mode=kind)  # type: ignore[arg-type]


def evaluate(scenarios: list[Scenario], embedder: Any, reranker: Any, say=print) -> tuple[dict[str, list[Prediction]], dict[str, str]]:
    """Predictions for each system that finished every scenario, and the error of each that did not."""
    by_conv: dict[str, list[Scenario]] = defaultdict(list)
    for sc in scenarios:
        by_conv[sc.conversation.id].append(sc)
    rows: dict[str, dict[str, Prediction]] = {name: {} for name in SYSTEMS}
    failed: dict[str, str] = {}
    for n, (conv_id, group) in enumerate(by_conv.items(), 1):
        for chunk, size in CHUNKS.items():
            store = InMemoryTranscriptStore()
            units, members = chunks(group[0].conversation.turns, size)
            for u in units:
                store.add(u)
            retriever = Retriever(store, embedder)
            for sc in group:
                query, last = query_for(sc)
                if last >= 0:
                    retriever.search(conv_id, query, last, k=1, mode="dense")  # index the chunks outside the timing
                for kind in KINDS:
                    name = f"{kind}_{chunk}"
                    if name in failed:
                        continue
                    start = time.perf_counter()
                    try:
                        ids = ranked_turns(_search(retriever, reranker, kind, conv_id, query, last), members) if last >= 0 else []
                    except RerankerFallback:
                        raise
                    except Exception as exc:
                        failed[name] = f"{type(exc).__name__}: {exc}"
                        say(f"system {name} failed: {failed[name]}")
                        continue
                    ms = int((time.perf_counter() - start) * 1000)
                    rows[name][sc.id] = Prediction(scenario_id=sc.id, ranked_turn_ids=ids, stage_latency_ms={"total": ms})
        if n % 10 == 0 or n == len(by_conv):
            say(f"  {n}/{len(by_conv)} conversations")
    order = [sc.id for sc in scenarios]
    return {name: [r[i] for i in order] for name, r in rows.items() if name not in failed}, failed


def pick_winner(scores: dict[str, dict[str, dict[str, float]]]) -> str:
    """scores[source][system] = {"recall@5", "mrr"}. The rule is WINNER_RULE."""
    systems = set.intersection(*(set(s) for s in scores.values()))

    def mean(name: str, metric: str) -> float:
        return sum(scores[src][name][metric] for src in scores) / len(scores)

    return min(sorted(systems, key=SYSTEMS.index), key=lambda n: (-mean(n, "recall@5"), -mean(n, "mrr")))


def _load(source: str, root: Path) -> tuple[list[Scenario], str]:
    """The dev scenarios of a source and the hash the record will carry; only dev is ever loaded."""
    scenario_dir = SCENARIO_SOURCES[source][1]
    manifest_rel = SCENARIO_SOURCES[source][0]
    entry = json.loads((root / manifest_rel).read_text(encoding="utf-8"))["splits"][SPLIT]
    digest = split_hash(SPLIT, root, source)
    if digest != entry["hash"]:
        raise RuntimeError(f"the {source} scenario file does not match its manifest")
    return load_scenarios(SPLIT, root / scenario_dir), digest


def _proof(reranker: Any) -> None:
    """Fail now, before any work, if the reranker cannot run."""
    turn = Turn(id="p:t0", conversation_id="p", seq=0, speaker="A", text="a", start_ms=0, end_ms=1, source="synthetic")
    reranker.rerank("a", [ScoredTurn(turn=turn, score=1.0, parts={}), ScoredTurn(turn=turn, score=0.5, parts={})], 2)
    if getattr(reranker, "fallback_reason", None):
        raise RerankerFallback(f"the reranker did not run: {reranker.fallback_reason}")


def run_e1(*, root: Path = ROOT, embedder: Any, reranker: Any, say=print) -> dict[str, Any]:
    """Run every system on every source, write one record per source, then the config of the winner.
    A reranker that did not run fails the run before anything is written. A system that errors is listed in
    `attempts` and left out; then there is no winner and no config."""
    if not committed_unchanged({"git_commit": head_commit(root)}, root, SCENARIO_SOURCES["qmsum"][0].as_posix()):
        raise RuntimeError("the qmsum manifest must be committed before the run")
    started = datetime.now(timezone.utc).isoformat()
    embedder = BatchedEmbedder(embedder)
    embedder.embed(["ping"])
    _proof(reranker)
    done: dict[str, tuple[list[Scenario], str, dict[str, list[Prediction]], dict[str, str]]] = {}
    for source in SOURCES:
        scenarios, digest = _load(source, root)
        say(f"{source}: {len(scenarios)} dev scenarios")
        preds, failed = evaluate(scenarios, embedder, reranker, say)
        done[source] = (scenarios, digest, preds, failed)
    if any(not preds for _, _, preds, _ in done.values()):
        raise RuntimeError("a source has no system that finished; nothing is written")

    metrics = {src: {n: {k: v for k, v in recompute_metrics(scs, p[n], KS).items() if base_name(k) in SAVED}
                     for n in p} for src, (scs, _, p, _) in done.items()}
    complete = all(not failed for *_, failed in done.values())
    winner = pick_winner(metrics) if complete else None
    paths = {}
    for source, (scs, digest, preds, failed) in done.items():
        attempts = [{"setting": n, "kind": n.rsplit("_", 1)[0], "chunk_turns": CHUNKS[n.rsplit("_", 1)[1]],
                     "ok": n in preds, "error": failed.get(n)} for n in SYSTEMS]
        intervals = {n: {"recall@5": iv} for n in preds if (iv := recompute_intervals(scs, preds[n], ["recall@5"])["recall@5"])}
        comparisons = []
        for other in preds if winner else []:
            if other != winner and (c := recompute_comparison(scs, preds[winner], preds[other], "recall@5")):
                comparisons.append({"a": winner, "b": other, **c})
        config = {"experiment": "E1", "k": K, "pool": POOL, "chunks": CHUNKS, "query_rule": QUERY_RULE,
                  "winner_rule": WINNER_RULE, "winner": winner, "started_at": started, "labels_reviewed": False}
        paths[source] = write_run(
            "E1", SPLIT, config, {"embedder": embedder.model, "reranker": reranker.model_name}, {}, metrics[source],
            intervals, attempts, preds, digest, comparisons=comparisons, scenario_source=source, root=root)
    if not complete:
        return {"winner": None, "complete": False, "records": paths}
    cfg = _config(winner, done, metrics, paths, reranker)
    (root / CONFIG_REL).parent.mkdir(parents=True, exist_ok=True)
    (root / CONFIG_REL).write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
    return {"winner": winner, "complete": True, "records": paths}


def _config(winner: str, done: dict, metrics: dict, paths: dict[str, Path], reranker: Any) -> dict[str, Any]:
    kind, chunk = winner.rsplit("_", 1)
    undecided: dict[str, list[str]] = defaultdict(list)  # others the winner does not measurably beat
    for source, path in paths.items():
        for c in json.loads(path.read_text(encoding="utf-8"))["comparisons"]:
            if c["ci"][0] <= 0 <= c["ci"][1]:
                undecided[c["b"]].append(source)
    return {
        "split": SPLIT, "winner": winner, "mode": kind.replace("_rerank", ""), "rerank": kind.endswith("rerank"),
        "chunk_turns": CHUNKS[chunk], "k": K, "pool": POOL, "reranker_model": reranker.model_name,
        "rule": WINNER_RULE, "query_rule": QUERY_RULE,
        "scores": {src: {n: {k: m[k] for k in ("recall@5", "mrr")} for n, m in ms.items()} for src, ms in metrics.items()},
        "no_measurable_difference_from": dict(undecided),
        "run_ids": {src: p.stem for src, p in paths.items()},
    }


def main() -> int:
    from contextlens.llm.client import Embedder
    from contextlens.retrieval.rerank import Reranker

    result = run_e1(embedder=Embedder(), reranker=Reranker())
    print(json.dumps({"winner": result["winner"], "complete": result["complete"],
                      "records": [p.as_posix() for p in result["records"].values()]}, indent=1))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    sys.exit(main())
