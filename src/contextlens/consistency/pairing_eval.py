"""Offline evaluation of conflict pairing on E10's labelled cases (conflict triplets and short controls), no LLM calls.

    python -m contextlens.consistency.pairing_eval            # dev, the selection split
    python -m contextlens.consistency.pairing_eval --test     # test: run once, for the one chosen variant

Reports, per variant: recall of contradiction+update, label accuracy, cited-turn accuracy, false-alert rate on
consistent/insufficient cases, and the median / 95th-percentile time of the call. Embeddings (the transcript search) come
from the local Ollama embedder (not generation).

Today's path is approximated by its first stage alone: P(contradiction) >= the cascade's flag threshold counts as a
confirmed contradiction (the prompted second stage is not called, so this is the best the cascade could do), otherwise the
trained model's best non-contradiction label.
"""
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from contextlens.consistency import pairing as P
from contextlens.data.e10_cases import load_cases
from contextlens.memory.store import InMemoryTranscriptStore
from contextlens.retrieval.hybrid import Retriever
from contextlens.schemas import Scenario, Turn

CHECKER = "<local path>"
MODERN = "tasksource/ModernBERT-base-nli"
XSMALL = "cross-encoder/nli-deberta-v3-xsmall"
FLAG_THRESHOLD = 0.95  # configs/cascade.json
POS = ("cf-contradiction", "cf-update", "ct-contradiction", "ct-update")
NEG = ("cf-consistent", "ct-consistent", "ct-insufficient")


def cases(split: str) -> list[Scenario]:
    scs, _, _ = load_cases(split)
    return [s for s in scs if s.gold_label in ("contradiction", "update", "consistent", "insufficient")
            and s.category.startswith(("conflict", "control"))]


def group(sc: Scenario) -> str:
    return ("cf-" if sc.category.startswith("conflict") else "ct-") + str(sc.gold_label)


class SearchTools:
    """The part of agents.tools.Tools that `find_conflict` uses."""

    def __init__(self, retriever: Retriever, conv_id: str, as_of_seq: int) -> None:
        self.r, self.conv_id, self.as_of = retriever, conv_id, as_of_seq

    def search_transcript(self, query: str, k: int = 8) -> list[Any]:
        return self.r.search(self.conv_id, query, self.as_of, k=k)


def today_label(trained: Any, retr: Retriever, sc: Scenario, st: Turn, window: list[Turn]) -> tuple[str, list[str]]:
    hits = [h.turn for h in retr.search(sc.conversation.id, st.text, sc.request.as_of_seq, k=8) if h.turn.seq < st.seq]
    keep = {t.id: t for t in hits}
    for t in reversed(window):
        if len(keep) >= 12:
            break
        keep.setdefault(t.id, t)
    earlier = sorted(keep.values(), key=lambda t: t.seq)
    return stage1(trained, st, earlier), [t.id for t in earlier]


def stage1(trained: Any, st: Turn, earlier: list[Turn]) -> str:
    p = trained.probabilities(st, earlier)
    if p["contradiction"] >= FLAG_THRESHOLD:
        return "contradiction"
    lab = max((x for x in p if x != "contradiction"), key=p.__getitem__)
    return lab if p[lab] >= 0.5 else "insufficient"


def summarize(name: str, rows: list[tuple[Scenario, str, list[str]]], times: list[float]) -> str:
    res: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])  # n, flagged, label right, cited right
    for sc, lab, ids in rows:
        x = res[group(sc)]
        gold_prev = {g for g in sc.gold_evidence_turn_ids}
        x[0] += 1
        x[1] += lab in ("update", "contradiction")
        x[2] += lab == sc.gold_label
        x[3] += lab in ("update", "contradiction") and bool(set(ids) & gold_prev)
    n = sum(res[g][0] for g in POS)
    rc, acc, cite = (sum(res[g][i] for g in POS) for i in (1, 2, 3))
    fa, nf = sum(res[g][1] for g in NEG), sum(res[g][0] for g in NEG)

    def d(g: str) -> str:
        return f"{res[g][1]}/{res[g][0]}"
    lat = f" | ms p50 {statistics.median(times):.0f} p95 {sorted(times)[int(0.95 * (len(times) - 1))]:.0f}" if times else ""
    return (f"{name:34s} recall {rc}/{n} ({rc / max(n, 1):.0%}) label {acc}/{n} cite {cite}/{n} | cfContra {d('cf-contradiction')} "
            f"cfUpd {d('cf-update')} ctContra {d('ct-contradiction')} ctUpd {d('ct-update')} | false alerts {fa}/{nf} "
            f"(cfCons {d('cf-consistent')} ctCons {d('ct-consistent')} ctIns {d('ct-insufficient')}){lat}")


def run(split: str, variants: list[tuple[str, P.Rule, tuple[str, ...], bool]], say=print) -> None:
    from contextlens.experiments.e5_cards import EmbedCache
    from contextlens.llm.client import Embedder

    trained, _ = P.load_models(CHECKER)
    emb = EmbedCache(Embedder())
    prepared = []
    for sc in cases(split):
        turns = sorted((t for t in sc.conversation.turns if t.seq <= sc.request.as_of_seq), key=lambda t: t.seq)
        store = InMemoryTranscriptStore()
        for t in turns:
            store.add(t)
        retr = Retriever(store, emb)
        emb.embed([t.text for t in turns])  # the transcript's vectors are warm, as they are in a running meeting
        prepared.append((sc, turns[-1], turns[:-1], retr))
    rows = [(sc, *today_label(trained, retr, sc, st, window)) for sc, st, window, retr in prepared]
    say(summarize("today (stage 1, pile of <=12)", rows, []))
    rows = []
    for sc, st, window, _ in prepared:
        gold = [t for t in window if t.id in sc.gold_evidence_turn_ids]
        rows.append((sc, stage1(trained, st, gold) if gold else "insufficient", [t.id for t in gold]))
    say(summarize("gold earlier turn alone (stage 1)", rows, []))
    for name, rule, nli_names, use_tools in variants:
        _, nlis = P.load_models(CHECKER, nli_names)
        for sc, st, window, retr in prepared[:2]:  # warm up kernels
            P.find_conflict(SearchTools(retr, sc.conversation.id, sc.request.as_of_seq) if use_tools else None,
                            st, window, trained=trained, nlis=nlis, rule=rule)
        rows, times = [], []
        for sc, st, window, retr in prepared:
            tools = SearchTools(retr, sc.conversation.id, sc.request.as_of_seq) if use_tools else None
            t0 = time.perf_counter()
            lab, _, ids = P.find_conflict(tools, st, window, trained=trained, nlis=nlis, rule=rule)
            times.append((time.perf_counter() - t0) * 1000)
            rows.append((sc, lab, ids))
        say(summarize(name, rows, times))


VARIANTS: list[tuple[str, P.Rule, tuple[str, ...], bool]] = [
    ("pair: trained only", P.Rule(), (), False),
    ("pair: trained only +search", P.Rule(), (), True),
    ("pair: ModernBERT NLI t.5", P.Rule(nli_threshold=0.5), (MODERN,), False),
    ("pair: ModernBERT NLI t.7", P.Rule(nli_threshold=0.7), (MODERN,), False),
    ("pair: ModernBERT NLI t.7 +search", P.Rule(nli_threshold=0.7), (MODERN,), True),
    ("pair: ModernBERT t.7 no gates", P.Rule(nli_threshold=0.7, speaker_gate=False, hedge_gate=False, restate_gate=False),
     (MODERN,), False),
    ("pair: xsmall NLI t.7", P.Rule(nli_threshold=0.7), (XSMALL,), False),
    ("pair: Modern+xsmall mean t.5", P.Rule(nli_threshold=0.5), (MODERN, XSMALL), False),
    ("pair: Modern+xsmall mean t.7", P.Rule(nli_threshold=0.7), (MODERN, XSMALL), False),
    ("pair: ModernBERT t.7 no rescue", P.Rule(nli_threshold=0.7, update_rescue=None), (MODERN,), False),
]


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    test = "--test" in args
    if test:  # the single chosen variant only
        chosen = [v for v in VARIANTS if v[0] == "pair: ModernBERT NLI t.7 +search"]
        print("TEST SPLIT, single chosen variant", file=sys.stderr)
        run("test", chosen)
    else:
        run("dev", VARIANTS)
    return 0


if __name__ == "__main__":
    Path(".").resolve()
    raise SystemExit(main())
