"""Prompt regression recordings: the real writer's raw replies for 30 invented dev cases, and an offline replay of them.

Record (needs Ollama and a GPU turn, a clean tree):  python -m contextlens.experiments.regression
The recording holds the invented scenarios, every raw model reply in order (memory notes first, then the agent's steps),
the card summary the code produced, and the digest of the system message the agent sent. Replaying feeds the replies
back through the same code, with no model; a changed prompt changes the digest and so fails until it is re-recorded.
"""
import hashlib
import json
import os
import random
import re
import sys
import threading
import zlib
from pathlib import Path
from typing import Any

from contextlens.agents.single import SingleAgent
from contextlens.agents.tools import Tools
from contextlens.data.scenarios import load_scenarios
from contextlens.eval.records import ROOT, SCENARIO_SOURCES, head_commit, is_dirty, split_hash
from contextlens.experiments import e2_context as e2
from contextlens.llm.client import FakeLLM, LLMClient, LLMError, LLMResult, _Base
from contextlens.memory.extract import MemoryStore
from contextlens.replay.runner import Context
from contextlens.retrieval.hybrid import Retriever
from contextlens.schemas import Card, Scenario

PROMPT_VERSION = "v3"  # chosen by E6 (results/runs/E6-20261105T223904724358Z.json)
SYSTEM = "single"
SEED = "t7.2-regression-1"
N_CASES = 30
RECORDED_REL = Path("tests/regression/recorded/v3-single.json")


class ReplayMismatch(RuntimeError):
    """The code no longer consumes the recorded replies the way it did when they were recorded."""


class HashEmbedder:
    """Tool results do not reach the recorded replies, so replay needs only some stable vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * 32
            for w in re.findall(r"\w+", t.lower()):
                v[zlib.crc32(w.encode()) % 32] += 1.0
            out.append(v)
        return out


class RecordingLLM(_Base):
    """Passes each call to `inner` and keeps the model's text, retries included, in call order."""

    def __init__(self, inner: Any) -> None:
        self.inner, self.model, self.texts = inner, getattr(inner, "model", ""), []
        self._lock = threading.Lock()

    def _raw(self, messages, schema, max_tokens, temperature, timeout_s) -> LLMResult:
        res = self.inner._raw(messages, schema, max_tokens, temperature, timeout_s)
        with self._lock:
            self.texts.append(res.text)
        return res


def select_cases(scenarios: list[Scenario], n: int = N_CASES, seed: str = SEED) -> list[Scenario]:
    """About n/k cases from each of the k categories (the first n % k alphabetically get one more), drawn with `seed`
    from invented dev cases only."""
    pool = sorted((s for s in scenarios if s.origin == "synthetic" and s.conversation.source == "synthetic"
                   and s.conversation.split == "dev"), key=lambda s: s.id)
    cats = sorted({s.category for s in pool})
    out: list[Scenario] = []
    for i, cat in enumerate(cats):
        mine = [s for s in pool if s.category == cat]
        out += sorted(random.Random(f"{seed}:{cat}").sample(mine, n // len(cats) + (i < n % len(cats))), key=lambda s: s.id)
    return out


_EMPTY = Scenario.model_validate({
    "id": "x", "conversation": {"id": "x", "turns": [], "source": "synthetic"},
    "request": {"conversation_id": "x", "as_of_seq": 0}, "category": "recall", "gold_evidence_turn_ids": [], "gold_label": None, "gold_answer": None,
    "origin": "synthetic"})


def _agent(llm: LLMClient, sc: Scenario, embedder: Any, memory: MemoryStore) -> tuple[SingleAgent, Context]:
    store = e2.build_store(sc)
    tools = Tools(store, Retriever(store, embedder), memory, sc.request.as_of_seq, sc.conversation.id)
    return SingleAgent(llm, tools, PROMPT_VERSION, deadline_s=e2.DEADLINE_S), Context(store=store, as_of_seq=sc.request.as_of_seq)


def system_sha256() -> str:
    """The digest of the system message the agent sends now (the E6 record's `system_message_sha256`)."""
    agent = _agent(FakeLLM([]), _EMPTY, HashEmbedder(), MemoryStore())[0]
    return hashlib.sha256(agent.system.encode("utf-8")).hexdigest()


def _run(sc: Scenario, llm: LLMClient, embedder: Any) -> tuple[Card, list[Any]]:
    memory, cost = e2.build_memory(sc, llm)
    if cost is None:
        raise e2.RunBroken(f"{sc.id}: memory notes failed")
    agent, ctx = _agent(llm, sc, embedder, memory)
    return agent.answer(sc.request, ctx), ctx.drops


def summarize(card: Card, drops: list[Any]) -> dict[str, Any]:
    """What a regression asserts: the status, each finding's label and citations, and why findings were dropped."""
    return {
        "mode": card.mode, "status": card.status,
        "findings": [{"kind": f.kind, "label": f.label, "evidence": [[e.turn_id, e.quote] for e in f.evidence]}
                     for f in card.findings],
        "drops": [[d.reason, list(d.turn_ids)] for d in drops],
    }


def replay_case(entry: dict[str, Any], embedder: Any | None = None) -> dict[str, Any]:
    sc = Scenario.model_validate(entry["scenario"])
    llm = FakeLLM(list(entry["calls"]))
    try:
        card, drops = _run(sc, llm, embedder or HashEmbedder())
    except e2.RunBroken as exc:
        raise ReplayMismatch(f"{sc.id}: {exc}") from exc
    if llm._responses:
        raise ReplayMismatch(f"{sc.id}: {len(llm._responses)} recorded replies were never asked for")
    return summarize(card, drops)


def record_case(sc: Scenario, writer: LLMClient, embedder: Any) -> dict[str, Any]:
    if sc.origin != "synthetic" or sc.conversation.source != "synthetic":
        raise e2.RunBroken(f"{sc.id}: only invented cases may be recorded")
    rec = RecordingLLM(writer)
    try:
        card, drops = _run(sc, rec, embedder)
    except LLMError as exc:
        raise e2.RunBroken(f"{sc.id}: a model call failed: {type(exc).__name__}") from exc
    if any(m.startswith(e2.BROKEN_PREFIXES) or m.startswith(e2.AGENT_PREFIXES) for m in card.missing):
        raise e2.RunBroken(f"{sc.id}: the card is an error card: {card.missing}")
    entry = {"id": sc.id, "category": sc.category, "scenario": json.loads(sc.model_dump_json()),
             "calls": rec.texts, "expected": summarize(card, drops)}
    if replay_case(entry, embedder) != entry["expected"]:
        raise e2.RunBroken(f"{sc.id}: the recording does not replay to the same card (a call may have timed out)")
    return entry


def record_all(root: Path, scenarios: list[Scenario], writer: LLMClient, embedder: Any, writer_model: str,
               say: Any = print) -> dict[str, Any]:
    if is_dirty(root):
        raise e2.RunBroken("the working tree has uncommitted changes; commit first")
    picked = select_cases(scenarios)
    cases = []
    for i, sc in enumerate(picked, 1):
        say(f"recording {i}/{len(picked)} {sc.id}")
        cases.append(record_case(sc, writer, embedder))
    return {"prompt_version": PROMPT_VERSION, "system": SYSTEM, "system_message_sha256": system_sha256(),
            "seed": SEED, "writer_model": writer_model, "commit": head_commit(root), "cases": cases}


def main() -> int:
    from contextlens.llm.client import Embedder, OllamaClient
    from contextlens.llm.ollama_ctl import OllamaCtl, full_name

    ctl = OllamaCtl()
    try:
        manifest = json.loads((ROOT / SCENARIO_SOURCES[e2.SCENARIO_SOURCE][0]).read_text(encoding="utf-8"))
        if split_hash("dev", ROOT, e2.SCENARIO_SOURCE) != manifest["splits"]["dev"]["hash"]:
            raise e2.RunBroken("the dev scenario file does not match its manifest")
        scenarios = load_scenarios("dev", ROOT / SCENARIO_SOURCES[e2.SCENARIO_SOURCE][1])
        if is_dirty(ROOT):
            raise e2.RunBroken("the working tree has uncommitted changes; commit first")
        missing = [m for m in (e2.WRITER_MODEL, e2.EMBED_MODEL) if full_name(m) not in ctl.installed()]
        if missing:
            raise e2.RunBroken(f"models not installed in Ollama: {missing}")
        ctl.load(e2.WRITER_MODEL)
        ctl.load(e2.EMBED_MODEL, embedding=True)
        try:
            doc = record_all(ROOT, scenarios, OllamaClient(e2.WRITER_MODEL), Embedder(e2.EMBED_MODEL), e2.WRITER_MODEL,
                             say=lambda m: print(m, file=sys.stderr))
        finally:
            ctl.unload(e2.WRITER_MODEL)
            ctl.unload(e2.EMBED_MODEL, embedding=True)
    except e2.RunBroken as exc:
        print(f"recording stopped, nothing written: {exc}", file=sys.stderr)
        return 2
    path = ROOT / RECORDED_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    print(json.dumps({"recorded": len(doc["cases"]), "file": RECORDED_REL.as_posix(),
                      "system_message_sha256": doc["system_message_sha256"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
