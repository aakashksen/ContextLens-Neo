"""The agent team as a LangGraph graph: freeze -> (memory | consistency) in parallel -> assistance."""

import functools
import threading
import time
from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from contextlens.agents import consistency_agent, memory_agent
from contextlens.agents.assistance_agent import decide
from contextlens.agents.evidence import EvidenceStore
from contextlens.agents.single import (
    LATEST_EXCHANGE,
    MARGIN_S,
    _Abandoned,
    _abstained,
    _bounded,
    _finish,
    _recent,
    _Run,
)
from contextlens.agents.specialist import Outcome, run_specialist
from contextlens.agents.tools import Tools
from contextlens.llm.client import LLMClient, LLMResult
from contextlens.replay.runner import Context
from contextlens.schemas import ButtonRequest, Card, Turn

AGENTS = ("memory", "consistency", "assistance")


@dataclass
class _Job:
    """Everything one button press needs. Agents share the evidence store and nothing else."""

    req: ButtonRequest
    ctx: Context
    start: float
    end: float
    evidence: EvidenceStore = field(default_factory=EvidenceStore)
    turns: list[Turn] = field(default_factory=list)
    runs: dict[str, _Run] = field(default_factory=lambda: {n: _Run() for n in AGENTS})
    ms: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    degraded: bool = False
    card: Card | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class _State(TypedDict):
    job: _Job


def _tokens(results: list[LLMResult]) -> int:
    return sum(r.prompt_tokens + r.completion_tokens for r in results)


class TeamAssistant:
    """Memory and consistency agents work in parallel; the assistance agent writes the card from their verified
    findings and may ask one specialist for one follow-up. A card is returned within deadline_s + MARGIN_S."""

    def __init__(
        self, llm_by_agent: dict[str, LLMClient], tools: Tools, prompt_version: str,
        deadline_s: float = 8.0, max_followups: int = 1,
    ) -> None:  # fmt: skip
        self.llms, self.tools, self.deadline_s, self.max_followups = llm_by_agent, tools, deadline_s, max_followups
        has_checker = tools.checker is not None
        self._specs = {
            "memory": (memory_agent.KIND, memory_agent.system(prompt_version), memory_agent.TOOLS),
            "consistency": (
                consistency_agent.KIND,
                consistency_agent.system(prompt_version, has_checker),
                consistency_agent.tool_names(has_checker),
            ),
        }
        g = StateGraph(_State)
        g.add_node("freeze", self._freeze)
        g.add_node("memory", functools.partial(self._specialist, "memory"))
        g.add_node("consistency", functools.partial(self._specialist, "consistency"))
        g.add_node("assist", self._assist)
        g.add_edge(START, "freeze")
        g.add_conditional_edges("freeze", self._route, ["memory", "consistency", END])
        g.add_edge(["memory", "consistency"], "assist")
        g.add_edge("assist", END)
        self.graph = g.compile()

    def answer(self, req: ButtonRequest, ctx: Context) -> Card:
        start = time.perf_counter()
        job = _Job(req, ctx, start, start + self.deadline_s)
        self.graph.invoke({"job": job})
        assert job.card is not None
        return job.card

    # -- controller --

    def _freeze(self, state: _State) -> dict:
        job = state["job"]
        try:
            job.turns = _bounded(
                functools.partial(_recent, job.ctx, job.req, LATEST_EXCHANGE), self.deadline_s + MARGIN_S
            )
        except _Abandoned:
            job.card = _abstained("the store did not answer in time")
        except Exception as exc:
            job.card = _abstained(f"store failed: {type(exc).__name__}")
        else:
            if not job.turns:
                job.card = _abstained("the conversation has no turns yet")
        return {}

    @staticmethod
    def _route(state: _State) -> list[str]:
        return [END] if state["job"].card is not None else ["memory", "consistency"]

    # -- specialists --

    def _specialist(self, name: str, state: _State, request: str | None = None) -> dict:
        job = state["job"]
        kind, system, tool_names = self._specs[name]
        t0 = time.perf_counter()
        try:
            out = run_specialist(
                name=name, kind=kind, system=system, tool_names=tool_names, llm=self.llms[name],
                tools=self.tools, ctx=job.ctx, req=job.req, turns=job.turns, end=job.end,
                run=job.runs[name], request=request,
            )  # fmt: skip
        except Exception as exc:  # one agent breaking must not take the others down
            out = Outcome(notes=[f"{name}: failed: {type(exc).__name__}"], degraded=True)
        for f in out.findings:
            job.evidence.add(f)
        with job.lock:
            job.notes += out.notes
            job.degraded = job.degraded or out.degraded
            job.ms[name] = job.ms.get(name, 0.0) + (time.perf_counter() - t0) * 1000
        return {}

    # -- assistance --

    def _assist(self, state: _State) -> dict:
        job = state["job"]
        req, ctx = job.req, job.ctx
        llm = self.llms["assistance"]
        mine = job.runs["assistance"]
        t0 = time.perf_counter()
        mode, note = "recall", None
        if job.evidence.all():
            mode, note = decide(
                llm, req, job.evidence.all, lambda agent, text: self._specialist(agent, state, text),
                job.end, mine, self.max_followups,
            )  # fmt: skip
        findings = job.evidence.all()
        missing = [*job.notes, *([note] if note else [])]
        if findings:
            status = "partial" if job.degraded or note else "answered"
            card = Card(mode=mode, status=status, findings=findings, missing=missing)
        else:
            card = _abstained("no verified findings")
            card = card.model_copy(update={"missing": [*missing, *card.missing]})
        total = _Run()
        total.results = [r for n in AGENTS[:2] for r in job.runs[n].results] + mine.results
        total.abandoned = sum(r.abandoned for r in job.runs.values())
        spec_n = len(total.results) - len(mine.results)
        card = _finish(card, ctx, req, job.start, total, llm, job.end, MARGIN_S)
        job.ms["assistance"] = job.ms.get("assistance", 0.0) + (time.perf_counter() - t0) * 1000
        job.card = card
        self._record(job, total, spec_n)
        return {}

    @staticmethod
    def _record(job: _Job, total: _Run, spec_n: int) -> None:
        stage = job.ctx.stage_ms
        for name in AGENTS[:2]:
            stage[f"{name}_tokens"] = _tokens(job.runs[name].results)
        stage["assistance_tokens"] = _tokens(total.results[spec_n:])
        for name in AGENTS:
            stage[f"{name}_ms"] = int(job.ms.get(name, 0.0))
        if total.abandoned:
            stage["abandoned_calls"] = total.abandoned
