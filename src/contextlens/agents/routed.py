"""`team_routed`: one specialist per press, chosen by code, and a checker that runs in code.

Before any model call: search the transcript, run the consistency checker (agents/precheck.py). Then
- a question was asked: the memory specialist; no question: the assistance specialist (what is relevant to the latest exchange);
- the checker said contradiction or update: the consistency specialist also runs, and its findings carry the checker's label;
- findings are merged (same cited turns, or the same statement, count once), then the usual checks and abstention apply.
No model decides who runs or what the mode is."""

import functools
import time

from contextlens.agents import consistency_agent, memory_agent
from contextlens.agents.precheck import Prep, prepare
from contextlens.agents.single import (
    LATEST_EXCHANGE, MARGIN_S, _Abandoned, _abstained, _bounded, _finish, _recent, _Run,
)  # fmt: skip
from contextlens.agents.specialist import Outcome, run_specialist, system_prompt
from contextlens.agents.tools import Tools
from contextlens.llm.client import LLMClient
from contextlens.prompts.registry import get_prompt
from contextlens.replay.runner import Context
from contextlens.schemas import ButtonRequest, Card, Finding

_ASSIST_ROLE = (
    "You are the assistance agent. No question was asked, so find what earlier in the conversation is relevant to the "
    "latest exchange, and who said it. Name a speaker only as the transcript shows them."
)


def merge(findings: list[Finding]) -> list[Finding]:
    """Findings that cite the same turns, or say the same thing, once. The one with a label (then more evidence) stays."""
    kept: list[Finding] = []
    for f in sorted(findings, key=lambda f: (f.label is None, -len(f.evidence))):
        ids = {e.turn_id for e in f.evidence}
        words = " ".join(f.statement.lower().split())
        if not any(ids == {e.turn_id for e in k.evidence} or words == " ".join(k.statement.lower().split()) for k in kept):
            kept.append(f)
    order = {id(f): i for i, f in enumerate(findings)}
    return sorted(kept, key=lambda f: order[id(f)])


class RoutedTeam:
    def __init__(self, llm: LLMClient, tools: Tools, prompt_version: str, deadline_s: float = 8.0) -> None:
        self.llm, self.tools, self.deadline_s = llm, tools, deadline_s
        recall = get_prompt("recall", prompt_version).text
        self._specs = {
            "memory": (memory_agent.KIND, memory_agent.system(prompt_version), memory_agent.TOOLS),
            "assistance": ("recall", system_prompt(_ASSIST_ROLE, recall, "assistance", "recall", memory_agent.TOOLS),
                           memory_agent.TOOLS),
            "consistency": (consistency_agent.KIND, consistency_agent.system(prompt_version, False),
                            consistency_agent.tool_names(False)),
        }  # fmt: skip

    def answer(self, req: ButtonRequest, ctx: Context) -> Card:
        start = time.perf_counter()
        end = start + self.deadline_s
        try:
            turns = _bounded(functools.partial(_recent, ctx, req, LATEST_EXCHANGE), self.deadline_s + MARGIN_S)
        except _Abandoned:
            return _abstained("the store did not answer in time")
        except Exception as exc:
            return _abstained(f"store failed: {type(exc).__name__}")
        if not turns:
            return _abstained("the conversation has no turns yet")
        prep = prepare(self.tools, req, turns, end)
        asked = bool((req.question or "").strip())
        names = ["memory" if asked else "assistance", *(["consistency"] if prep.flagged else [])]
        runs = {n: _Run() for n in names}
        found: list[Finding] = []
        notes, degraded, primary_empty = list(prep.notes), False, False
        for n in names:
            t0 = time.perf_counter()
            kind, system, tool_names = self._specs[n]
            try:
                out = run_specialist(
                    name=n, kind=kind, system=system, tool_names=tool_names, llm=self.llm, tools=self.tools, ctx=ctx,
                    req=req, turns=turns, end=end, run=runs[n], context=prep.context,
                )  # fmt: skip
            except Exception as exc:
                out = Outcome(notes=[f"{n}: failed: {type(exc).__name__}"], degraded=True)
            if n == "consistency":  # the label is the checker's, set in code
                out.findings = [f.model_copy(update={"label": prep.label}) for f in out.findings]
            elif not out.findings and not out.notes and not out.degraded:
                primary_empty = True
            found += out.findings
            notes += out.notes
            degraded = degraded or out.degraded
            ctx.stage_ms[f"{n}_ms"] = int((time.perf_counter() - t0) * 1000)
            ctx.stage_ms[f"{n}_tokens"] = sum(r.prompt_tokens + r.completion_tokens for r in runs[n].results)
            ctx.stage_ms[f"ran_{n}"] = 1
        ctx.stage_ms.update(prep.stage())
        findings = merge(found)
        total = _Run()
        total.results = [r for n in names for r in runs[n].results]
        total.abandoned = sum(r.abandoned for r in runs.values())
        if asked and primary_empty and not prep.flagged:  # memory looked and found nothing: abstain (entry 48)
            card = _abstained("memory found nothing relevant")
        elif findings:
            card = Card(mode="recall", status="partial" if degraded else "answered", findings=findings)
        else:
            card = _abstained("no verified findings")
        card = card.model_copy(update={"missing": [*notes, *card.missing]})
        try:
            return _finish(card, ctx, req, start, total, self.llm, end, MARGIN_S)
        except Exception as exc:
            return _abstained(f"assistance: failed: {type(exc).__name__}")
