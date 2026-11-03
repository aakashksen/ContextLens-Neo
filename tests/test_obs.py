import time

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from prometheus_client import CollectorRegistry, generate_latest

from contextlens.agents.graph import TeamAssistant
from contextlens.agents.tools import Tools
from contextlens.agents.single import SingleAgent
from contextlens.llm.client import FakeLLM, LLMResult
from contextlens.memory.extract import MemoryStore
from contextlens.memory.store import InMemoryTranscriptStore
from contextlens.obs.tracing import Telemetry, TracedAssistant
from contextlens.replay.runner import Context
from contextlens.retrieval.hybrid import Retriever
from contextlens.schemas import ButtonRequest, Card, Turn

MARK = "ZQXMARK"


def turn(seq, text, speaker):
    return Turn(id=f"t{seq}", conversation_id=f"c{MARK}", seq=seq, speaker=speaker, text=text,
                start_ms=seq * 1000, end_ms=seq * 1000 + 900, source="synthetic")  # fmt: skip


class NoEmbed:
    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def finding(agent, turn_id, quote, statement):
    return {"agent": agent, "kind": "recall", "statement": statement, "confidence": 0.9,
            "evidence": [{"turn_id": turn_id, "quote": quote, "speaker": f"sp{MARK}", "start_ms": 0}]}  # fmt: skip


def done(*f):
    return {"tool": "final", "findings": list(f)}


class Boom:
    def answer(self, req, ctx):
        raise RuntimeError(f"secret {MARK}")


class WithText:
    """Returns a card whose suggestion and missing notes hold the marker."""

    def answer(self, req, ctx):
        return Card(mode="recall", status="partial", suggestion=f"say {MARK}", missing=[f"gap {MARK}"],
                    tokens=7, cost_usd=0.001, findings=[])  # fmt: skip


class TokLLM(FakeLLM):
    def _raw(self, messages, schema, max_tokens, temperature, timeout_s):
        r = super()._raw(messages, schema, max_tokens, temperature, timeout_s)
        return LLMResult(text=r.text, model=r.model, prompt_tokens=10, completion_tokens=5)


def build(provider=None, registry=None, fail=False, inner_override=None):
    turns = [
        turn(1, f"The deadline is Monday {MARK}", f"Ravi{MARK}"),
        turn(2, f"We moved the deadline to Wednesday {MARK}", f"Sue{MARK}"),
        turn(3, f"What was the deadline? {MARK}", f"Ana{MARK}"),
    ]
    store = InMemoryTranscriptStore()
    for t in turns:
        store.add(t)
    cid = f"c{MARK}"
    tools = Tools(store, Retriever(store, NoEmbed()), MemoryStore(), 3, cid)
    mem = done(finding("memory", "t1", f"The deadline is Monday {MARK}", f"Monday {MARK}"))
    con = done(finding("consistency", "t2", f"moved the deadline to Wednesday {MARK}", f"moved {MARK}"))
    plan = {"mode": "recall", "followup": None, "suggestion": f"say {MARK}"}
    llms = {"memory": TokLLM([mem]), "consistency": TokLLM([con]), "assistance": TokLLM([plan])}
    inner = inner_override or (Boom() if fail else TeamAssistant(llms, tools, "v1"))
    tel = Telemetry(tracer_provider=provider, registry=registry)
    ctx = Context(store=store, as_of_seq=3)
    req = ButtonRequest(conversation_id=cid, as_of_seq=3, question=f"deadline? {MARK}")
    return TracedAssistant(inner, tel), req, ctx


def provider_with_memory():
    exp = InMemorySpanExporter()
    prov = TracerProvider()
    prov.add_span_processor(SimpleSpanProcessor(exp))
    return prov, exp


def test_press_creates_span_per_agent():
    prov, exp = provider_with_memory()
    asst, req, ctx = build(prov, CollectorRegistry())
    card = asst.answer(req, ctx)
    asst.flush()
    assert card.status == "answered"
    spans = exp.get_finished_spans()
    assert {"press", "agent.memory", "agent.consistency", "agent.assistance"} <= {s.name for s in spans}
    root = next(s for s in spans if s.name == "press")
    for s in spans:
        if s.name.startswith("agent."):
            assert s.parent.span_id == root.context.span_id
    assert root.attributes["card.status"] == "answered"


def test_no_transcript_text_in_spans_or_metrics():
    for fail, fixed in ((False, None), (True, None), (False, WithText())):
        prov, exp = provider_with_memory()
        reg = CollectorRegistry()
        asst, req, ctx = build(prov, reg, fail=fail, inner_override=fixed)
        if fail:
            with pytest.raises(RuntimeError):
                asst.answer(req, ctx)
        else:
            asst.answer(req, ctx)
        asst.flush()
        spans = exp.get_finished_spans()
        assert spans  # the scan has something to look at
        blobs = []
        for s in spans:
            blobs += [s.name, str(dict(s.attributes)), str(s.status.description or "")]
            blobs += [f"{e.name} {dict(e.attributes or {})}" for e in s.events]
        blobs.append(generate_latest(reg).decode())
        assert MARK not in "\n".join(blobs)


def test_error_is_counted_by_type_only():
    reg = CollectorRegistry()
    asst, req, ctx = build(None, reg, fail=True)
    with pytest.raises(RuntimeError):
        asst.answer(req, ctx)
    asst.flush()
    assert 'contextlens_press_errors_total{error="RuntimeError"} 1.0' in generate_latest(reg).decode()


def test_metrics_cover_latency_tokens_cost():
    reg = CollectorRegistry()
    asst, req, ctx = build(None, reg)
    asst.answer(req, ctx)
    asst.flush()
    out = generate_latest(reg).decode()
    for name in ("contextlens_press_seconds_count", "contextlens_agent_seconds_count",
                 "contextlens_tokens_total", "contextlens_cost_usd_total"):  # fmt: skip
        assert name in out
    assert 'agent="memory"' in out


def test_failing_exporter_does_not_affect_card():
    class Bad(SpanExporter):
        def export(self, spans):
            raise OSError("down")

    prov = TracerProvider()
    prov.add_span_processor(SimpleSpanProcessor(Bad()))
    asst, req, ctx = build(prov, CollectorRegistry())
    assert asst.answer(req, ctx).status == "answered"
    asst.flush()


def test_broken_telemetry_does_not_affect_card():
    asst, req, ctx = build(None, CollectorRegistry())

    def broken(*a, **k):
        raise ZeroDivisionError

    asst.telemetry.record = broken
    assert asst.answer(req, ctx).status == "answered"
    asst.flush()


def counter_value(reg, name, labels=None):
    return reg.get_sample_value(name, labels or {}) or 0.0


def test_team_press_spans_and_tokens_are_true_and_not_double_counted():
    prov, exp = provider_with_memory()
    reg = CollectorRegistry()
    asst, req, ctx = build(prov, reg)
    card = asst.answer(req, ctx)
    asst.flush()
    assert card.tokens == 45
    per = {a: counter_value(reg, "contextlens_tokens_total", {"agent": a}) for a in
           ("memory", "consistency", "assistance")}  # fmt: skip
    assert per == {"memory": 15, "consistency": 15, "assistance": 15}
    assert sum(per.values()) == card.tokens
    assert counter_value(reg, "contextlens_tokens_total", {"agent": "other"}) == 0


def single_press(registry, provider=None):
    turns = [turn(1, f"The deadline is Monday {MARK}", f"Ravi{MARK}"),
             turn(2, f"What was the deadline? {MARK}", f"Ana{MARK}")]  # fmt: skip
    store = InMemoryTranscriptStore()
    for t in turns:
        store.add(t)
    tools = Tools(store, Retriever(store, NoEmbed()), MemoryStore(), 2, f"c{MARK}")
    f = {"agent": "single", "kind": "recall", "statement": f"s {MARK}", "confidence": 0.9,
         "evidence": [{"turn_id": "t1", "quote": f"The deadline is Monday {MARK}", "speaker": f"Ravi{MARK}",
                       "start_ms": 0}]}  # fmt: skip
    final = {"tool": "final", "card": {"mode": "recall", "status": "answered", "findings": [f],
                                      "suggestion": None, "missing": []}}  # fmt: skip
    agent = SingleAgent(TokLLM([final]), tools, "v1")
    asst = TracedAssistant(agent, Telemetry(tracer_provider=provider, registry=registry))
    req = ButtonRequest(conversation_id=f"c{MARK}", as_of_seq=2, question=f"deadline? {MARK}")
    return asst, req, Context(store=store, as_of_seq=2)


def test_single_agent_press_has_stage_spans_and_token_metric():
    prov, exp = provider_with_memory()
    reg = CollectorRegistry()
    asst, req, ctx = single_press(reg, prov)
    card = asst.answer(req, ctx)
    asst.flush()
    assert card.status == "answered" and card.tokens == 15
    spans = exp.get_finished_spans()
    names = {s.name for s in spans}
    assert {"press", "agent.single", "stage.llm", "stage.tools"} <= names
    root = next(s for s in spans if s.name == "press")
    agent = next(s for s in spans if s.name == "agent.single")
    assert agent.parent.span_id == root.context.span_id
    for s in spans:
        if s.name.startswith("stage."):
            assert s.parent.span_id == agent.context.span_id
    assert counter_value(reg, "contextlens_tokens_total", {"agent": "single"}) == card.tokens
    out = generate_latest(reg).decode()
    assert 'contextlens_agent_seconds_count{agent="single"} 1.0' in out
    assert MARK not in out


def test_labels_stay_bounded_for_unknown_stage_keys():
    reg = CollectorRegistry()
    tel = Telemetry(registry=reg)
    stages = {f"weird{i}_ms": 5 for i in range(50)} | {f"weird{i}_tokens": 1 for i in range(50)}
    tel.record(0, 10_000_000, stages, Card(mode="recall", status="partial", findings=[], tokens=50), None)
    out = generate_latest(reg).decode()
    agents = {line.split('agent="')[1].split('"')[0] for line in out.splitlines()
              if 'agent="' in line and not line.startswith("#")}  # fmt: skip
    assert agents <= {"memory", "consistency", "assistance", "single", "other"}
    assert counter_value(reg, "contextlens_tokens_total", {"agent": "other"}) == 50


def test_agent_spans_say_their_start_is_approximate():
    prov, exp = provider_with_memory()
    asst, req, ctx = build(prov, CollectorRegistry())
    asst.answer(req, ctx)
    asst.flush()
    agents = [s for s in exp.get_finished_spans() if s.name.startswith("agent.")]
    assert agents and all(s.attributes["start_approximate"] is True for s in agents)
    root = next(s for s in exp.get_finished_spans() if s.name == "press")
    assert all(s.start_time == root.start_time for s in agents)


def test_slow_exporter_does_not_delay_the_card():
    class Slow(SpanExporter):
        def export(self, spans):
            time.sleep(0.5)

    prov = TracerProvider()
    prov.add_span_processor(SimpleSpanProcessor(Slow()))
    asst, req, ctx = build(prov, CollectorRegistry())
    t0 = time.perf_counter()
    card = asst.answer(req, ctx)
    assert card.status == "answered"
    assert time.perf_counter() - t0 < 0.5  # 4 spans at 0.5s would be 2s if run inline
    asst.flush(timeout_s=10)
