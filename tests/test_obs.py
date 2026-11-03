import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from prometheus_client import CollectorRegistry, generate_latest

from contextlens.agents.graph import TeamAssistant
from contextlens.agents.tools import Tools
from contextlens.llm.client import FakeLLM
from contextlens.memory.extract import MemoryStore
from contextlens.memory.store import InMemoryTranscriptStore
from contextlens.obs.tracing import Telemetry, TracedAssistant
from contextlens.replay.runner import Context
from contextlens.retrieval.hybrid import Retriever
from contextlens.schemas import ButtonRequest, Turn

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


def build(provider=None, registry=None, fail=False):
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
    llms = {"memory": FakeLLM([mem]), "consistency": FakeLLM([con]), "assistance": FakeLLM([plan])}
    inner = Boom() if fail else TeamAssistant(llms, tools, "v1")
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
    assert card.status == "answered"
    spans = exp.get_finished_spans()
    assert {"press", "agent.memory", "agent.consistency", "agent.assistance"} <= {s.name for s in spans}
    root = next(s for s in spans if s.name == "press")
    for s in spans:
        if s.name.startswith("agent."):
            assert s.parent.span_id == root.context.span_id
    assert root.attributes["card.status"] == "answered"


def test_no_transcript_text_in_spans_or_metrics():
    for fail in (False, True):
        prov, exp = provider_with_memory()
        reg = CollectorRegistry()
        asst, req, ctx = build(prov, reg, fail=fail)
        if fail:
            with pytest.raises(RuntimeError):
                asst.answer(req, ctx)
        else:
            asst.answer(req, ctx)
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
    assert 'contextlens_press_errors_total{error="RuntimeError"} 1.0' in generate_latest(reg).decode()


def test_metrics_cover_latency_tokens_cost():
    reg = CollectorRegistry()
    asst, req, ctx = build(None, reg)
    asst.answer(req, ctx)
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


def test_broken_telemetry_does_not_affect_card():
    asst, req, ctx = build(None, CollectorRegistry())

    def broken(*a, **k):
        raise ZeroDivisionError

    asst.telemetry.record = broken
    assert asst.answer(req, ctx).status == "answered"
