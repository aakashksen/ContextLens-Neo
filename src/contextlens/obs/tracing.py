"""Local traces and metrics for a button press. Nothing here reads transcript text.

Only counts, durations, and a few bounded labels (agent, status, mode, exception type name) are recorded.
No exporter is set up: spans go wherever the caller's TracerProvider sends them, and metrics sit in a
Prometheus registry that stays in this process until something chooses to serve it."""

import logging
import time

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from prometheus_client import CollectorRegistry, Counter, Histogram, start_http_server

from contextlens.replay.runner import Assistant, Context
from contextlens.schemas import ButtonRequest, Card

log = logging.getLogger(__name__)

AGENTS = ("memory", "consistency", "assistance")  # the only agent label values; anything else is "other"
_BUCKETS = (0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10, 15)


def _agent(name: str) -> str:
    return name if name in AGENTS else "other"


class Telemetry:
    """Spans and metrics. Pass a registry/provider to keep them apart (tests); the defaults are private."""

    def __init__(
        self, tracer_provider: TracerProvider | None = None, registry: CollectorRegistry | None = None
    ) -> None:
        self.registry = registry if registry is not None else CollectorRegistry()
        self.tracer = (tracer_provider or TracerProvider()).get_tracer("contextlens")
        r = self.registry
        self._press_s = Histogram("contextlens_press_seconds", "Press latency", ["status"], buckets=_BUCKETS, registry=r)
        self._agent_s = Histogram("contextlens_agent_seconds", "Agent stage time", ["agent"], buckets=_BUCKETS, registry=r)
        self._tokens = Counter("contextlens_tokens", "Model tokens", ["agent"], registry=r)
        self._cost = Counter("contextlens_cost_usd", "Model cost in USD", registry=r)
        self._presses = Counter("contextlens_presses", "Presses", ["status", "mode"], registry=r)
        self._errors = Counter("contextlens_press_errors", "Presses that raised", ["error"], registry=r)

    def record(
        self, start_ns: int, end_ns: int, stages: dict[str, int], card: Card | None, error: BaseException | None
    ) -> None:
        """`stages` is the assistant's stage_ms for this press. Never given text; `error` is used for its type only."""
        root = self.tracer.start_span("press", start_time=start_ns)
        if card is not None:
            root.set_attribute("card.status", card.status)
            root.set_attribute("card.mode", card.mode)
            root.set_attribute("card.findings", len(card.findings))
            root.set_attribute("card.tokens", card.tokens)
            root.set_attribute("card.cost_usd", card.cost_usd)
            self._presses.labels(card.status, card.mode).inc()
            self._press_s.labels(card.status).observe((end_ns - start_ns) / 1e9)
            self._cost.inc(max(card.cost_usd, 0.0))
        if error is not None:
            kind = type(error).__name__[:40]
            root.set_attribute("error.type", kind)
            root.set_status(trace.StatusCode.ERROR)  # no description: a message could hold transcript text
            self._errors.labels(kind).inc()
        ctx = trace.set_span_in_context(root)
        tokens: dict[str, int] = {}
        ms: dict[str, int] = {}
        for key, value in stages.items():
            if not isinstance(value, int | float):
                continue
            if key.endswith("_ms"):
                a = _agent(key[:-3])
                ms[a] = ms.get(a, 0) + int(value)
            elif key.endswith("_tokens"):
                a = _agent(key[:-7])
                tokens[a] = tokens.get(a, 0) + int(value)
        for a, v in ms.items():
            span = self.tracer.start_span(f"agent.{a}", context=ctx, start_time=start_ns)
            span.set_attribute("agent", a)
            span.set_attribute("duration_ms", v)
            span.set_attribute("tokens", tokens.get(a, 0))
            span.end(end_time=start_ns + v * 1_000_000)
            self._agent_s.labels(a).observe(v / 1000)
        for a, v in tokens.items():
            self._tokens.labels(a).inc(v)
        root.end(end_time=end_ns)


class TracedAssistant:
    """Wraps any Assistant. Telemetry runs after the card exists and can never change it or raise."""

    def __init__(self, inner: Assistant, telemetry: Telemetry) -> None:
        self.inner, self.telemetry = inner, telemetry

    def answer(self, req: ButtonRequest, ctx: Context) -> Card:
        before = dict(ctx.stage_ms)
        t0 = time.time_ns()
        card: Card | None = None
        error: BaseException | None = None
        try:
            card = self.inner.answer(req, ctx)
            return card
        except BaseException as exc:
            error = exc
            raise
        finally:
            t1 = time.time_ns()
            stages = {k: v - before.get(k, 0) for k, v in ctx.stage_ms.items()}
            try:
                self.telemetry.record(t0, t1, stages, card, error)
            except Exception:
                log.warning("telemetry failed; the card is unaffected")


def serve_metrics(telemetry: Telemetry, port: int, host: str = "127.0.0.1") -> None:
    """Opt-in: serve /metrics. The default host keeps it on this machine; Prometheus in Docker needs a wider
    host that the person chooses on purpose."""
    start_http_server(port, addr=host, registry=telemetry.registry)
