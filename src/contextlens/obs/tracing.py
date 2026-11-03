"""Local traces and metrics for a button press. Nothing here reads transcript text.

Span times are approximate. The assistant reports only how long each agent took, not when it started, so
every agent span is given the press start as its start and ends after its own duration (agents run in
parallel, so this is close for them, but a later stage is drawn earlier than it ran). Agent spans carry
the attribute start_approximate=true; the durations are the true ones.

Telemetry runs on a background thread after the card exists, so a slow exporter cannot delay the card.
Only counts, durations, and a few bounded labels (agent, status, mode, exception type name) are recorded.
No exporter is set up: spans go wherever the caller's TracerProvider sends them, and metrics sit in a
Prometheus registry that stays in this process until something chooses to serve it."""

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from prometheus_client import CollectorRegistry, Counter, Histogram, start_http_server

from contextlens.replay.runner import Assistant, Context
from contextlens.schemas import ButtonRequest, Card

log = logging.getLogger(__name__)

AGENTS = ("memory", "consistency", "assistance", "single")  # the only agent label values; else "other"
SINGLE_STAGES = ("llm", "tools", "suggestion")  # SingleAgent's stage_ms keys; they are stages of agent "single"
MAX_PENDING = 256  # telemetry jobs waiting for the background thread; beyond this they are dropped
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
        self.dropped = Counter(
            "contextlens_dropped_presses", "Presses whose telemetry was dropped (backlog full)", registry=r
        )

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
        single: dict[str, int] = {}
        for key, value in stages.items():
            if not isinstance(value, int | float):
                continue
            if key in SINGLE_STAGES:
                single[key] = int(value)
                ms["single"] = ms.get("single", 0) + int(value)
            elif key.endswith("_ms"):
                a = _agent(key[:-3])
                ms[a] = ms.get(a, 0) + int(value)
            elif key.endswith("_tokens"):
                a = _agent(key[:-7])
                tokens[a] = tokens.get(a, 0) + int(value)
        if card is not None and card.tokens > sum(tokens.values()):
            # an assistant that reports no per-agent tokens (SingleAgent): the rest of the card's total
            rest = "single" if single else "other"
            tokens[rest] = tokens.get(rest, 0) + card.tokens - sum(tokens.values())
        for a, v in ms.items():
            span = self.tracer.start_span(f"agent.{a}", context=ctx, start_time=start_ns)
            span.set_attribute("agent", a)
            span.set_attribute("duration_ms", v)
            span.set_attribute("tokens", tokens.get(a, 0))
            span.set_attribute("start_approximate", True)
            if a == "single":
                actx = trace.set_span_in_context(span)
                at = start_ns
                for name, d in single.items():
                    st = self.tracer.start_span(f"stage.{name}", context=actx, start_time=at)
                    st.set_attribute("stage", name)
                    st.set_attribute("duration_ms", d)
                    st.set_attribute("start_approximate", True)
                    at += d * 1_000_000
                    st.end(end_time=at)
            span.end(end_time=start_ns + v * 1_000_000)
            self._agent_s.labels(a).observe(v / 1000)
        for a, v in tokens.items():
            self._tokens.labels(a).inc(v)
        root.end(end_time=end_ns)


class TracedAssistant:
    """Wraps any Assistant. Telemetry runs on a background thread after the card exists, so it can never
    change the card, raise, or hold it up. Call flush() to wait for it (tests, shutdown)."""

    def __init__(self, inner: Assistant, telemetry: Telemetry) -> None:
        self.inner, self.telemetry = inner, telemetry
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="telemetry")
        self._pending: dict[Future, str] = {}  # job -> its conversation
        self._lock = threading.Lock()

    def _run(self, args: tuple) -> None:
        try:
            self.telemetry.record(*args)
        except Exception:
            log.warning("telemetry failed; the card is unaffected")

    def _submit(self, args: tuple, conversation_id: str) -> None:
        with self._lock:
            if len(self._pending) >= MAX_PENDING:
                log.warning("telemetry backlog full; dropping one press")
                self.telemetry.dropped.inc()
                return
            fut = self._pool.submit(self._run, args)
            self._pending[fut] = conversation_id
        fut.add_done_callback(self._forget)

    def _forget(self, fut: Future) -> None:
        with self._lock:
            self._pending.pop(fut, None)

    def discard(self, conversation_id: str) -> None:
        """Drop this conversation's queued jobs, which hold its cards in memory. One already running finishes."""
        with self._lock:
            mine = [f for f, c in self._pending.items() if c == conversation_id]
        for fut in mine:
            fut.cancel()  # only succeeds for a job that has not started; its done callback forgets it

    def flush(self, timeout_s: float = 5.0) -> None:
        with self._lock:
            futs = list(self._pending)
        wait(futs, timeout=timeout_s)

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
                self._submit((t0, t1, stages, card, error), req.conversation_id)
            except Exception:
                log.warning("telemetry failed; the card is unaffected")


def serve_metrics(telemetry: Telemetry, port: int, host: str = "127.0.0.1") -> None:
    """Opt-in: serve /metrics. The default host keeps it on this machine; Prometheus in Docker needs a wider
    host that the person chooses on purpose."""
    start_http_server(port, addr=host, registry=telemetry.registry)
