"""
OpenTelemetry instrumentation — Dash0-native, vendor-neutral.

Key design decisions:

1. Step-level spans, not request-level
   A slow RAG response could be slow retrieval OR slow generation.
   Without separate spans you can't tell which to fix.
   With Dash0, you'd see this in their trace waterfall immediately.

2. Consistent attribute naming
   We use a shared helper (record_rag_metrics, record_eval_metrics)
   so attribute names are identical across all spans.
   Inconsistent naming (latency vs latency_ms vs response_time)
   is the silent killer of observability — you can't aggregate what
   you can't find.

3. Dash0 auth via OTLP headers
   Dash0 accepts standard OTLP with an Authorization header.
   No proprietary SDK required — this is the OpenTelemetry promise.
"""

from contextlib import contextmanager
from typing import Any, Generator

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Span, Status, StatusCode

from src.settings import settings

_tracer: trace.Tracer | None = None
_initialised: bool = False


def setup_telemetry() -> None:
    """
    Initialise OpenTelemetry.

    When OTEL_ENABLED=true: exports to Dash0 (or any OTLP collector)
    via gRPC with Bearer token auth.

    When OTEL_ENABLED=false: prints spans to console.
    Useful during development to verify instrumentation without
    needing a running collector.
    """
    global _tracer, _initialised

    if _initialised:
        return

    resource = Resource.create({
        "service.name": settings.otel_service_name,
        "service.version": "0.2.0",
        "deployment.environment": "development",
        # Semantic conventions for AI systems (emerging OTel standard)
        "ai.system": "rag",
        "ai.vector_store": "mongodb-atlas",
        "ai.llm.provider": "openrouter",
        "ai.embedding.provider": "voyage-ai",
    })

    provider = TracerProvider(resource=resource)

    if settings.otel_enabled:
        # Dash0 accepts standard OTLP with Authorization header
        headers: dict[str, str] = {}
        if settings.dash0_auth_token:
            headers["Authorization"] = f"Bearer {settings.dash0_auth_token}"

        exporter = OTLPSpanExporter(
            endpoint=f"{settings.otel_endpoint}/v1/traces",
            headers=headers,
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        print(f"[telemetry] Exporting to {settings.otel_endpoint}")
    else:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        print("[telemetry] Console mode — set OTEL_ENABLED=true for Dash0 export")

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(settings.otel_service_name)
    _initialised = True


def get_tracer() -> trace.Tracer:
    if _tracer is None:
        setup_telemetry()
    assert _tracer is not None
    return _tracer


@contextmanager
def traced_span(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Generator[Span, None, None]:
    """
    Context manager for a traced operation.

    Always sets ERROR status on exceptions so failed RAG calls
    are visible in Dash0 without manual error handling at every site.

    Usage:
        with traced_span("rag.retrieval", {"query": q, "top_k": 5}) as span:
            docs = await retrieve(q)
            span.set_attribute("results.count", len(docs))
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                if isinstance(value, (str, bool, int, float)):
                    span.set_attribute(key, value)
                else:
                    span.set_attribute(key, str(value))
        try:
            yield span
        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
            raise


def record_rag_metrics(
    span: Span,
    *,
    total_latency_ms: float,
    retrieval_latency_ms: float,
    generation_latency_ms: float,
    context_count: int,
    answer_length: int,
    corpus_sources: list[str],
) -> None:
    """Attach standard RAG metrics to a span with consistent attribute names."""
    span.set_attribute("rag.latency_ms.total", total_latency_ms)
    span.set_attribute("rag.latency_ms.retrieval", retrieval_latency_ms)
    span.set_attribute("rag.latency_ms.generation", generation_latency_ms)
    span.set_attribute("rag.context.count", context_count)
    span.set_attribute("rag.answer.length", answer_length)
    span.set_attribute("rag.corpus.sources", ",".join(corpus_sources))


def record_router_metrics(
    span: Span,
    *,
    action: str,
    source_filter: str,
    was_reformulated: bool,
    was_decomposed: bool,
    sub_query_count: int,
) -> None:
    """Attach query router decision metrics to a span."""
    span.set_attribute("router.action", action)
    span.set_attribute("router.source_filter", source_filter)
    span.set_attribute("router.reformulated", was_reformulated)
    span.set_attribute("router.decomposed", was_decomposed)
    span.set_attribute("router.sub_query_count", sub_query_count)


def record_eval_metrics(
    span: Span,
    *,
    faithfulness: float,
    relevancy: float,
    precision: float,
    status: str,
    overall_score: float,
) -> None:
    """Attach RAGAS evaluation scores to a span."""
    span.set_attribute("eval.faithfulness", faithfulness)
    span.set_attribute("eval.answer_relevancy", relevancy)
    span.set_attribute("eval.context_precision", precision)
    span.set_attribute("eval.overall_score", overall_score)
    span.set_attribute("eval.status", status)
    span.set_attribute("eval.passed", status == "pass")
