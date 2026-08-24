"""OpenTelemetry setup and privacy-safe request trace annotations."""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping

from opentelemetry import propagate, trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from src.config import Settings

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_configured = False
_tracer = trace.get_tracer("raguard.api")


def configure_tracing(settings: Settings) -> None:
    """Configure OTLP export only when an endpoint is explicitly supplied."""
    global _configured, _tracer
    endpoint = settings.otel_exporter_otlp_endpoint.strip()
    if not endpoint:
        logger.info("OpenTelemetry export disabled; OTEL_EXPORTER_OTLP_ENDPOINT is unset")
        return

    with _lock:
        if _configured:
            return
        provider = TracerProvider(
            resource=Resource.create({"service.name": settings.otel_service_name})
        )
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("raguard.api")
        _configured = True
        logger.info("OpenTelemetry OTLP tracing configured [endpoint=%s]", endpoint)


def shutdown_tracing() -> None:
    """Flush exported spans without failing application shutdown."""
    provider = trace.get_tracer_provider()
    shutdown = getattr(provider, "shutdown", None)
    if callable(shutdown):
        shutdown()


def start_request_span(method: str, path: str, headers: Mapping[str, str]):
    """Start a server span using W3C trace context without recording query text."""
    context = propagate.extract(headers)
    return _tracer.start_as_current_span(
        f"{method} {path}",
        context=context,
        attributes={"http.request.method": method, "url.path": path},
    )


def inject_trace_context() -> dict[str, str]:
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    return carrier


def annotate_query_span(
    *, request_id: str, outcome: str | None = None, failure_reason: str | None = None
) -> None:
    """Attach bounded query lifecycle fields to the current span."""
    span = trace.get_current_span()
    if not span.is_recording():
        return
    span.set_attribute("raguard.request_id", request_id)
    if outcome is not None:
        span.set_attribute("raguard.outcome", outcome)
    if failure_reason is not None:
        span.set_attribute("raguard.failure_reason", failure_reason)
