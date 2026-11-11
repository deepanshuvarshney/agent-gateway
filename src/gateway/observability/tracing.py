"""OpenTelemetry setup. Default exporter is none so CI stays quiet."""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

from gateway.config import Settings


def setup_tracing(settings: Settings) -> None:
    if settings.otel_exporter == "none":
        return
    provider = TracerProvider()
    if settings.otel_exporter == "console":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    elif settings.otel_exporter == "otlp":
        # Imported lazily so a console-only install path does not need the endpoint.
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter(endpoint=settings.otel_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    else:
        raise ValueError(
            f"Unknown GATEWAY_OTEL_EXPORTER '{settings.otel_exporter}'. Use none, console, or otlp."
        )
    trace.set_tracer_provider(provider)


def get_tracer() -> trace.Tracer:
    return trace.get_tracer("gateway")
