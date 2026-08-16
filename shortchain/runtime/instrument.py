"""Own ``TracerProvider`` + OpenLLMetry instrumentor enablement (PR 5).

Mirrors the *useful* parts of ``traceloop/sdk/tracing/tracing.py`` without
the Traceloop client: own Resource, own exporter (OTLP HTTP, optional
second display exporter, or ``file://`` JSONL dump), always
``ThreadingInstrumentor``, and per-instrumentor construction so a missing
framework / constructor change cannot abort ``ShortChain.init``.

Why not ``Traceloop.init`` (K6): it defaults to ``https://api.traceloop.com``,
prints Traceloop branding, optionally starts a Fetcher, and hard-depends on
every instrumentation package. We keep the public surface ours.
"""

from __future__ import annotations

import inspect
import os
import threading
from typing import Any, Callable

from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExportResult,
    SpanExporter,
)
from opentelemetry.trace import ProxyTracerProvider, get_tracer_provider, set_tracer_provider

from shortchain.utils.logging import get_logger

log = get_logger(__name__)

# Instrumentor enablement table (semconv-ai 0.5 / OpenLLMetry 0.62 line).
# ``use_attributes`` is a CONSTRUCTOR kwarg on LangChain/OpenAI/Anthropic,
# not an ``instrument()`` arg; LiteLLM 0.62 uses ``use_legacy_attributes``.
# Each entry: (module, class name, constructor kwargs tried signature-aware).
INSTRUMENTOR_SPECS: list[tuple[str, str, dict[str, Any]]] = [
    ("opentelemetry.instrumentation.langchain", "LangchainInstrumentor", {"use_attributes": True}),
    ("opentelemetry.instrumentation.openai", "OpenAIInstrumentor", {"use_attributes": True}),
    (
        "opentelemetry.instrumentation.openai_agents",
        "OpenAIAgentsInstrumentor",
        {"replace_existing_processors": True},
    ),
    ("opentelemetry.instrumentation.crewai", "CrewAIInstrumentor", {}),
    ("opentelemetry.instrumentation.agno", "AgnoInstrumentor", {}),
    ("opentelemetry.instrumentation.mcp", "McpInstrumentor", {}),
    ("opentelemetry.instrumentation.anthropic", "AnthropicInstrumentor", {"use_attributes": True}),
    ("opentelemetry.instrumentation.litellm", "LiteLLMInstrumentor", {"use_legacy_attributes": True}),
]

AUTO_INSTRUMENTS = {spec[0].rsplit(".", 1)[-1] for spec in INSTRUMENTOR_SPECS}

# Never auto-enabled: noisy spans; users can opt in explicitly.
_BLOCKED_BY_DEFAULT = {"requests", "urllib3", "redis", "fastapi", "flask", "django"}


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------


class FileSpanExporter(SpanExporter):
    """``file://`` dev dump: one JSON line per ended span (OTelSpan shape).

    NOT the production path (K16) — no dual-export, no quality gating
    server-side. Assemble these lines offline with ``OtelTrajectoryLoader``.
    """

    def __init__(self, path: str, lock: threading.Lock | None = None) -> None:
        self._path = path
        self._lock = lock or threading.Lock()
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)

    def export(self, spans: list[ReadableSpan]) -> SpanExportResult:
        import json

        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                for span in spans:
                    f.write(json.dumps(_span_to_dict(span)) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:  # pragma: no cover - trivial
        return None


def _span_to_dict(span: ReadableSpan) -> dict[str, Any]:
    """OtelSpan-shaped dict for file dumps (attributes as plain values)."""
    attributes = dict(span.attributes or {})
    return {
        "trace_id": span.context.trace_id.to_bytes(16, "big").hex(),
        "span_id": span.context.span_id.to_bytes(8, "big").hex(),
        "parent_span_id": (
            span.parent.span_id.to_bytes(8, "big").hex()
            if span.parent
            else None
        ),
        "name": span.name,
        "start_time_unix_nano": span.start_time,
        "end_time_unix_nano": span.end_time,
        "status_code": span.status.status_code.name if span.status else "UNSET",
        "attributes": attributes,
        "resource": dict(span.resource.attributes or {}),
        "events": [],
    }


# ---------------------------------------------------------------------------
# Processors
# ---------------------------------------------------------------------------


class SpanPostprocessProcessor(BatchSpanProcessor):
    """Runs an optional user callback on each ended span (redaction hook).

    Copies Traceloop's ``on_end`` unfreeze of ``_attributes._immutable`` so
    ``span_postprocess`` may still mutate attributes.
    """

    def __init__(
        self,
        span_exporter: SpanExporter,
        postprocess: Callable[[ReadableSpan], None] | None = None,
        max_queue_size: int = 2048,
        schedule_delay_millis: int = 5000,
        max_export_batch_size: int = 512,
        export_timeout_millis: int = 30_000,
    ) -> None:
        super().__init__(
            span_exporter,
            max_queue_size=max_queue_size,
            schedule_delay_millis=schedule_delay_millis,
            max_export_batch_size=max_export_batch_size,
            export_timeout_millis=export_timeout_millis,
        )
        self._postprocess = postprocess

    def on_end(self, span: ReadableSpan) -> None:
        if self._postprocess is not None:
            try:
                if hasattr(span, "_attributes"):
                    span._attributes._immutable = False  # type: ignore[attr-defined]
                self._postprocess(span)
            except Exception:  # never let redaction break export
                log.exception("span_postprocess callback failed")
        super().on_end(span)


# ---------------------------------------------------------------------------
# Provider setup
# ---------------------------------------------------------------------------


def setup_tracer_provider(
    *,
    app_name: str = "shortchain-app",
    resource_attributes: dict[str, Any] | None = None,
) -> TracerProvider:
    """Create (or reuse) the global TracerProvider with our Resource.

    Only takes ownership when the global provider is still the
    ``ProxyTracerProvider``; otherwise processors attach to the existing
    provider (same logic as Traceloop's ``init_tracer_provider``).
    """
    attributes = {SERVICE_NAME: app_name, **(resource_attributes or {})}
    resource = Resource.create(attributes)
    provider = get_tracer_provider()
    if isinstance(provider, ProxyTracerProvider):
        provider = TracerProvider(resource=resource)
        set_tracer_provider(provider)
    return provider


def attach_exporters(
    provider: TracerProvider,
    *,
    endpoint: str = "http://127.0.0.1:4318",
    api_key: str | None = None,
    display_endpoint: str | None = None,
    disable_batch: bool = False,
    span_postprocess: Callable[[ReadableSpan], None] | None = None,
    headers: dict[str, str] | None = None,
) -> list[SpanExporter]:
    """Attach OTLP HTTP (and optional display) processors to *provider*.

    Returns the exporters attached (tests inspect them directly).
    ``file://path`` endpoints become local JSONL span dumps instead of HTTP.
    """
    exporter = _build_exporter(endpoint, api_key, headers)
    provider.add_span_processor(
        _make_processor(exporter, disable_batch, span_postprocess)
    )
    exporters = [exporter]

    if display_endpoint:
        display = _build_exporter(display_endpoint, api_key, headers)
        provider.add_span_processor(_make_processor(display, disable_batch, None))
        exporters.append(display)
    return exporters


def _build_exporter(
    endpoint: str,
    api_key: str | None,
    headers: dict[str, str] | None = None,
) -> SpanExporter:
    if endpoint.startswith("file://"):
        return FileSpanExporter(endpoint[len("file://"):])
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )

    url = endpoint if endpoint.endswith("/v1/traces") else f"{endpoint.rstrip('/')}/v1/traces"
    merged_headers = dict(headers or {})
    if api_key:
        merged_headers.setdefault("Authorization", f"Bearer {api_key}")
    return OTLPSpanExporter(endpoint=url, headers=merged_headers or None)


def _make_processor(
    exporter: SpanExporter,
    disable_batch: bool,
    postprocess: Callable[[ReadableSpan], None] | None,
):
    if disable_batch:
        return SimpleSpanProcessor(exporter)
    if postprocess is not None:
        return SpanPostprocessProcessor(exporter, postprocess)
    return BatchSpanProcessor(exporter)


# ---------------------------------------------------------------------------
# Instrumentor enablement
# ---------------------------------------------------------------------------


def enable_instrumentors(
    provider: TracerProvider,
    *,
    instruments: set[str] | None = None,
    block_instruments: set[str] | None = None,
    content_tracing: bool = True,
) -> tuple[list[str], list[str]]:
    """Enable installed instrumentors (independently, never aborting init).

    Returns ``(enabled, skipped)`` short names. ``instruments=None`` means
    auto-detect every spec module that imports cleanly; ``block_instruments``
    always wins. Mirrors ``TRACELOOP_TRACE_CONTENT`` for the instrumentors'
    own reading.
    """
    # Instrumentors read TRACELOOP_TRACE_CONTENT, not SHORTCHAIN_*.
    os.environ["TRACELOOP_TRACE_CONTENT"] = (
        "true" if content_tracing else "false"
    )

    block = set(block_instruments or ())
    if instruments is None:
        wanted: set[str] | None = None  # auto-detect
    else:
        wanted = set(instruments) - block

    enabled: list[str] = []
    skipped: list[str] = []
    for module_name, class_name, kwargs in INSTRUMENTOR_SPECS:
        short = module_name.rsplit(".", 1)[-1]
        if short in block or short in _BLOCKED_BY_DEFAULT and instruments is None:
            skipped.append(short)
            continue
        if wanted is not None and short not in wanted:
            continue
        try:
            module = __import__(module_name, fromlist=[class_name])
            instrumentor_class = getattr(module, class_name)
            sig = inspect.signature(instrumentor_class.__init__)
            ctor_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            instrumentor = instrumentor_class(**ctor_kwargs)
            instrumentor.instrument(tracer_provider=provider)
            enabled.append(short)
        except Exception as exc:  # ImportError / TypeError / bad constructor
            skipped.append(f"{short} ({type(exc).__name__})")
            log.debug(f"instrumentor {short} not enabled: {exc}")

    # Association + task root survive LangChain thread-pool hops.
    try:
        threading_mod = __import__(
            "opentelemetry.instrumentation.threading", fromlist=["ThreadingInstrumentor"]
        )
        if "threading" not in block:
            threading_mod.ThreadingInstrumentor().instrument()
            enabled.append("threading")
    except Exception as exc:  # pragma: no cover
        skipped.append(f"threading ({type(exc).__name__})")
    return enabled, skipped


def set_content_tracing_env(content_tracing: bool) -> None:
    """Public helper so the SDK can set the mirror before init."""
    os.environ["TRACELOOP_TRACE_CONTENT"] = "true" if content_tracing else "false"