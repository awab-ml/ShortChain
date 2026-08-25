"""Shared pytest fixtures for the SDK / task-root test files.

The OTel global ``TracerProvider`` can only be set ONCE per process (OTEL
refuses overrides), so every test module that traces through the global
provider must share one env: an in-memory exporter + the association
injection processor.

The provider is therefore created exactly once, at import time, when the
``otel`` extra is installed. Creating it eagerly keeps the global stable no
matter which test module runs first — some SDK tests call
``setup_tracer_provider`` directly, which only replaces a
``ProxyTracerProvider``, so a pre-set real provider prevents them from
installing a bare exporter-less provider that would starve later task-root
tests.

Core-only installs (``pip install -e ".[dev]"``, no OTEL) are unaffected:
no provider is created, and OTEL-backed tests skip.
"""

from __future__ import annotations

import pytest

_HAS_OTEL = False
_global_provider = None
_global_memory = None

try:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from opentelemetry.trace import set_tracer_provider

    from shortchain.telemetry.association import AssociationInjectionSpanProcessor
    from shortchain.telemetry import task_span

    _global_provider = TracerProvider()
    _global_memory = InMemorySpanExporter()
    _global_provider.add_span_processor(AssociationInjectionSpanProcessor())
    _global_provider.add_span_processor(SimpleSpanProcessor(_global_memory))
    set_tracer_provider(_global_provider)
    _HAS_OTEL = True
except ImportError:  # pragma: no cover - exercised on core-only installs
    pass


@pytest.fixture(scope="session")
def otel_global_env() -> tuple:
    if not _HAS_OTEL:
        pytest.skip("opentelemetry extra is not installed")
    return _global_provider, _global_memory


@pytest.fixture
def otel_global_clear(otel_global_env):
    """Clear exported spans + any leaked task root before each test."""
    _, memory = otel_global_env
    memory.clear()
    task_span.end_task()  # close any leaked task root between tests
    memory.clear()
    yield memory
    memory.clear()
    return memory
