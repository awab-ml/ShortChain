"""Shared pytest fixtures for the SDK / task-root test files.

The OTel global TracerProvider can only be set ONCE per process (OTEL
refuses overrides), so every test module that traces through the global
provider must share one env: an in-memory exporter + the association
injection processor.

These imports are optional: ``pip install -e ".[dev]"`` must still collect
and run the core training tests. Runtime/SDK tests skip when the ``otel``
extra is missing.
"""

from __future__ import annotations

import pytest

try:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from opentelemetry.trace import set_tracer_provider

    from shortchain.telemetry.association import AssociationInjectionSpanProcessor
    from shortchain.telemetry import task_span

    _HAS_OTEL = True
except ImportError:  # pragma: no cover - exercised on core-only installs
    _HAS_OTEL = False


@pytest.fixture(scope="session")
def otel_global_env() -> tuple:
    if not _HAS_OTEL:
        pytest.skip("opentelemetry extra is not installed")
    provider = TracerProvider()
    memory = InMemorySpanExporter()
    provider.add_span_processor(AssociationInjectionSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(memory))
    set_tracer_provider(provider)
    return provider, memory


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
