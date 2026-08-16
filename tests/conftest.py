"""Shared pytest fixtures for the SDK / task-root test files.

The OTel global TracerProvider can only be set ONCE per process (OTEL
refuses overrides), so every test module that traces through the global
provider must share one env: an in-memory exporter + the association
injection processor.
"""

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import set_tracer_provider

from shortchain.runtime.association import AssociationInjectionSpanProcessor
from shortchain.runtime import task_span


@pytest.fixture(scope="session")
def otel_global_env() -> tuple[TracerProvider, InMemorySpanExporter]:
    provider = TracerProvider()
    memory = InMemorySpanExporter()
    provider.add_span_processor(AssociationInjectionSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(memory))
    set_tracer_provider(provider)
    return provider, memory


@pytest.fixture
def otel_global_clear(otel_global_env) -> InMemorySpanExporter:
    """Clear exported spans + any leaked task root before each test."""
    _, memory = otel_global_env
    memory.clear()
    task_span.end_task()  # close any leaked task root between tests
    memory.clear()
    yield memory
    memory.clear()
    return memory