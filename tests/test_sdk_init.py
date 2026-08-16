"""Tests for the SDK tracer-provider / instrumentor setup (Section 5.1)."""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

pytest.importorskip("opentelemetry.sdk.trace")

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import ProxyTracerProvider

from shortchain.runtime import instrument as inst


def make_provider() -> TracerProvider:
    provider = inst.setup_tracer_provider(app_name="unit-test")
    memory = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    return provider


class TestSetupProvider:
    def test_creates_provider_when_global_is_proxy(self, monkeypatch):
        captured: dict = {}

        def fake_get():
            return ProxyTracerProvider()

        def fake_set(provider):
            captured["provider"] = provider

        monkeypatch.setattr(inst, "get_tracer_provider", fake_get)
        monkeypatch.setattr(inst, "set_tracer_provider", fake_set)
        provider = inst.setup_tracer_provider(app_name="app-x")
        assert isinstance(provider, TracerProvider)
        assert captured["provider"] is provider
        assert provider.resource.attributes["service.name"] == "app-x"

    def test_resource_attributes_merged(self, monkeypatch):
        captured: dict = {}

        def fake_get():
            return ProxyTracerProvider()

        def fake_set(provider):
            captured["provider"] = provider

        monkeypatch.setattr(inst, "get_tracer_provider", fake_get)
        monkeypatch.setattr(inst, "set_tracer_provider", fake_set)
        provider = inst.setup_tracer_provider(
            app_name="app-y", resource_attributes={"env": "test"}
        )
        assert provider.resource.attributes["env"] == "test"
        assert provider.resource.attributes["service.name"] == "app-y"

    def test_reuses_existing_provider(self, monkeypatch):
        existing = TracerProvider()

        def fake_get():
            return existing

        monkeypatch.setattr(inst, "get_tracer_provider", fake_get)
        provider = inst.setup_tracer_provider(app_name="second")
        assert provider is existing


class TestAttachExporters:
    def test_http_endpoint_normalised(self):
        provider = TracerProvider()
        exporters = inst.attach_exporters(provider, endpoint="http://127.0.0.1:4318")
        assert len(exporters) == 1
        assert exporters[0]._endpoint.endswith("/v1/traces")

    def test_display_endpoint_attaches_second(self):
        provider = TracerProvider()
        exporters = inst.attach_exporters(
            provider,
            endpoint="http://127.0.0.1:4318",
            display_endpoint="http://display:4318",
        )
        assert len(exporters) == 2

    def test_file_endpoint_creates_dump(self, tmp_path: Path):
        provider = TracerProvider()
        exporters = inst.attach_exporters(
            provider, endpoint=f"file://{tmp_path / 'spans.jsonl'}"
        )
        assert isinstance(exporters[0], inst.FileSpanExporter)

    def test_span_postprocess_called(self):
        provider = TracerProvider()
        memory = InMemorySpanExporter()
        calls: list[str] = []

        def postprocess(span) -> None:
            calls.append(span.name)

        provider.add_span_processor(inst.SpanPostprocessProcessor(memory, postprocess))
        tracer = provider.get_tracer("t")
        with tracer.start_as_current_span("hello"):
            pass
        memory.force_flush()
        assert calls == ["hello"]


class TestFileSpanExporter:
    def test_writes_json_lines(self, tmp_path: Path):
        path = tmp_path / "spans.jsonl"
        exporter = inst.FileSpanExporter(str(path))
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("t")
        with tracer.start_as_current_span("execute_tool foo"):
            pass
        exporter.shutdown()
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["name"] == "execute_tool foo"
        assert len(record["trace_id"]) == 32
        assert len(record["span_id"]) == 16


class TestContentTracing:
    def test_mirrors_traceloop_env(self, monkeypatch):
        monkeypatch.delenv("TRACELOOP_TRACE_CONTENT", raising=False)
        inst.set_content_tracing_env(True)
        assert inst.enable_instrumentors == inst.enable_instrumentors  # trivial

        import os

        os.environ.pop("TRACELOOP_TRACE_CONTENT", None)
        inst.set_content_tracing_env(True)
        assert os.environ["TRACELOOP_TRACE_CONTENT"] == "true"
        inst.set_content_tracing_env(False)
        assert os.environ["TRACELOOP_TRACE_CONTENT"] == "false"


class TestEnableInstrumentors:
    def test_no_framework_imports_do_not_abort(self):
        """Missing frameworks (langchain/openai/anthropic not installed in
        the test env) must not crash init; installed ones still enable."""
        provider = TracerProvider()
        enabled, skipped = inst.enable_instrumentors(provider)
        assert isinstance(enabled, list)
        assert "threading" in enabled
        # CrewAI / Agno / MCP / LiteLLM import cleanly without frameworks.
        assert "crewai" in enabled
        assert "agno" in enabled
        assert "mcp" in enabled
        assert "litellm" in enabled
        # langchain/openai/anthropic require their frameworks: skipped, not fatal.
        assert any("langchain" in s for s in skipped)

    def test_constructor_kwargs_signature_aware(self, monkeypatch):
        """use_attributes vs use_legacy_attributes resolved per signature."""
        calls: dict = {}

        class FakeLiteLLM:
            def __init__(self, *, use_legacy_attributes: bool = True):
                calls["ctor"] = {"use_legacy_attributes": use_legacy_attributes}

            def instrument(self, **kwargs):
                calls["instrument"] = kwargs

        fake_module = types.ModuleType("opentelemetry.instrumentation.litellm")
        fake_module.LiteLLMInstrumentor = FakeLiteLLM
        monkeypatch.setitem(
            __import__("sys").modules, "opentelemetry.instrumentation.litellm", fake_module
        )

        provider = TracerProvider()
        inst.enable_instrumentors(provider, instruments={"litellm"})
        assert calls["ctor"] == {"use_legacy_attributes": True}
        assert "tracer_provider" in calls["instrument"]

    def test_block_instruments_wins(self):
        provider = TracerProvider()
        enabled, skipped = inst.enable_instrumentors(
            provider, block_instruments={"mcp", "threading"}
        )
        assert "mcp" not in enabled
        assert "threading" not in enabled
        assert any("mcp" in s for s in skipped)

    def test_explicit_instruments_subset(self):
        provider = TracerProvider()
        enabled, _ = inst.enable_instrumentors(provider, instruments={"threading"})
        assert enabled == ["threading"]