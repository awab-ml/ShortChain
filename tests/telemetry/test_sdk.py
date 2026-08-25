"""Tests for ShortChain.init (Section 5.2)."""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("opentelemetry.sdk.trace")

import shortchain.telemetry.sdk as sdk_mod
from shortchain.sdk import ShortChain
from shortchain.telemetry import instrument as inst
from shortchain.telemetry.sdk import _env_bool, _is_local_endpoint


class TestInitSignals:
    def test_init_sets_up_provider_and_exports(self, monkeypatch):
        """No receiver required: inspect provider + exporter wiring."""
        calls: dict = {}

        def fake_setup(**kw):
            from opentelemetry.sdk.trace import TracerProvider

            return TracerProvider()

        def fake_attach(provider, **kw):
            calls["attach"] = kw
            return ["exporter-a"]

        def fake_enable(provider, **kw):
            calls["enable"] = kw
            return ["threading"], []

        monkeypatch.setattr(sdk_mod, "setup_tracer_provider", fake_setup)
        monkeypatch.setattr(sdk_mod, "attach_exporters", fake_attach)
        monkeypatch.setattr(sdk_mod, "enable_instrumentors", fake_enable)

        ShortChain.init(
            api_key="sk-test",
            app_name="support-agent",
            endpoint="http://127.0.0.1:4318",
            content_tracing=True,
            headers={"X-Tenant": "acme"},
        )
        assert calls["attach"]["endpoint"] == "http://127.0.0.1:4318"
        assert calls["attach"]["api_key"] == "sk-test"
        assert calls["attach"]["headers"] == {"X-Tenant": "acme"}
        assert calls["enable"]["content_tracing"] is True

    def test_init_disabled_by_flag(self, monkeypatch):
        calls: list[str] = []

        def fake_setup(**kw):
            calls.append("setup")
            from opentelemetry.sdk.trace import TracerProvider

            return TracerProvider()

        monkeypatch.setattr(sdk_mod, "setup_tracer_provider", fake_setup)
        ShortChain.init(enabled=False)
        assert calls == []

    def test_init_disabled_by_env(self, monkeypatch):
        monkeypatch.setenv("SHORTCHAIN_TRACING_ENABLED", "0")
        calls = []

        def fake_setup(**kw):
            calls.append("setup")
            from opentelemetry.sdk.trace import TracerProvider

            return TracerProvider()

        monkeypatch.setattr(sdk_mod, "setup_tracer_provider", fake_setup)
        ShortChain.init()
        assert calls == []


class TestEnvHelpers:
    def test_env_bool_override(self, monkeypatch):
        monkeypatch.setenv("SHORTCHAIN_TRACE_CONTENT", "false")
        assert _env_bool("SHORTCHAIN_TRACE_CONTENT", True) is False
        monkeypatch.delenv("SHORTCHAIN_TRACE_CONTENT")
        assert _env_bool("SHORTCHAIN_TRACE_CONTENT", True) is True

    def test_local_endpoint(self):
        assert _is_local_endpoint("http://127.0.0.1:4318")
        assert _is_local_endpoint("http://localhost:4318")
        assert _is_local_endpoint("file://x.jsonl")
        assert not _is_local_endpoint("http://collector.example.com:4318")


class TestSdkFacade:
    def test_public_import_path(self):
        from shortchain.sdk import ShortChain as SC

        assert SC is ShortChain

    def test_no_traceloop_import(self):
        """The SDK must not depend on traceloop-sdk at import time."""
        import shortchain.sdk  # noqa: F401

        assert "traceloop" not in sys.modules


class TestSpanProduction:
    def test_span_postprocess_redaction_hook(self):
        """span_postprocess can redact attributes on ended spans."""
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        provider = inst.setup_tracer_provider()
        memory = InMemorySpanExporter()
        redacted: list[str] = []

        def redact(span):
            # SpanPostprocessProcessor unfreezes _attributes before the hook.
            if span._attributes.get("password"):
                span._attributes["password"] = "***"
            redacted.append("hook-called")

        provider.add_span_processor(inst.SpanPostprocessProcessor(memory, redact))
        tracer = provider.get_tracer("t")
        with tracer.start_as_current_span("x") as current:
            current.set_attribute("password", "s3cret")
        provider.force_flush()
        exported = memory.get_finished_spans()
        assert len(exported) == 1
        assert exported[0].attributes["password"] == "***"
        assert redacted == ["hook-called"]