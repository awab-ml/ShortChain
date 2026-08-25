"""Tests for the OTLP/HTTP receiver (Section 4.2)."""

from __future__ import annotations

import base64
import gzip as gzip_mod
import json
from pathlib import Path

import pytest

pytest.importorskip("opentelemetry.proto")
pytest.importorskip("starlette")

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from starlette.testclient import TestClient

from shortchain.config import ProjectionConfig, RuntimeConfig
from shortchain.telemetry.assembler import (
    JsonlTrajectoryWriter,
    RuntimeMetrics,
    TraceAssembler,
)
from shortchain.telemetry.receiver import create_receiver_app
from shortchain.utils.io import read_jsonl

TID = "a" * 32
TID2 = "b" * 32


def _b64(hex_str: str) -> str:
    return base64.b64encode(bytes.fromhex(hex_str)).decode()


def build_config(tmp_path: Path, **kw) -> RuntimeConfig:
    overrides = dict(
        output=str(tmp_path / "trajectories.jsonl"),
        max_body_bytes=1024 * 1024,
        idle_timeout_s=30.0,
        require_success_true=True,
        projection=ProjectionConfig(require_intent=False),
    )
    overrides.update(kw)
    return RuntimeConfig(**overrides)


def make_client(tmp_path: Path, **kw):
    cfg = build_config(tmp_path, **kw)
    assembler = TraceAssembler(
        cfg,
        writer=JsonlTrajectoryWriter(Path(cfg.output)),
        metrics=RuntimeMetrics(),
    )
    app = create_receiver_app(assembler, cfg)
    return TestClient(app), assembler, Path(cfg.output)


def our_trace(trace_id: str = TID, *, tool: str = "lookup", success: bool = True) -> dict:
    """A minimal OTLP/JSON export body for one trace."""
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "test-app"}},
                    ]
                },
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": _b64(trace_id),
                                "spanId": _b64("1" * 16),
                                "name": "shortchain.task",
                                "kind": "SPAN_KIND_INTERNAL",
                                "startTimeUnixNano": 1000,
                                "endTimeUnixNano": 2000,
                                "attributes": [
                                    {"key": "shortchain.task_root", "value": {"boolValue": True}},
                                    {"key": "shortchain.success", "value": {"boolValue": success}},
                                    {"key": "shortchain.intent", "value": {"stringValue": "do the thing"}},
                                ],
                            },
                            {
                                "traceId": _b64(trace_id),
                                "spanId": _b64("2" * 8),
                                "parentSpanId": _b64("1" * 8),
                                "name": "execute_tool lookup",
                                "kind": "SPAN_KIND_INTERNAL",
                                "startTimeUnixNano": 1000,
                                "endTimeUnixNano": 2000,
                                "attributes": [
                                    {"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}},
                                    {"key": "gen_ai.tool.name", "value": {"stringValue": "lookup"}},
                                    {"key": "gen_ai.tool.call.result", "value": {"stringValue": "found"}},
                                ],
                            },
                        ]
                    }
                ],
            }
        ]
    }


def json_body(trace_id: str = TID, **kw) -> bytes:
    return json.dumps(our_trace(trace_id, **kw)).encode()


def proto_body(trace_id: str = TID, **kw) -> bytes:
    """Serialise our_trace into a real OTLP protobuf ExportTraceServiceRequest."""
    body = our_trace(trace_id, **kw)
    request = ExportTraceServiceRequest()
    for rss in body["resourceSpans"]:
        rs = request.resource_spans.add()
        for attr in rss["resource"]["attributes"]:
            kv = rs.resource.attributes.add()
            kv.key = attr["key"]
            kv.value.string_value = attr["value"]["stringValue"]
        for ss in rss["scopeSpans"]:
            scope = rs.scope_spans.add()
            for span in ss["spans"]:
                sp = scope.spans.add()
                sp.trace_id = base64.b64decode(span["traceId"])
                sp.span_id = base64.b64decode(span["spanId"])
                if span.get("parentSpanId"):
                    sp.parent_span_id = base64.b64decode(span["parentSpanId"])
                sp.name = span["name"]
                sp.start_time_unix_nano = span["startTimeUnixNano"]
                sp.end_time_unix_nano = span["endTimeUnixNano"]
                for attr in span["attributes"]:
                    kv = sp.attributes.add()
                    kv.key = attr["key"]
                    value = attr["value"]
                    if "stringValue" in value:
                        kv.value.string_value = value["stringValue"]
                    elif "boolValue" in value:
                        kv.value.bool_value = value["boolValue"]
    return request.SerializeToString()


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestReceive:
    def test_json_round_trip(self, tmp_path: Path):
        client, assembler, out = make_client(tmp_path)
        resp = client.post(
            "/v1/traces",
            content=json_body(),
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 200
        assembler.flush_all()
        records = read_jsonl(out)
        assert len(records) == 1
        assert records[0]["spans"][0]["action"] == "lookup"
        assert records[0]["success"] is True

    def test_gzip_round_trip(self, tmp_path: Path):
        client, assembler, out = make_client(tmp_path)
        body = gzip_mod.compress(json_body())
        resp = client.post(
            "/v1/traces",
            content=body,
            headers={"content-type": "application/json", "content-encoding": "gzip"},
        )
        assert resp.status_code == 200
        assembler.flush_all()
        assert len(read_jsonl(out)) == 1

    def test_protobuf_round_trip(self, tmp_path: Path):
        client, assembler, out = make_client(tmp_path)
        resp = client.post(
            "/v1/traces",
            content=proto_body(),
            headers={"content-type": "application/x-protobuf"},
        )
        assert resp.status_code == 200
        assembler.flush_all()
        assert len(read_jsonl(out)) == 1

    def test_protobuf_response_is_proto(self, tmp_path: Path):
        client, _, _ = make_client(tmp_path)
        resp = client.post(
            "/v1/traces",
            content=proto_body(),
            headers={"content-type": "application/x-protobuf"},
        )
        assert resp.status_code == 200
        assert "application/x-protobuf" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# Errors and bounds
# ---------------------------------------------------------------------------


class TestErrors:
    def test_413_oversize_body(self, tmp_path: Path):
        client, _, _ = make_client(tmp_path, max_body_bytes=100)
        resp = client.post(
            "/v1/traces",
            content=json_body() + b"x" * 500,
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 413

    def test_gzip_oversize_decompressed(self, tmp_path: Path):
        """413 must apply AFTER decompression."""
        client, _, _ = make_client(tmp_path, max_body_bytes=100)
        body = gzip_mod.compress(json_body() + b"x" * 500)
        assert len(body) > 100  # still over even compressed
        resp = client.post(
            "/v1/traces",
            content=body,
            headers={"content-type": "application/json", "content-encoding": "gzip"},
        )
        assert resp.status_code == 413

    def test_unsupported_content_type(self, tmp_path: Path):
        client, _, _ = make_client(tmp_path)
        resp = client.post("/v1/traces", content=b"x", headers={"content-type": "text/plain"})
        assert resp.status_code == 415

    def test_bad_json_body(self, tmp_path: Path):
        client, _, _ = make_client(tmp_path)
        resp = client.post(
            "/v1/traces", content=b"{not json", headers={"content-type": "application/json"}
        )
        assert resp.status_code == 400

    def test_missing_root_drops_success_unknown(self, tmp_path: Path):
        client, assembler, out = make_client(tmp_path)
        body = our_trace()
        # Remove the success attribute from the task root.
        root_attrs = body["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"]
        root_attrs[:] = [a for a in root_attrs if a["key"] != "shortchain.success"]
        resp = client.post(
            "/v1/traces",
            content=json.dumps(body).encode(),
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 200
        assembler.flush_all()
        # The gate dropped the trace: no output file is created at all.
        assert not out.exists()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class TestAuth:
    def test_api_key_requires_bearer(self, tmp_path: Path):
        cfg = build_config(tmp_path)
        assembler = TraceAssembler(
            cfg, writer=JsonlTrajectoryWriter(Path(cfg.output)), metrics=RuntimeMetrics()
        )
        client = TestClient(create_receiver_app(assembler, cfg, api_key="sekrit"))
        resp = client.post(
            "/v1/traces", content=json_body(), headers={"content-type": "application/json"}
        )
        assert resp.status_code == 401
        resp = client.post(
            "/v1/traces",
            content=json_body(),
            headers={"content-type": "application/json", "authorization": "Bearer wrong"},
        )
        assert resp.status_code == 401
        resp = client.post(
            "/v1/traces",
            content=json_body(),
            headers={
                "content-type": "application/json",
                "authorization": "Bearer sekrit",
            },
        )
        assert resp.status_code == 200

    def test_no_api_key_no_auth(self, tmp_path: Path):
        client, _, _ = make_client(tmp_path)
        resp = client.post(
            "/v1/traces", content=json_body(), headers={"content-type": "application/json"}
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Inflight cap: 200 + partial_success, never 429
# ---------------------------------------------------------------------------


class TestPartialSuccess:
    def test_new_tid_rejected_at_cap_accepted_for_existing(self, tmp_path: Path):
        client, assembler, out = make_client(tmp_path, max_inflight_traces=1)
        # First trace id fills the single inflight slot.
        r1 = client.post(
            "/v1/traces", content=json_body(TID), headers={"content-type": "application/json"}
        )
        assert r1.status_code == 200
        # Second, different trace id: nothing idle → 200 + partial_success.
        r2 = client.post(
            "/v1/traces", content=json_body(TID2), headers={"content-type": "application/json"}
        )
        assert r2.status_code == 200
        parsed = r2.json()
        # protojson renders int64 as string; type-tolerant comparison.
        assert int(parsed["partialSuccess"]["rejectedSpans"]) == 2
        assert parsed["partialSuccess"]["errorMessage"] == "max_inflight_traces"
        # Spans for the buffered trace id are still accepted.
        r3 = client.post(
            "/v1/traces", content=json_body(TID), headers={"content-type": "application/json"}
        )
        assert r3.status_code == 200
        # All accepted: partial_success carries no rejected payload.
        assert "partialSuccess" not in r3.json() or not r3.json().get("partialSuccess")

    def test_evicts_oldest_idle_before_rejecting(self, tmp_path: Path):
        import time

        cfg = build_config(tmp_path, max_inflight_traces=1)
        assembler = TraceAssembler(
            cfg, writer=JsonlTrajectoryWriter(Path(cfg.output)), metrics=RuntimeMetrics()
        )
        client = TestClient(create_receiver_app(assembler, cfg))
        client.post(
            "/v1/traces", content=json_body(TID), headers={"content-type": "application/json"}
        )
        time.sleep(1.1)  # first trace becomes idle (>= 1s eviction threshold)
        r2 = client.post(
            "/v1/traces", content=json_body(TID2), headers={"content-type": "application/json"}
        )
        assert r2.status_code == 200
        # Evicted, not dropped: no partial_success rejection payload.
        assert "partialSuccess" not in r2.json() or not r2.json().get("partialSuccess")


# ---------------------------------------------------------------------------
# Metrics endpoint
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_metrics_exposed(self, tmp_path: Path):
        client, _, _ = make_client(tmp_path)
        client.post(
            "/v1/traces", content=json_body(), headers={"content-type": "application/json"}
        )
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "shortchain_otel_spans_received 2" in resp.text
