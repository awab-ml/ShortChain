"""Thin in-repo OTLP/HTTP receiver.

Accepts what the SDK sends (OTLP/HTTP protobuf; OTLP JSON for fixtures /
curl), decodes into :class:`OtelSpan`, feeds the :class:`TraceAssembler`,
and answers **HTTP 200** with an ``ExportTraceServiceResponse`` — never 429
(``OTLPSpanExporter`` would retry 429 and storm a full assembler). At the
inflight cap for NEW trace ids the response carries
``partial_success.rejected_spans``; spans for already-buffered trace ids
are still accepted.

Receiver is a thin consumer by design: no retry queues, gRPC, TLS, or tail
sampling — that is the OpenTelemetry Collector's job. ``workers=1`` is
mandatory (CLI enforces it).
"""

from __future__ import annotations

import base64
import binascii
import gzip
import json
from typing import Any

from google.protobuf.json_format import MessageToDict
from google.protobuf.message import DecodeError
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTracePartialSuccess,
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from shortchain.config import RuntimeConfig
from shortchain.ingest.otel import OtelSpan
from shortchain.telemetry.assembler import TraceAssembler
from shortchain.utils.logging import get_logger

log = get_logger(__name__)

PROTOBUF_CONTENT_TYPE = "application/x-protobuf"
JSON_CONTENT_TYPE = "application/json"

_STATUS_CODE_MAP = {0: "UNSET", 1: "OK", 2: "ERROR"}


# ---------------------------------------------------------------------------
# Attribute / span decoding helpers
# ---------------------------------------------------------------------------


def _any_value_to_python(value: Any) -> Any:
    """Convert an OTLP AnyValue (protobuf or JSON dict form) to plain data."""
    if value is None:
        return None
    if hasattr(value, "WhichOneof"):
        field = value.WhichOneof("value")
        if field == "string_value":
            return value.string_value
        if field == "bool_value":
            return value.bool_value
        if field == "int_value":
            return value.int_value
        if field == "double_value":
            return value.double_value
        if field == "array_value":
            return [_any_value_to_python(v) for v in value.array_value.values]
        if field == "kvlist_value":
            return {kv.key: _any_value_to_python(kv.value) for kv in value.kvlist_value.values}
        if field == "bytes_value":
            return value.bytes_value.hex()
        return None
    if isinstance(value, dict):
        # protojson-ish dict (MessageToDict output shape)
        try:
            return _parse_value_message(value)
        except KeyError:
            return None
    return value


def _parse_value_message(value: dict) -> Any:
    for json_key, proto_field in (
        ("stringValue", "string_value"),
        ("intValue", "int_value"),
        ("doubleValue", "double_value"),
        ("boolValue", "bool_value"),
        ("bytesValue", "bytes_value"),
    ):
        if json_key in value:
            return value[json_key]
    if "arrayValue" in value:
        return [_any_value_to_python(v) for v in value["arrayValue"].get("values", [])]
    if "kvlistValue" in value:
        return {
            kv["key"]: _any_value_to_python(kv.get("value"))
            for kv in value["kvlistValue"].get("values", [])
        }
    return None


def _proto_attrs_to_dict(attributes: Any) -> dict[str, Any]:
    return {kv.key: _any_value_to_python(kv.value) for kv in attributes}


def _json_attrs_to_dict(attributes: list) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in attributes:
        key = item.get("key")
        if not key:
            continue
        result[key] = _any_value_to_python(item.get("value"))
    return result


def _proto_events_to_list(events: Any) -> list[dict[str, Any]]:
    return [
        {
            "time_unix_nano": event.time_unix_nano,
            "name": event.name,
            "attributes": _proto_attrs_to_dict(event.attributes),
        }
        for event in events
    ]


def _json_events_to_list(events: list) -> list[dict[str, Any]]:
    return [
        {
            "time_unix_nano": event.get("timeUnixNano", 0),
            "name": event.get("name", ""),
            "attributes": _json_attrs_to_dict(event.get("attributes", [])),
        }
        for event in events
    ]


def _hex_from_json(value: Any) -> str:
    """OTLP JSON encodes trace/span ids as base64 strings."""
    if isinstance(value, str):
        try:
            return base64.b64decode(value).hex()
        except (binascii.Error, ValueError):
            return value  # already hex
    return str(value)


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------


def decode_protobuf(payload: bytes) -> list[OtelSpan]:
    """Decode an OTLP/protobuf ``ExportTraceServiceRequest`` body."""
    request = ExportTraceServiceRequest()
    try:
        request.ParseFromString(payload)
    except DecodeError as exc:
        raise ValueError(f"invalid OTLP protobuf: {exc}") from exc

    spans: list[OtelSpan] = []
    for resource_spans in request.resource_spans:
        resource = (
            _proto_attrs_to_dict(resource_spans.resource.attributes)
            if resource_spans.HasField("resource")
            else {}
        )
        for scope_spans in resource_spans.scope_spans:
            for span in scope_spans.spans:
                spans.append(
                    OtelSpan(
                        trace_id=span.trace_id.hex(),
                        span_id=span.span_id.hex(),
                        parent_span_id=span.parent_span_id.hex() or None,
                        name=span.name,
                        start_time_unix_nano=span.start_time_unix_nano,
                        end_time_unix_nano=span.end_time_unix_nano,
                        status_code=_STATUS_CODE_MAP.get(span.status.code, "UNSET"),
                        attributes=_proto_attrs_to_dict(span.attributes),
                        resource=resource,
                        events=_proto_events_to_list(span.events),
                    )
                )
    return spans


def decode_json(payload: bytes) -> list[OtelSpan]:
    """Decode an OTLP/JSON ``ExportTraceServiceRequest`` body."""
    try:
        body = json.loads(payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid OTLP JSON: {exc}") from exc

    spans: list[OtelSpan] = []
    for resource_spans in body.get("resourceSpans", []):
        resource_attrs = resource_spans.get("resource", {}).get("attributes", [])
        resource = _json_attrs_to_dict(resource_attrs)
        for scope_spans in resource_spans.get("scopeSpans", []):
            for span in scope_spans.get("spans", []):
                spans.append(
                    OtelSpan(
                        trace_id=_hex_from_json(span.get("traceId", "")),
                        span_id=_hex_from_json(span.get("spanId", "")),
                        parent_span_id=(
                            _hex_from_json(span["parentSpanId"])
                            if span.get("parentSpanId")
                            else None
                        ),
                        name=span.get("name", ""),
                        start_time_unix_nano=span.get("startTimeUnixNano", 0),
                        end_time_unix_nano=span.get("endTimeUnixNano", 0),
                        status_code=_status_from_json(span.get("status", {})),
                        attributes=_json_attrs_to_dict(span.get("attributes", [])),
                        resource=resource,
                        events=_json_events_to_list(span.get("events", [])),
                    )
                )
    return spans


def _status_from_json(status: dict | None) -> str:
    if not status:
        return "UNSET"
    return _STATUS_CODE_MAP.get(status.get("code", 0), "UNSET")


def decode_otlp(payload: bytes, content_type: str) -> list[OtelSpan]:
    """Decode an OTLP export body (protobuf or JSON) into ``OtelSpan``s."""
    if content_type == PROTOBUF_CONTENT_TYPE:
        return decode_protobuf(payload)
    if content_type == JSON_CONTENT_TYPE:
        return decode_json_type(payload)
    raise ValueError(f"unsupported content type: {content_type}")


def decode_json_type(payload: bytes) -> list[OtelSpan]:
    return decode_json(payload)


# ---------------------------------------------------------------------------
# Starlette application
# ---------------------------------------------------------------------------


def create_receiver_app(
    assembler: TraceAssembler,
    config: RuntimeConfig | None = None,
    *,
    api_key: str | None = None,
) -> Starlette:
    """Starlette app: ``POST /v1/traces`` (OTLP) + ``GET /metrics``."""
    cfg = config or assembler.config

    async def receive_traces(request: Request) -> Response:
        content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if content_type not in (PROTOBUF_CONTENT_TYPE, JSON_CONTENT_TYPE):
            return Response("", status_code=415, media_type="text/plain")

        if api_key:
            auth = request.headers.get("authorization", "")
            if auth != f"Bearer {api_key}":
                return Response("", status_code=401, media_type="text/plain")

        encoding = request.headers.get("content-encoding", "").strip().lower()
        raw = await request.body()
        if encoding == "gzip":
            try:
                raw = gzip.decompress(raw)
            except (OSError, EOFError):
                return Response("", status_code=400, media_type="text/plain")
        elif encoding and encoding not in {"", "identity"}:
            return Response("", status_code=415, media_type="text/plain")

        if len(raw) > cfg.max_body_bytes:
            return Response("", status_code=413, media_type="text/plain")

        try:
            spans = decode_otlp(raw, content_type)
        except (ValueError, json.JSONDecodeError) as exc:
            log.warning(f"decode error: {exc}")
            return Response("", status_code=400, media_type="text/plain")

        rejected = assembler.append(spans)
        partial = ExportTracePartialSuccess(
            rejected_spans=rejected,
            error_message="max_inflight_traces" if rejected else "",
        )
        if content_type == JSON_CONTENT_TYPE:
            payload = MessageToDict(ExportTraceServiceResponse(partial_success=partial))
            return JSONResponse(payload)
        payload = ExportTraceServiceResponse(partial_success=partial).SerializeToString()
        return Response(payload, media_type=PROTOBUF_CONTENT_TYPE)

    async def metrics(request: Request) -> Response:
        stats = assembler.stats()
        lines = ["# shortchain receiver metrics"]
        metrics_payload = stats["metrics"]
        for name, value in sorted(metrics_payload.items()):
            if isinstance(value, dict):
                for label, label_value in sorted(value.items()):
                    lines.append(
                        f'shortchain_otel_{name}{{reason="{label}"}} {label_value}'
                    )
            else:
                lines.append(f"shortchain_otel_{name} {value}")
        return Response("\n".join(lines) + "\n", media_type="text/plain")

    app = Starlette()
    app.add_route("/v1/traces", receive_traces, methods=["POST"])
    app.add_route("/metrics", metrics, methods=["GET"])
    return app


def _build_export_json(
    rejected_spans: int = 0,
    error_message: str = "",
) -> JSONResponse:
    response = ExportTraceServiceResponse(
        partial_success=ExportTracePartialSuccess(
            rejected_spans=rejected_spans,
            error_message=error_message,
        )
    )
    return JSONResponse(MessageToDict(response))