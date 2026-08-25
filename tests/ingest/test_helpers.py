"""Tests for OTEL view models and attribute helpers (Section 2.1)."""

from __future__ import annotations

from shortchain.config import ProjectionConfig
from shortchain.ingest.otel import (
    OtelSpan,
    OtelTrace,
    _as_dict,
    _as_list,
    _first_attr,
    _parse_optional_bool,
)


# ---------------------------------------------------------------------------
# OtelSpan / OtelTrace views
# ---------------------------------------------------------------------------


class TestOtelSpan:
    def test_defaults(self):
        span = OtelSpan(
            trace_id="a" * 32,
            span_id="b" * 16,
            name="execute_tool foo",
            start_time_unix_nano=1,
            end_time_unix_nano=2,
        )
        assert span.parent_span_id is None
        assert span.status_code == "UNSET"
        assert span.attributes == {}
        assert span.resource == {}
        assert span.events == []

    def test_extra_fields(self):
        OtelSpan(
            trace_id="a" * 32,
            span_id="b" * 16,
            name="execute_tool foo",
            start_time_unix_nano=1,
            end_time_unix_nano=2,
            parent_span_id="c" * 16,
            status_code="ERROR",
            attributes={"gen_ai.tool.name": "foo"},
            resource={"service.name": "app"},
        )


class TestOtelTrace:
    def test_complete_reason_default(self):
        trace = OtelTrace(trace_id="a" * 32, spans=[])
        assert trace.complete_reason == "timeout"

    def test_with_spans(self):
        span = OtelSpan(
            trace_id="a" * 32,
            span_id="b" * 16,
            name="n",
            start_time_unix_nano=1,
            end_time_unix_nano=2,
        )
        trace = OtelTrace(trace_id="a" * 32, spans=[span], complete_reason="explicit")
        assert trace.spans[0].span_id == "b" * 16
        assert trace.complete_reason == "explicit"


# ---------------------------------------------------------------------------
# ProjectionConfig
# ---------------------------------------------------------------------------


class TestProjectionConfig:
    def test_defaults(self):
        cfg = ProjectionConfig()
        assert cfg.intent_strategy == "first_user"
        assert cfg.accept_gen_ai_task_id is False
        assert cfg.accept_task_status is False
        assert cfg.success_tools == []
        assert cfg.drop_tools == []
        assert cfg.max_observation_chars == 2000
        assert cfg.max_thought_chars == 2000
        assert cfg.require_intent is True
        assert cfg.require_tool_spans is True
        assert cfg.require_known_success is True
        assert cfg.max_spans == 200

    def test_list_defaults_are_independent(self):
        a = ProjectionConfig()
        b = ProjectionConfig()
        a.drop_tools.append("x")
        assert b.drop_tools == []


# ---------------------------------------------------------------------------
# _as_dict / _as_list
# ---------------------------------------------------------------------------


class TestAsDict:
    def test_dict_passthrough(self):
        assert _as_dict({"a": 1}) == {"a": 1}

    def test_json_string(self):
        assert _as_dict('{"a": 1}') == {"a": 1}

    def test_non_parsable_string(self):
        assert _as_dict("not json") == {}

    def test_json_string_scalar(self):
        assert _as_dict('"scalar"') == {}

    def test_other_types(self):
        assert _as_dict(None) == {}
        assert _as_dict([1, 2]) == {}
        assert _as_dict(42) == {}


class TestAsList:
    def test_list_passthrough(self):
        assert _as_list([1, 2]) == [1, 2]

    def test_json_string(self):
        assert _as_list('[1, 2]') == [1, 2]

    def test_singleton_string(self):
        assert _as_list("42") == [42]

    def test_bad_string(self):
        assert _as_list("nope") == []

    def test_other_types(self):
        assert _as_list(None) == []
        assert _as_list({"a": 1}) == []


# ---------------------------------------------------------------------------
# _first_attr
# ---------------------------------------------------------------------------


class TestFirstAttr:
    def test_first_present_non_empty(self):
        attrs = {"a": "", "b": "value"}
        assert _first_attr(attrs, "a", "b") == "value"

    def test_missing_all_keys(self):
        assert _first_attr({"z": 1}, "a", "b") is None

    def test_false_is_a_value(self):
        """False must be returned — the success extractor depends on this."""
        assert _first_attr({"a": False}, "a", "b") is False

    def test_zero_is_a_value(self):
        assert _first_attr({"a": 0}, "a", "b") == 0

    def test_empty_string_skipped(self):
        assert _first_attr({"a": "", "b": ""}, "a", "b") is None

    def test_none_skipped(self):
        assert _first_attr({"a": None, "b": "x"}, "a", "b") == "x"

    def test_first_key_wins_over_second(self):
        attrs = {"a": "first", "b": "second"}
        assert _first_attr(attrs, "a", "b") == "first"
        assert _first_attr(attrs, "b", "a") == "second"


# ---------------------------------------------------------------------------
# _parse_optional_bool
# ---------------------------------------------------------------------------


class TestParseOptionalBool:
    def test_native_bools(self):
        assert _parse_optional_bool(True) is True
        assert _parse_optional_bool(False) is False

    def test_numeric(self):
        assert _parse_optional_bool(1) is True
        assert _parse_optional_bool(0) is False
        assert _parse_optional_bool(1.0) is True
        assert _parse_optional_bool(0.0) is False

    def test_strings(self):
        assert _parse_optional_bool("true") is True
        assert _parse_optional_bool("True") is True
        assert _parse_optional_bool("false") is False
        assert _parse_optional_bool("False") is False
        assert _parse_optional_bool("1") is True
        assert _parse_optional_bool("0") is False
        assert _parse_optional_bool("yes") is True
        assert _parse_optional_bool("no") is False

    def test_absent_and_unparseable(self):
        assert _parse_optional_bool(None) is None
        assert _parse_optional_bool("") is None
        assert _parse_optional_bool("maybe") is None
        assert _parse_optional_bool(2) is None
        assert _parse_optional_bool([True]) is None