"""OpenTelemetry / OpenLLMetry trace views and projection onto Trajectory.

Defines the *views* the projector consumes (``OtelSpan`` / ``OtelTrace``)
plus the low-level attribute helpers shared with the HALO-style loaders.
These are projector inputs, never seen by ``DatasetBuilder`` — the canonical
training ontology remains ``shortchain/ingest/schema.py`` (Trajectory / Span).

Attribute handling notes:
- OTLP attribute values may arrive as JSON strings (OTLP JSON + some
  instrumentations dump ``json.dumps(...)``); ``_as_dict`` / ``_as_list``
  normalise those.
- Lookup is ``_first_attr`` (first present, non-empty key), never
  ``dict.get(key, next_key)`` — that would treat the next key name as a
  default value.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from shortchain.config import ProjectionConfig


class OtelSpan(BaseModel):
    """A single span from an assembled OTLP trace (projector input view)."""

    trace_id: str                 # 32-char hex
    span_id: str                  # 16-char hex
    parent_span_id: str | None = None
    name: str
    start_time_unix_nano: int
    end_time_unix_nano: int
    status_code: str = "UNSET"    # UNSET | OK | ERROR  — NOT task success
    attributes: dict[str, Any] = Field(default_factory=dict)
    resource: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)


class OtelTrace(BaseModel):
    """Assembled spans for one ``trace_id`` (projector input view)."""

    trace_id: str
    spans: list[OtelSpan]
    complete_reason: str = "timeout"  # timeout | idle | root_ended | explicit


# ---------------------------------------------------------------------------
# Attribute helpers
# ---------------------------------------------------------------------------


def _as_dict(value: Any) -> dict[str, Any]:
    """Normalise an attributes field (dict or JSON string) to a dict."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
            return loaded if isinstance(loaded, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _as_list(value: Any) -> list[Any]:
    """Normalise a JSON-list field (list or JSON string) to a list."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
            return loaded if isinstance(loaded, list) else [loaded]
        except (json.JSONDecodeError, TypeError):
            return []
    return []


def _first_attr(attrs: dict[str, Any], *keys: str) -> Any:
    """Return the first present, non-empty attribute among *keys*.

    ``None`` and ``""`` are treated as missing; ``False`` / ``0`` are valid
    values. Never use ``dict.get(key, next_key)`` — that would treat the
    next key name as a default value.
    """
    for key in keys:
        if key not in attrs:
            continue
        value = attrs[key]
        if value is None or value == "":
            continue
        return value
    return None


def _parse_optional_bool(raw: Any) -> bool | None:
    """Parse a boolean attribute value.

    Returns
    -------
    bool | None
        ``True`` / ``False`` for known values (including the *string* forms
        OTLP JSON may deliver, e.g. ``"false"`` / ``"0"``); ``None`` when the
        value is absent or unparseable. ``False`` is a *present* value and is
        never collapsed to ``None``.
    """
    if raw is None or raw == "":
        return None
    if raw is True or raw is False:
        return raw
    if isinstance(raw, (int, float)) and raw in (0, 1):
        return bool(int(raw))
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no"):
            return False
    return None  # unparseable → absent (do not guess)


# ---------------------------------------------------------------------------
# Span classification
# ---------------------------------------------------------------------------

# OpenLLMetry operation names / span kinds. A span is assigned exactly one
# role; ``SKIP_OPS`` is checked FIRST so retriever/embedding spans (which
# LangChain marks ``traceloop.span.kind=task``) never become task rows.

TOOL_OPS = {"execute_tool"}
LLM_OPS = {"chat", "text_completion", "completion", "llm_request"}
SKIP_OPS = {"embeddings", "vector_db_retrieve", "handoff"}
SKIP_NAMES = {"mcp_tools", "tools/list", "shortchain.task"}


def _kind(span: OtelSpan) -> str:
    """Lower-cased span kind (traceloop.span.kind, else openinference)."""
    kind = _first_attr(
        span.attributes,
        "traceloop.span.kind",
        "openinference.span.kind",
    )
    return str(kind or "").lower()


def classify(span: OtelSpan) -> str:
    """Assign exactly one role: root|tool|llm|agent|task|skip|other."""
    attrs = span.attributes
    op = _first_attr(attrs, "gen_ai.operation.name", "llm.request.type")
    kind_l = _kind(span)

    if op in SKIP_OPS or kind_l in {"session", "handoff", "server"}:
        return "skip"
    if span.name in SKIP_NAMES or _first_attr(attrs, "shortchain.task_root"):
        if span.name == "shortchain.task" or _first_attr(attrs, "shortchain.task_root"):
            return "root"
        return "skip"
    if op in TOOL_OPS:
        return "tool"
    if _first_attr(attrs, "gen_ai.tool.name", "tool.name"):
        return "tool"
    if span.name.startswith("execute_tool ") or span.name.endswith(".tool"):
        return "tool"
    if span.name.startswith("function.") and kind_l in {"tool", ""}:
        return "tool"  # OpenInference / data/traces.jsonl
    if kind_l == "tool":
        return "tool"  # name still required at emit time
    if op in LLM_OPS or kind_l == "llm":
        return "llm"
    if kind_l in {"workflow", "agent"} or op == "invoke_agent":
        return "agent"
    if kind_l == "task" or op == "execute_task":
        return "task"
    return "other"


def _parse_span_name(name: str) -> str:
    """Extract a bare tool name from an instrumented span name (or "")."""
    if name.startswith("execute_tool "):
        return name[len("execute_tool "):].strip()
    if name == "tools/call.tool":
        return ""  # MCP client span: name lives in entity.input.tool_name
    if name.endswith(".tool"):
        return name[: -len(".tool")].strip()
    if name.startswith("function."):
        return name[len("function."):].strip()
    return ""


def extract_tool_name(span: OtelSpan) -> str | None:
    """Bare tool name for *span*, or ``None`` when it must not be emitted.

    Called only after ``classify`` returned ``"tool"``. Returns ``None`` for
    catalog listings / malformed spans (e.g. nameless ``kind=TOOL``
    ``mcp_tools`` rows) — those must never become ShortChain ``Span`` rows.
    """
    attrs = span.attributes
    name = _first_attr(attrs, "gen_ai.tool.name", "tool.name")
    if name is not None and not isinstance(name, str):
        name = None
    if not name and _kind(span) == "tool":
        name = _first_attr(attrs, "traceloop.entity.name")
        if name is not None and not isinstance(name, str):
            name = None
    if not name and span.name == "tools/call.tool":
        name = _first_attr(attrs, "traceloop.entity.input.tool_name")

    if not name:
        name = _parse_span_name(span.name)
    if not name:
        return None

    name = str(name).split("(")[0].strip()
    if not name or name in SKIP_NAMES:
        return None
    return name


# ---------------------------------------------------------------------------
# Trace structure helpers
# ---------------------------------------------------------------------------


def _span_by_id(trace: OtelTrace) -> dict[str, OtelSpan]:
    return {s.span_id: s for s in trace.spans}


def _find_root(trace: OtelTrace) -> OtelSpan | None:
    """Prefer the SDK ``shortchain.task`` span, else the outermost agent span.

    Fallbacks: a ``workflow``/``agent`` span without a classified parent of
    the same kind, then the span with empty ``parent_span_id``, then the
    earliest ``start_time_unix_nano``.
    """
    sdk = [
        s for s in trace.spans
        if s.name == "shortchain.task" or _first_attr(s.attributes, "shortchain.task_root")
    ]
    if sdk:
        return min(sdk, key=lambda s: s.start_time_unix_nano)

    by_id = {s.span_id: s for s in trace.spans}
    for s in sorted(trace.spans, key=lambda x: x.start_time_unix_nano):
        if _kind(s) not in {"workflow", "agent"}:
            continue
        parent = by_id.get(s.parent_span_id or "")
        if parent is not None and _kind(parent) in {"workflow", "agent"}:
            continue
        return s
    for s in sorted(trace.spans, key=lambda x: x.start_time_unix_nano):
        if not s.parent_span_id:
            return s
    return min(trace.spans, key=lambda s: s.start_time_unix_nano) if trace.spans else None


def _association_props(span: OtelSpan | None) -> dict[str, Any]:
    """Merged association properties for a span (blob + shortchain.* aliases)."""
    props: dict[str, Any] = {}
    if span is None:
        return props
    raw = _first_attr(span.attributes, "traceloop.association.properties")
    if raw:
        props.update(_as_dict(raw))
    for key in ("task_id", "intent", "app_name", "success"):
        value = _first_attr(span.attributes, f"shortchain.{key}")
        if value is not None:
            props[key] = value
    return props


# ---------------------------------------------------------------------------
# Message helpers (OpenLLMetry gen_ai.input.messages / OpenInference llm.*)
# ---------------------------------------------------------------------------


def _message_role(msg: Any) -> str:
    if not isinstance(msg, dict):
        return ""
    return str(msg.get("role") or "").lower()


def _message_text(msg: Any) -> str:
    """Text of a chat message (gen_ai parts[] or plain content)."""
    if not isinstance(msg, dict):
        return ""
    parts = msg.get("parts")
    if isinstance(parts, list):
        texts = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("text", "input_text", "content"):
                content = part.get("content")
                if isinstance(content, str):
                    texts.append(content)
        if texts:
            return " ".join(t for t in texts if t).strip()
    content = msg.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = [
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict)
        ]
        return " ".join(t for t in texts if t).strip()
    return ""


def _message_has_tool_calls(msg: Any) -> bool:
    if not isinstance(msg, dict):
        return False
    parts = msg.get("parts")
    if isinstance(parts, list):
        for part in parts:
            if isinstance(part, dict) and part.get("type") in ("tool_call", "function_call"):
                return True
    return bool(_as_list(msg.get("tool_calls")))


def _llm_messages(span: OtelSpan) -> list[Any]:
    """Input messages of an LLM span (gen_ai.input.messages / indexed attrs)."""
    attrs = span.attributes
    value = _first_attr(attrs, "gen_ai.input.messages", "llm.input_messages")
    if value:
        return _as_list(value)
    messages: list[Any] = []
    suffix = ".message.content"
    indexed: dict[int, Any] = {}
    for key in attrs:
        if not key.startswith("llm.input_messages."):
            continue
        body = key[len("llm.input_messages."):]
        if not body.endswith(suffix):
            continue
        try:
            idx = int(body[: -len(suffix)])
        except ValueError:
            continue
        indexed[idx] = attrs[key]
    for idx in sorted(indexed):
        messages.append({"role": "user", "content": indexed[idx]})
    return messages


def _iter_user_messages(trace: OtelTrace) -> list[tuple[OtelSpan, int, dict]]:
    """Ordered (span, index_in_span, message) for user/human messages."""
    result: list[tuple[OtelSpan, int, dict[str, Any]]] = []
    for span in sorted(trace.spans, key=lambda s: s.start_time_unix_nano):
        if classify(span) != "llm":
            continue
        for idx, msg in enumerate(_llm_messages(span)):
            if not isinstance(msg, dict):
                continue
            if _message_role(msg) not in {"user", "human"}:
                continue
            if not _message_text(msg):
                continue
            result.append((span, idx, msg))
    return result


def _first_tool_call_message(
    trace: OtelTrace,
) -> tuple[OtelSpan, int, Any] | None:
    """Earliest message that carries tool calls (for last_user_before_tools)."""
    for span in sorted(trace.spans, key=lambda s: s.start_time_unix_nano):
        if classify(span) != "llm":
            continue
        for idx, msg in enumerate(_llm_messages(span)):
            if _message_has_tool_calls(msg):
                return span, idx, msg
    return None


def _choose_intent_message(
    trace: OtelTrace,
    strategy: str,
) -> tuple[str, str] | None:
    """Pick intent text + source from user messages.

    ``first_user``: first non-empty user message in the earliest LLM span.
    ``last_user_before_tools``: last user message before the first tool-call
    message, skipping an over-long preamble when a shorter user message
    exists later (HALO's "real task" idea, no AppWorld markers).
    """
    users = _iter_user_messages(trace)
    if not users:
        return None
    if strategy == "last_user_before_tools":
        first_tool = _first_tool_call_message(trace)
        if first_tool is not None:
            tool_span, _, _ = first_tool
            candidates = [
                m for m in users
                if m[0].start_time_unix_nano <= tool_span.start_time_unix_nano
            ]
            if candidates:
                users = candidates
        preamble = users[0]
        if len(_message_text(preamble[2])) > 2000 and len(users) > 1:
            users = users[1:]
        if users:
            return _message_text(users[-1][2]), "last_user_before_tools"
        return None
    return _message_text(users[0][2]), "user_message"


# ---------------------------------------------------------------------------
# Field extractors
# ---------------------------------------------------------------------------


def _unwrap_task_input(value: Any) -> str:
    """Unwrap ``gen_ai.task.input`` / entity-input blobs to a string."""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if parsed is None:
            return value.strip() if value.strip() else ""
        if isinstance(parsed, str):
            return parsed.strip() if parsed.strip() else ""
        value = parsed
    if isinstance(value, dict):
        for key in ("input", "question", "query"):
            nested = value.get(key)
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
        inputs = value.get("inputs")
        if inputs is not None:
            return _unwrap_task_input_safe(inputs)
        messages = value.get("messages")
        for msg in _as_list(messages):
            if _message_role(msg) in {"user", "human"}:
                text = _message_text(msg)
                if text:
                    return text
        for key in ("content", "text"):
            nested = value.get(key)
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    if isinstance(value, str) and value.strip():
        return value.strip()
    return ""


def _unwrap_task_input_safe(value: Any) -> str:
    """Thin wrapper so ``extract_task_input`` recursion stays bounded."""
    if isinstance(value, dict):
        for key in ("input", "question", "query"):
            nested = value.get(key)
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
        return json.dumps(value) if value else ""
    return str(value).strip() if isinstance(value, str) and value.strip() else ""


def extract_intent(
    trace: OtelTrace,
    config: ProjectionConfig,
    root: OtelSpan | None,
) -> tuple[str, str]:
    """Return ``(intent, intent_source)`` (source "" means missing)."""
    props = _association_props(root)
    value = _first_attr(props, "intent", "shortchain.intent")
    if value:
        return str(value), "association"

    for span in sorted(trace.spans, key=lambda s: s.start_time_unix_nano):
        raw = _first_attr(span.attributes, "gen_ai.task.input")
        if raw:
            unwrapped = _unwrap_task_input(raw)
            if unwrapped:
                return unwrapped, "task_input"

    chosen = _choose_intent_message(trace, config.intent_strategy)
    if chosen is not None:
        return chosen

    if root is not None:
        raw = _first_attr(root.attributes, "traceloop.entity.input")
        if raw:
            unwrapped = _unwrap_task_input(raw)
            if unwrapped:
                return unwrapped, "entity_input"

    return "", ""


def extract_task_id(
    trace: OtelTrace,
    config: ProjectionConfig,
    root: OtelSpan | None,
) -> str:
    """Stable task id: association, optional gen_ai.task.id, conversation, trace."""
    props = _association_props(root)
    value = _first_attr(props, "task_id", "shortchain.task_id")
    if value is not None:
        return str(value)

    if config.accept_gen_ai_task_id:
        for span in sorted(trace.spans, key=lambda s: s.start_time_unix_nano):
            if classify(span) not in {"root", "agent"}:
                continue
            value = _first_attr(span.attributes, "gen_ai.task.id")
            if value:
                return str(value)

    for span in sorted(trace.spans, key=lambda s: s.start_time_unix_nano):
        value = _first_attr(span.attributes, "gen_ai.conversation.id")
        if value:
            return str(value)
    return trace.trace_id


def extract_success(span: OtelSpan) -> bool | None:
    """None = absent/unparseable; False = known failure; True = known success."""
    raw = _first_attr(
        span.attributes,
        "shortchain.success",
        "traceloop.association.properties.success",
    )
    if raw is None:
        blob = _first_attr(span.attributes, "traceloop.association.properties")
        if blob:
            raw = _as_dict(blob).get("success")
    return _parse_optional_bool(raw)


def resolve_success(
    trace: OtelTrace,
    config: ProjectionConfig,
    root: OtelSpan | None,
) -> tuple[bool, str]:
    """Return ``(success, success_source)``; unknown → ``(False, "unknown")``.

    Priority: (1) SDK root association, (2) any-span association alias,
    (3) ``gen_ai.task.status`` only if ``accept_task_status``,
    (4) ``success_tools`` heuristic if enabled, (5) unknown.
    """
    root_first = [root] if root is not None else []
    others = sorted(
        (s for s in trace.spans if s is not root),
        key=lambda s: s.start_time_unix_nano,
    )
    for span in [*root_first, *others]:
        value = extract_success(span)
        if value is not None:
            return value, "association"

    if config.accept_task_status:
        for span in [*root_first, *others]:
            status = str(_first_attr(span.attributes, "gen_ai.task.status") or "").lower()
            if status in ("success", "failure"):
                return status == "success", "task_status"

    if config.success_tools:
        last_tool: str | None = None
        for span in sorted(trace.spans, key=lambda s: s.start_time_unix_nano):
            if classify(span) != "tool":
                continue
            name = extract_tool_name(span)
            if name:
                last_tool = name
        if last_tool is not None:
            return last_tool in config.success_tools, "heuristic"

    return False, "unknown"


def extract_app_name(
    trace: OtelTrace,
    root: OtelSpan | None,
) -> str:
    """Association first, then resource service.name, workflow/agent names."""
    props = _association_props(root)
    value = _first_attr(props, "app_name", "shortchain.app_name")
    if value:
        return str(value)

    for span in [root, *trace.spans]:
        if span is None:
            continue
        value = _first_attr(span.resource, "service.name")
        if value:
            return str(value)
    for span in [root, *trace.spans]:
        if span is None:
            continue
        value = _first_attr(span.attributes, "traceloop.workflow.name")
        if value:
            return str(value)
        value = _first_attr(span.attributes, "gen_ai.agent.name")
        if value:
            return str(value)
    return ""


def nearest_agent_name(
    span: OtelSpan,
    by_id: dict[str, OtelSpan],
) -> str:
    """Agent name for a tool span: self, nearest ancestor agent span, else ""."""
    value = _first_attr(span.attributes, "gen_ai.agent.name")
    if value:
        return str(value)
    current = by_id.get(span.parent_span_id or "")
    seen: set[str] = set()
    while current is not None and current.span_id not in seen:
        seen.add(current.span_id)
        value = _first_attr(current.attributes, "gen_ai.agent.name")
        if value:
            return str(value)
        current = by_id.get(current.parent_span_id or "")
    return ""


def extract_tool_arguments(span: OtelSpan) -> str:
    """JSON-ish argument string for a tool span ("" when absent)."""
    attrs = span.attributes
    value = _first_attr(
        attrs,
        "gen_ai.tool.call.arguments",
        "traceloop.entity.input",
        "input.value",
    )
    if value is None:
        return ""
    loaded = _as_dict(value)
    if loaded:
        for key in ("arguments", "input_str"):
            nested = loaded.get(key)
            if nested is not None and not isinstance(nested, (dict, list)):
                return str(nested)
        return json.dumps(loaded, ensure_ascii=False)
    return str(value)


def extract_tool_observation(
    span: OtelSpan,
    max_chars: int = 2000,
) -> str:
    """Observation for a tool span (empty string when absent)."""
    value = _first_attr(
        span.attributes,
        "gen_ai.tool.call.result",
        "traceloop.entity.output",
        "output.value",
        "gen_ai.task.output",
    )
    if value is None:
        return ""
    if isinstance(value, dict):
        value = json.dumps(value, ensure_ascii=False)
    return str(value)[:max_chars]


def extract_tool_thoughts(
    trace: OtelTrace,
    span: OtelSpan,
    max_chars: int = 2000,
) -> str:
    """Best-effort thoughts from the preceding sibling LLM span (or "")."""
    llm_spans = sorted(
        (s for s in trace.spans if classify(s) == "llm"),
        key=lambda s: s.end_time_unix_nano,
    )
    previous: OtelSpan | None = None
    for candidate in llm_spans:
        if candidate.end_time_unix_nano <= span.start_time_unix_nano:
            previous = candidate
    if previous is None:
        return ""

    text = _assistant_text_before_tool_calls(previous)
    if not text:
        completion = _first_attr(previous.attributes, "gen_ai.completion")
        if completion is None:
            completion = _first_attr(previous.attributes, "llm.completions")
        if completion is not None:
            if isinstance(completion, list):
                text = ", ".join(
                    str(c.get("text") or "") for c in completion if isinstance(c, dict)
                )
            else:
                text = str(completion)
    return text[:max_chars]


def _assistant_text_before_tool_calls(span: OtelSpan) -> str:
    """Assistant text parts before the first tool-call part (output messages)."""
    attrs = span.attributes
    messages = _first_attr(attrs, "gen_ai.output.messages", "gen_ai.output.message")
    if messages is None:
        completion = _first_attr(attrs, "gen_ai.completion")
        return str(completion) if completion is not None else ""
    results: list[str] = []
    for msg in _as_list(messages):
        if not isinstance(msg, dict) or _message_role(msg) != "assistant":
            continue
        parts = msg.get("parts")
        if not isinstance(parts, list):
            text = _message_text(msg)
            if text:
                results.append(text)
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("tool_call", "function_call"):
                break
            if part.get("type") in ("text", "input_text", "content"):
                content = part.get("content")
                if isinstance(content, str) and content.strip():
                    results.append(content)
    return " ".join(results).strip()