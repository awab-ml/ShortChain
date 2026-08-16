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