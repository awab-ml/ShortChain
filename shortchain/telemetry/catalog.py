"""Tool-catalog extraction from OTEL traces (and user-file merge).

Builds ``{tool_name: description}`` from every available source before
``DatasetBuilder`` derives the negative-sampling pool and tool features:

1. LLM spans' ``gen_ai.tool.definitions`` (v0.5+ single JSON-array attribute)
   and the legacy indexed form ``gen_ai.tool.definitions.{i}.name`` /
   ``.{i}.description``.
2. Legacy ``llm.request.functions`` indexed attributes (``{i}.name``).
3. Tool spans' ``gen_ai.tool.description`` / ``tool.description`` (Agno).
4. MCP listing spans (``mcp.tools.listed`` / ``tools/list`` output).
5. Names seen only as calls get ``description=""``.

Merge rule: the user-provided catalog (``--catalog`` in
``scripts/build_dataset.py``) wins for any overlapping name.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from shortchain.ingest.otel import (
    OtelSpan,
    OtelTrace,
    _as_list,
    _first_attr,
    classify,
    extract_tool_name,
)


def _definition_pairs(value: Any) -> list[tuple[str, str]]:
    """Normalise a definitions blob to ``[(name, description)]``."""
    pairs: list[tuple[str, str]] = []
    for entry in _as_list(value):
        if isinstance(entry, str):
            pairs.append((entry, ""))
            continue
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "")
        if not name:
            continue
        desc = entry.get("description")
        pairs.append((name, str(desc) if desc is not None else ""))
    return pairs


def _indexed_definitions(attrs: dict[str, Any], prefix: str) -> list[tuple[str, str]]:
    """Parse indexed attributes ``{prefix}.{i}.name`` / ``.{i}.description``."""
    by_index: dict[int, dict[str, str]] = defaultdict(dict)
    for key, value in attrs.items():
        if not key.startswith(prefix + "."):
            continue
        rest = key[len(prefix) + 1:]
        parts = rest.split(".")
        if len(parts) < 2:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        field = parts[1]
        if field in ("name", "description"):
            by_index[index][field] = str(value)
    return [(d["name"], d.get("description", "")) for i, d in sorted(by_index.items()) if d.get("name")]


def _mcp_listed_names(value: Any) -> list[tuple[str, str]]:
    """Names from ``mcp.tools.listed``; descriptions are not usually present."""
    pairs: list[tuple[str, str]] = []
    listed = _as_list(value)
    for entry in listed:
        if isinstance(entry, str):
            pairs.append((entry, ""))
        elif isinstance(entry, dict) and entry.get("name"):
            pairs.append((str(entry["name"]), str(entry.get("description") or "")))
    return pairs


def _tools_list_output(span: OtelSpan) -> list[tuple[str, str]]:
    """Tool names from a ``tools/list`` MCP span's output attribute."""
    if span.name != "tools/list":
        return []
    value = _first_attr(
        span.attributes,
        "traceloop.entity.output",
        "output.value",
    )
    if value is None:
        return []
    listed = _as_list(value)
    pairs: list[tuple[str, str]] = []
    for entry in listed:
        if isinstance(entry, str):
            pairs.append((entry, ""))
        elif isinstance(entry, dict):
            name = entry.get("name")
            if name:
                pairs.append((str(name), str(entry.get("description") or "")))
    return pairs


def build_catalog(traces: list[OtelTrace]) -> dict[str, str]:
    """Discover the tool catalog across traces. User merge handled separately."""
    catalog: dict[str, str] = {}
    for trace in traces:
        for span in trace.spans:
            attrs = span.attributes
            role = classify(span)

            # 1. LLM span definitions (single JSON array, or indexed legacy).
            if role == "llm":
                raw = _first_attr(attrs, "gen_ai.tool.definitions")
                if raw:
                    for name, desc in _definition_pairs(raw):
                        catalog.setdefault(name, desc)
                for name, desc in _indexed_definitions(attrs, "gen_ai.tool.definitions"):
                    catalog.setdefault(name, desc)
                for name, desc in _indexed_definitions(attrs, "llm.request.functions"):
                    catalog.setdefault(name, desc)

            # 3. Tool span descriptions (Agno / generic).
            if role == "tool":
                name = extract_tool_name(span)
                if name:
                    desc = _first_attr(attrs, "gen_ai.tool.description", "tool.description")
                    catalog.setdefault(name, str(desc) if desc is not None else "")

            # 4. MCP listings.
            listed = _first_attr(attrs, "mcp.tools.listed")
            if listed:
                for name, desc in _mcp_listed_names(listed):
                    catalog.setdefault(name, desc)
            for name, desc in _tools_list_output(span):
                catalog.setdefault(name, desc)

    # 6. Names seen only as calls still enter the catalog (empty description),
    # so the negative pool always covers tools the agent actually invoked.
    for trace in traces:
        for span in trace.spans:
            if classify(span) != "tool":
                continue
            name = extract_tool_name(span)
            if name:
                catalog.setdefault(name, "")
    return dict(sorted(catalog.items()))


def merge_catalog(base: dict[str, str], user: dict[str, str]) -> dict[str, str]:
    """Merge *user* catalog on top of *base*; user descriptions win."""
    merged = dict(base)
    for name, desc in user.items():
        merged[name] = desc
    return dict(sorted(merged.items()))


def load_catalog_file(path: str) -> dict[str, str]:
    """Load a ``{tool_name: description}`` JSON fixture (e.g. catalog.json)."""
    with open(path) as f:
        data = json.load(f)
    return {str(k): str(v) for k, v in data.items()}