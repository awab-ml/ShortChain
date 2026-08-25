"""AppWorld API-spec loader (function_calling format) for ShortChain.

Reads the AppWorld ``data/api_docs/function_calling/*.json`` files produced by
``appworld download data`` and exposes, per canonical tool name, the
description and typed argument schema the agent would actually see in a real
deployment.

The canonical tool name in these files is already ``<app>__<api>`` (e.g.
``spotify__login``), which is exactly the identifier used in ShortChain
traces and catalogs, so no re-mapping is required.

These are *static, per-tool* metadata constants: nothing here is task- or
test-set specific, so using them as features cannot leak evaluation answers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


_APP_NAMES = ("amazon", "file_system", "gmail", "phone", "simple_note",
              "splitwise", "spotify", "todoist", "venmo")
_HELPER_APPS = {"supervisor", "api_docs"}


@dataclass(frozen=True)
class ParamSpec:
    """A single typed argument of an AppWorld API."""

    name: str
    type: str
    description: str = ""
    enum: tuple[str, ...] = ()
    items_type: str = ""


@dataclass(frozen=True)
class ToolSpec:
    """The function-calling definition of one AppWorld API (tool)."""

    app_name: str
    api_name: str
    description: str
    parameters: tuple[ParamSpec, ...] = field(default_factory=tuple)

    @property
    def n_params(self) -> int:
        return len(self.parameters)

    def candidate_text(self) -> str:
        """Candidate representation: name + description + argument hints."""
        args = "; ".join(
            f"{p.name} ({p.type}){': ' + p.description if p.description else ''}"
            for p in self.parameters
        )
        if args:
            return f"{self.api_name} | {self.description} | args: {args}"
        return f"{self.api_name} | {self.description}"


def _parse_param(name: str, spec: dict[str, Any]) -> ParamSpec:
    ptype = spec.get("type")
    items = spec.get("items")
    return ParamSpec(
        name=name,
        type=str(ptype) if ptype else "",
        description=str(spec.get("description") or ""),
        enum=tuple(str(e) for e in (spec.get("enum") or [])),
        items_type=str(items.get("type")) if isinstance(items, dict) else "",
    )


def _parse_app_file(path: Path) -> dict[str, ToolSpec]:
    app_name = path.stem
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = data if isinstance(data, list) else data.get("functions") or []
    specs: dict[str, ToolSpec] = {}
    for entry in entries:
        fn = entry.get("function") if isinstance(entry, dict) else entry
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name") or "")
        if not name:
            continue
        params = []
        properties = (fn.get("parameters") or {}).get("properties") or {}
        for pname, pval in properties.items():
            if isinstance(pval, dict):
                params.append(_parse_param(pname, pval))
        specs[name] = ToolSpec(
            app_name=app_name,
            api_name=name,
            description=str(fn.get("description") or ""),
            parameters=tuple(params),
        )
    return specs


def load_appworld_api_spec(fc_dir: str | Path) -> dict[str, ToolSpec]:
    """Load all tool specs from an ``api_docs/function_calling`` directory.

    Helper/control apps (``supervisor``, ``api_docs``) are excluded, matching
    the tool namespaces used by ShortChain training/eval.
    """
    fc = Path(fc_dir)
    if not fc.is_dir():
        raise FileNotFoundError(f"function_calling dir not found: {fc}")
    specs: dict[str, ToolSpec] = {}
    for p in sorted(fc.glob("*.json")):
        if p.stem in _HELPER_APPS:
            continue
        specs.update(_parse_app_file(p))
    return specs


def build_catalog_and_schemas(
    fc_dir: str | Path | None,
    tool_names: Iterable[str],
) -> tuple[dict[str, str], dict[str, ToolSpec]]:
    """Build ``(catalog, schemas)`` for the given tool names.

    ``catalog[tool]`` is the candidate text (used as ``tool_description``
    in training/eval and as the BM25/DSR document); ``schemas[tool]`` is the
    full typed spec used for schema features. Tools without an AppWorld
    spec keep an empty description and are omitted from ``schemas``.
    """
    names = set(tool_names)
    if not fc_dir:
        return {n: "" for n in sorted(names)}, {}

    all_specs = load_appworld_api_spec(fc_dir)
    catalog: dict[str, str] = {}
    schemas: dict[str, ToolSpec] = {}
    for name in sorted(names):
        spec = all_specs.get(name)
        if spec is not None:
            catalog[name] = spec.candidate_text()
            schemas[name] = spec
        else:
            catalog[name] = ""
    return catalog, schemas


def coverage_report(specs: dict[str, ToolSpec]) -> dict[str, Any]:
    """Summary stats over the loaded tool specs."""
    n_params = sum(s.n_params for s in specs.values())
    return {
        "n_tools": len(specs),
        "n_apis_total": len(specs),  # one spec per canonical tool
        "n_params_total": n_params,
        "apps": sorted({s.app_name for s in specs.values()}),
    }
