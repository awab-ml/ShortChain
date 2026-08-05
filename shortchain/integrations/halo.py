"""HALO/OpenInference AppWorld trace ingestion for ShortChain.

Reads the inference.net HALO appworld trace export (one JSON line per span)
and converts each *task* into a ShortChain :class:`Trajectory`.

Important handling:
- Each trace session runs several tutorial tasks and ONE real task
  (``# Real Task Instruction``). Only the real task's tool calls are used;
  we rebuild the conversation from the last LLM span's ``llm.input_messages``
  and slice after the real-task instruction so tutorials never contaminate
  the label set.
- ``supervisor__`` tools are agent-control calls (complete_task / show_profile /
  show_account_passwords) and are excluded from labels and pools.
- Timestamps in this export are synthetic (all identical), so slicing must be
  done by message position, not time.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from shortchain.ingest.schema import Span, Trajectory

_CONTROL_APPS = {"supervisor"}
# Match the exact task header only; a loose "real task instruction" substring
# also occurs inside the profile preamble, which would slice at the wrong spot.
_REAL_TASK_MARKERS = (
    "# Real Task Instruction",
    "## Real Task Instruction",
    "Real Task Instruction:",
)
_TUTORIAL_MARKERS = ("# Tutorial Task Instruction", "tutorial task instruction")


# ---------------------------------------------------------------------------
# Low-level parsing helpers
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


def app_of(tool_name: str) -> str:
    """Return the app part of a ``<app>__<api>`` tool name."""
    if "__" in tool_name:
        return tool_name.split("__", 1)[0].strip()
    return tool_name.strip()


def is_control_tool(tool_name: str) -> bool:
    """Whether a tool belongs to the agent-control (supervisor) namespace."""
    return app_of(tool_name) in _CONTROL_APPS


def _iter_rows(path: str | Path) -> Iterable[dict[str, Any]]:
    """Yield span rows from a HALO JSONL (or parquet) file."""
    p = Path(path)
    if p.suffix == ".parquet":
        import pandas as pd  # local import to keep core dependency-light

        df = pd.read_parquet(p)
        for _, row in df.iterrows():
            yield {k: v for k, v in row.items()}
        return
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _extract_calls_and_obs(msgs: list[dict], start_idx: int):
    """Walk messages from ``start_idx`` and yield ``(name, args, observation)``.

    Assistant messages carry ``tool_calls``; the tool results that follow are
    used as the observation for each call (best-effort pairing).
    """
    calls: list[tuple[str, str]] = []
    obs_buffer: list[str] = []

    for msg in msgs[start_idx:]:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "assistant":
            for tc in _as_list(msg.get("tool_calls")):
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function")
                if isinstance(fn, dict):
                    name = str(fn.get("name") or "")
                    args = str(fn.get("arguments") or "")
                    if name:
                        calls.append((name, args))
        elif role in ("tool", "function"):
            obs_buffer.append(str(msg.get("content") or ""))

    # Naive but robust pairing: consume one observation per call in order.
    result: list[tuple[str, str, str]] = []
    obs_iter = iter(obs_buffer)
    for name, args in calls:
        obs = next(obs_iter, "")
        result.append((name, args, obs))
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _iter_traces(path: str | Path) -> Iterable[dict[str, list[dict]]]:
    """Group span rows by trace_id (preserving file order)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in _iter_rows(path):
        tid = row.get("trace_id")
        if not tid:
            continue
        groups[tid].append(row)
    for tid, rows in groups.items():
        yield tid, rows


def build_trajectory_from_rows(rows: list[dict]) -> Trajectory | None:
    """Convert one trace's span rows into a ShortChain Trajectory.

    Returns ``None`` when no usable tool calls are found for the real task.
    """
    llm_rows = [
        r for r in rows
        if _as_dict(r.get("attributes")).get("openinference.span.kind") == "LLM"
    ]

    slices = []  # list of (name, args, obs)
    intent = ""
    real_task_found = False
    source = "halo_appworld_gemini3flash"

    if llm_rows:
        # File order == execution order (timestamps are synthetic/equal here),
        # so the LAST LLM span holds the fullest session history (incl. any
        # tutorial context + the real task instruction).
        full_rows = llm_rows
        attrs = _as_dict(full_rows[-1].get("attributes"))
        full_msgs = _as_list(attrs.get("llm.input_messages")) or _as_list(attrs)
        start = _real_task_index(full_msgs)
        if start is not None:
            real_task_found = True
            real = full_msgs[start]
            intent = _clean_intent(str(real.get("content") or ""))
            slices = _extract_calls_and_obs(full_msgs, start)
    else:
        source = "halo_appworld_fallback_toolspans"

    if not slices:
        # Fallback: use exported non-control tool spans (no message slicing).
        for r in rows:
            a = _as_dict(r.get("attributes"))
            if a.get("openinference.span.kind") != "TOOL":
                continue
            name = str(a.get("tool.name") or "")
            if not name or is_control_tool(name):
                continue
            args = str(a.get("input.value") or "{}")
            obs = str(a.get("output.value") or "")
            slices.append((name, args, obs))

    # Exclude control tools; dedupe preserving first-seen order.
    seen: set[str] = set()
    spans: list[Span] = []
    for name, _args, obs in slices:
        if is_control_tool(name) or name in seen:
            continue
        seen.add(name)
        spans.append(
            Span(
                action=name,
                thoughts="",
                observation=obs[:2000],
                agent_name="gemini-3-flash",
            )
        )

    if not spans:
        return None

    apps = Counter(app_of(s.action) for s in spans)
    app_name = apps.most_common(1)[0][0] if apps else ""

    # Soft success proxy: the exported TOOL span stream ends with complete_task.
    export_tools = [
        _as_dict(r.get("attributes")).get("tool.name")
        for r in rows
        if _as_dict(r.get("attributes")).get("openinference.span.kind") == "TOOL"
        and _as_dict(r.get("attributes")).get("tool.name")
    ]
    success = bool(export_tools) and str(export_tools[-1]) == "supervisor__complete_task"

    return Trajectory(
        task_id=str(rows[0].get("trace_id")),
        intent=intent,
        spans=spans,
        success=success,
        app_name=app_name,
        metadata={
            "source": source,
            "real_task_found": real_task_found,
            "apps": sorted(apps),
            "n_calls_total": len(slices),
            "n_unique_tools": len(spans),
        },
    )


def _real_task_index(msgs: list[Any]) -> int | None:
    """Index of the real-task user message (or last user message fallback)."""
    if not msgs:
        return None
    last_user = None
    for i, m in enumerate(msgs):
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        content = str(m.get("content") or "")
        for marker in _REAL_TASK_MARKERS:
            if marker.lower() in content.lower():
                return i
        if content.strip():
            last_user = i
    return last_user


def _clean_intent(content: str) -> str:
    for marker in _REAL_TASK_MARKERS:
        if content.startswith(marker):
            content = content[len(marker):]
            break
    for marker in _TUTORIAL_MARKERS:
        if content.startswith(marker):
            content = content[len(marker):]
            break
    return content.strip()


def load_appworld_traces(
    path: str | Path,
    success_only: bool = False,
) -> list[Trajectory]:
    """Load AppWorld trajectories from a HALO span export.

    Parameters
    ----------
    path
        ``.jsonl`` (or ``.parquet``) HALO export.
    success_only
        If ``True``, keep only traces whose exported span stream ends with
        ``supervisor__complete_task`` (soft success proxy; AppWorld's oracle
        score is not in this export).
    """
    trajs: list[Trajectory] = []
    for _, rows in _iter_traces(path):
        traj = build_trajectory_from_rows(rows)
        if traj is None:
            continue
        if success_only and not traj.success:
            continue
        trajs.append(traj)
    return trajs


def reconstruct_catalog(path: str | Path) -> dict[str, str]:
    """Reconstruct the tool catalog ``{tool_name: description}``.

    Only tool *names* are recoverable from this export (``mcp.tools.listed``,
    ``agent.tools`` and called tool names); descriptions are empty until the
    AppWorld API specification is supplied (see P2/P3).
    """
    names: set[str] = set()
    for _, rows in _iter_traces(path):
        for r in rows:
            a = _as_dict(r.get("attributes"))
            for field in ("mcp.tools.listed", "agent.tools"):
                for t in _as_list(a.get(field)):
                    if isinstance(t, str):
                        names.add(t)
            tn = a.get("tool.name")
            if isinstance(tn, str):
                names.add(tn)
    names = {n for n in names if n and not is_control_tool(n)}
    return {name: "" for name in sorted(names)}


def catalog_app_index(catalog: dict[str, str]) -> dict[str, list[str]]:
    """Return ``{app: [tool_name, ...]}`` for an app-scoped candidate index."""
    index: dict[str, list[str]] = defaultdict(list)
    for name in catalog:
        index[app_of(name)].append(name)
    return {app: sorted(tools) for app, tools in index.items()}
