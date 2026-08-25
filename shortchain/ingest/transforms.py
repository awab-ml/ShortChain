"""Reusable trajectory transforms.

These transforms operate on the generic ``Trajectory`` schema and are
independent of any specific benchmark.  Benchmark adapters decide *when*
to apply them; the implementation lives here so that every adapter can
share the same logic.
"""

from __future__ import annotations

from shortchain.ingest.schema import Trajectory
from shortchain.utils.logging import get_logger

log = get_logger(__name__)


def expand_to_span_trajectories(trajectory: Trajectory) -> list[Trajectory]:
    """Expand a multi-span trajectory into per-span sub-trajectories.

    For a trajectory with spans ``[s0, s1, s2]``, this produces three
    sub-trajectories:

    - Span 0: spans=[s0],       tools_used={tool(s0)}
    - Span 1: spans=[s0, s1],   tools_used={tool(s1)}
    - Span 2: spans=[s0, s1, s2], tools_used={tool(s2)}

    Each sub-trajectory inherits the parent's ``task_id`` (suffixed with
    ``_span_N``), ``intent``, ``app_name``, and ``success`` flag.  The
    ``metadata`` is shallow-copied and enriched with:

    - ``span_index``: the 0-based index of the target span
    - ``total_spans``: total number of spans in the original trajectory
    - ``available_tools``: the original trajectory's full ``tools_used`` set
    - ``previous_tools``: tools used in spans *before* the target span

    This is the core "span-level" expansion for multi-span evaluation. It is
    not benchmark-specific — any workload with sequential tool calls can
    benefit from it.

    Parameters
    ----------
    trajectory
        A ``Trajectory`` with one or more spans.

    Returns
    -------
    list[Trajectory]
        One sub-trajectory per span that has a valid ``tool_name``.
        Spans without a parseable tool are silently skipped.
    """
    expanded: list[Trajectory] = []

    for idx, span in enumerate(trajectory.spans):
        tool = span.tool_name
        if tool is None:
            continue

        # Collect tools used before this span
        previous_tools: list[str] = []
        for prior in trajectory.spans[:idx]:
            if prior.tool_name:
                previous_tools.append(prior.tool_name)

        meta = {
            **trajectory.metadata,
            "span_index": idx,
            "total_spans": len(trajectory.spans),
            "available_tools": sorted(trajectory.tools_used),
            "previous_tools": previous_tools,
        }

        sub = Trajectory(
            task_id=f"{trajectory.task_id}_span_{idx}",
            intent=trajectory.intent,
            spans=trajectory.spans[: idx + 1],
            success=trajectory.success,
            app_name=trajectory.app_name,
            tools_used={tool},
            metadata=meta,
        )
        expanded.append(sub)

    return expanded
