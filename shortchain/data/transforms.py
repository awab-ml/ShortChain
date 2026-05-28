"""Reusable trajectory transforms.

These transforms operate on the generic ``Trajectory`` schema and are
independent of any specific benchmark.  Benchmark adapters decide *when*
to apply them; the implementation lives here so that every adapter can
share the same logic.
"""

from __future__ import annotations

from shortchain.ingest.schema import Step, Trajectory
from shortchain.utils.logging import get_logger

log = get_logger(__name__)


def expand_to_step_trajectories(trajectory: Trajectory) -> list[Trajectory]:
    """Expand a multi-step trajectory into per-step sub-trajectories.

    For a trajectory with steps ``[s0, s1, s2]``, this produces three
    sub-trajectories:

    - Step 0: steps=[s0],       tools_used={tool(s0)}
    - Step 1: steps=[s0, s1],   tools_used={tool(s1)}
    - Step 2: steps=[s0, s1, s2], tools_used={tool(s2)}

    Each sub-trajectory inherits the parent's ``task_id`` (suffixed with
    ``_step_N``), ``intent``, ``app_name``, and ``success`` flag.  The
    ``metadata`` is shallow-copied and enriched with:

    - ``step_index``: the 0-based index of the target step
    - ``total_steps``: total number of steps in the original trajectory
    - ``available_tools``: the original trajectory's full ``tools_used`` set
    - ``previous_tools``: tools used in steps *before* the target step

    This is the core "step-level" expansion described in the ShortChain
    paper for multi-step evaluation.  It is not ToolBench-specific — any
    benchmark with sequential tool calls can benefit from it.

    Parameters
    ----------
    trajectory
        A ``Trajectory`` with one or more steps.

    Returns
    -------
    list[Trajectory]
        One sub-trajectory per step that has a valid ``tool_name``.
        Steps without a parseable tool are silently skipped.
    """
    expanded: list[Trajectory] = []

    for idx, step in enumerate(trajectory.steps):
        tool = step.tool_name
        if tool is None:
            continue

        # Collect tools used before this step
        previous_tools: list[str] = []
        for prior in trajectory.steps[:idx]:
            if prior.tool_name:
                previous_tools.append(prior.tool_name)

        meta = {
            **trajectory.metadata,
            "step_index": idx,
            "total_steps": len(trajectory.steps),
            "available_tools": sorted(trajectory.tools_used),
            "previous_tools": previous_tools,
        }

        sub = Trajectory(
            task_id=f"{trajectory.task_id}_step_{idx}",
            intent=trajectory.intent,
            steps=trajectory.steps[: idx + 1],
            success=trajectory.success,
            app_name=trajectory.app_name,
            tools_used={tool},
            metadata=meta,
        )
        expanded.append(sub)

    return expanded
