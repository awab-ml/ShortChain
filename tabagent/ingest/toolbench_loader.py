"""ToolBench preprocessed data loader.

Converts ToolBench's conversation-format preprocessed JSON files
(``toolllama_G123_dfs_train.json``) into TabAgent ``Trajectory`` objects.

Implements the ``TrajectoryLoader`` protocol.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from tabagent.ingest.schema import Step, Trajectory
from tabagent.ingest.toolbench_catalog import ToolBenchCatalog
from tabagent.utils.logging import get_logger

log = get_logger(__name__)

# Regex patterns for parsing assistant messages
_THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?=\nAction|\Z)", re.DOTALL)
_ACTION_RE = re.compile(r"Action:\s*(.+?)(?=\nAction Input|\Z)", re.DOTALL)
_ACTION_INPUT_RE = re.compile(r"Action Input:\s*(.+)", re.DOTALL)

# Pattern for detecting number of tools in system prompt
_TOOLS_LIST_RE = re.compile(r"You have access of the following tools:\n((?:\d+\..+\n?)+)")
_TOOL_ENTRY_RE = re.compile(r"\d+\.(\S+):")


class ToolBenchLoader:
    """Load ToolBench preprocessed data into TabAgent Trajectories.

    Parameters
    ----------
    catalog
        Optional ``ToolBenchCatalog`` for resolving action strings to
        canonical API keys and looking up categories.
    success_only
        If ``True``, skip trajectories that ended with ``give_up_and_restart``.
    """

    def __init__(
        self,
        catalog: ToolBenchCatalog | None = None,
        success_only: bool = True,
    ) -> None:
        self.catalog = catalog
        self.success_only = success_only

    # ------------------------------------------------------------------
    # Public API (TrajectoryLoader protocol)
    # ------------------------------------------------------------------

    def load(self, path: str | Path) -> list[Trajectory]:
        """Load trajectories from a preprocessed JSON file.

        Parameters
        ----------
        path
            Path to a ``toolllama_G123_dfs_*.json`` file.

        Returns
        -------
        list[Trajectory]
        """
        path = Path(path)
        data = _read_json(path)

        trajectories: list[Trajectory] = []
        skipped = 0

        for i, instance in enumerate(data):
            try:
                traj = self._parse_instance(instance, index=i)
                if self.success_only and not traj.success:
                    skipped += 1
                    continue
                trajectories.append(traj)
            except Exception as exc:
                log.warning(f"Skipping instance {i}: {exc}")
                skipped += 1

        log.info(
            f"Loaded [bold green]{len(trajectories)}[/bold green] trajectories "
            f"from {path.name} (skipped {skipped})"
        )
        return trajectories

    def load_with_filter(
        self,
        path: str | Path,
        scenario: str = "G1",
    ) -> list[Trajectory]:
        """Load and filter trajectories to a specific scenario.

        Scenario detection:
        - **G1** (single-tool): system prompt lists exactly 1 tool
        - **G2** (intra-category multi-tool): system prompt lists 2+ tools
        - **G3** (intra-collection multi-tool): system prompt lists 2+ tools
          from different categories

        For simplicity, G1 filtering checks that only 1 tool is listed
        in the system prompt (excluding the ``Finish`` pseudo-tool).

        Parameters
        ----------
        path
            Path to preprocessed JSON file.
        scenario
            ``"G1"``, ``"G2"``, or ``"G3"``.

        Returns
        -------
        list[Trajectory]
        """
        all_trajectories = self.load(path)

        if scenario == "G1":
            filtered = [t for t in all_trajectories if self._is_single_tool(t)]
        elif scenario == "G2":
            filtered = [t for t in all_trajectories if not self._is_single_tool(t)]
        else:
            # G3 or "all"
            filtered = all_trajectories

        log.info(
            f"Filtered to {scenario}: [bold]{len(filtered)}[/bold] / "
            f"{len(all_trajectories)} trajectories"
        )
        return filtered

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_instance(self, instance: dict[str, Any], index: int) -> Trajectory:
        """Convert one ToolBench conversation instance to a Trajectory."""
        conversations = instance.get("conversations", [])
        if not conversations:
            raise ValueError("Empty conversations list")

        # Extract components by role
        system_msg = ""
        user_msg = ""
        assistant_function_pairs: list[tuple[str, str]] = []

        current_assistant: str | None = None
        for conv in conversations:
            role = conv.get("from", "")
            value = conv.get("value", "")

            if role == "system":
                system_msg = value
            elif role == "user":
                user_msg = value
            elif role == "assistant":
                current_assistant = value
            elif role == "function":
                if current_assistant is not None:
                    assistant_function_pairs.append((current_assistant, value))
                    current_assistant = None

        # Handle trailing assistant message without function response
        if current_assistant is not None:
            assistant_function_pairs.append((current_assistant, ""))

        # Build task_id from index (ToolBench IDs are full instruction text)
        raw_id = instance.get("id", "")
        task_id = f"toolbench_{index}"

        # Extract intent from user message
        intent = _clean_user_message(user_msg)

        # Determine app_name from system prompt tools
        tool_names = _extract_tool_names_from_system(system_msg)
        app_name = self._resolve_app_name(tool_names)

        # Parse steps from assistant/function pairs
        steps: list[Step] = []
        tools_used: set[str] = set()

        for asst_msg, func_response in assistant_function_pairs:
            thought, action, action_input = parse_assistant_message(asst_msg)

            # Skip Finish actions — they're not real tool calls
            if action and action.strip() == "Finish":
                continue

            # Resolve action to catalog key
            resolved = action
            if action and self.catalog:
                resolved_key = self.catalog.resolve_action(action)
                if resolved_key:
                    resolved = resolved_key

            step = Step(
                agent_name="ToolBench",
                action=resolved,
                observation=func_response[:500] if func_response else None,
                thoughts=thought,
            )
            steps.append(step)
            if resolved:
                tools_used.add(resolved)

        # Determine success
        success = _determine_success(conversations)

        # Store the number of tools from system prompt in metadata
        metadata: dict[str, Any] = {
            "raw_id": raw_id[:200],
            "n_system_tools": len(tool_names),
            "system_tool_names": tool_names,
        }

        return Trajectory(
            task_id=task_id,
            intent=intent,
            steps=steps,
            success=success,
            app_name=app_name,
            tools_used=tools_used,
            metadata=metadata,
        )

    def _resolve_app_name(self, tool_names: list[str]) -> str:
        """Resolve app_name from tool names using the catalog's category map."""
        if not tool_names:
            return "unknown"

        if self.catalog:
            # Find category for first tool's APIs
            for tool_name in tool_names:
                tool_norm = tool_name.lower().replace(" ", "_")
                apis = self.catalog.tool_to_apis.get(tool_norm, [])
                if apis:
                    cat = self.catalog.get_category(apis[0])
                    if cat:
                        return cat

        # Fallback: use first tool name as app_name
        return tool_names[0] if tool_names else "unknown"

    @staticmethod
    def _is_single_tool(traj: Trajectory) -> bool:
        """Check if a trajectory used only a single tool (G1 scenario)."""
        n_tools = traj.metadata.get("n_system_tools", 0)
        return n_tools == 1


# ---------------------------------------------------------------------------
# Module-level parsing functions (exposed for testing)
# ---------------------------------------------------------------------------

def parse_assistant_message(value: str) -> tuple[str, str, str]:
    """Parse a ToolBench assistant message.

    Expected format::

        Thought: I need to do X.
        Action: some_api_for_some_tool
        Action Input: {"param": "value"}

    Parameters
    ----------
    value
        Raw assistant message text.

    Returns
    -------
    tuple[str, str, str]
        ``(thought, action, action_input)``
    """
    thought = ""
    action = ""
    action_input = ""

    thought_match = _THOUGHT_RE.search(value)
    if thought_match:
        thought = thought_match.group(1).strip()

    action_match = _ACTION_RE.search(value)
    if action_match:
        action = action_match.group(1).strip()

    input_match = _ACTION_INPUT_RE.search(value)
    if input_match:
        action_input = input_match.group(1).strip()

    return thought, action, action_input


def _determine_success(conversations: list[dict[str, Any]]) -> bool:
    """Determine if a ToolBench conversation ended successfully.

    Success is indicated by the final assistant ``Action: Finish`` with
    ``"return_type": "give_answer"`` in the Action Input.

    Failure is indicated by ``"return_type": "give_up_and_restart"``.
    """
    # Walk backward to find last assistant message
    for conv in reversed(conversations):
        if conv.get("from") == "assistant":
            value = conv.get("value", "")
            if "give_answer" in value:
                return True
            if "give_up_and_restart" in value or "give_up" in value:
                return False
            # If Finish not found, check for action pattern
            _, action, action_input = parse_assistant_message(value)
            if action == "Finish":
                if "give_answer" in action_input:
                    return True
                return False
            break

    # Default: assume success if no clear failure signal
    return True


def _extract_tool_names_from_system(system_msg: str) -> list[str]:
    """Extract tool names listed in the system prompt.

    The system prompt format is::

        You have access of the following tools:
        1.tool_name_a: description
        2.tool_name_b: description

    Returns
    -------
    list[str]
        Tool names (excluding "Finish").
    """
    tools: list[str] = []
    match = _TOOLS_LIST_RE.search(system_msg)
    if match:
        block = match.group(1)
        for entry_match in _TOOL_ENTRY_RE.finditer(block):
            name = entry_match.group(1).strip()
            if name.lower() != "finish":
                tools.append(name)
    return tools


def _clean_user_message(user_msg: str) -> str:
    """Clean the user message to extract the intent.

    Strips ``"Begin!"`` suffix and leading/trailing whitespace.
    """
    msg = user_msg.strip()
    # Remove "Begin!" instruction
    msg = re.sub(r"\s*Begin!\s*$", "", msg, flags=re.IGNORECASE)
    return msg.strip()


def _read_json(path: Path) -> list[dict[str, Any]]:
    """Read a JSON file and return a list of instances."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    return data
