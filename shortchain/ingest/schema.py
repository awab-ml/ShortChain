"""Pydantic data models for trajectory data.

These models define the internal representation of agent execution traces.
External formats are mapped to these models via configurable field mappings
in the loader.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class Span(BaseModel):
    """A single span in an agent execution trajectory."""

    agent_name: str = ""
    action: str | None = None          # tool / API called
    observation: str | None = None     # result of the action
    thoughts: str | None = None        # reasoning trace / chain-of-thought
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def tool_name(self) -> str | None:
        """Extract the tool/API name from the action field.

        Handles both plain tool names and structured action strings
        like ``"tool_name(arg1, arg2)"``.
        """
        if not self.action:
            return None
        # Strip arguments if present: "send_email(to='x')" → "send_email"
        name = self.action.split("(")[0].strip()
        return name if name else None


class Trajectory(BaseModel):
    """A complete agent execution trajectory (trace) for one task."""

    task_id: str
    intent: str                                     # user's original goal
    spans: list[Span] = Field(default_factory=list)
    success: bool = True
    app_name: str = ""                              # application context
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Derived fields (populated by validator)
    tools_used: set[str] = Field(default_factory=set)

    @model_validator(mode="after")
    def _derive_tools_used(self) -> "Trajectory":
        """Automatically derive the set of tools used from spans."""
        if not self.tools_used:
            tools: set[str] = set()
            for span in self.spans:
                name = span.tool_name
                if name:
                    tools.add(name)
            self.tools_used = tools
        return self

    @property
    def n_spans(self) -> int:
        """Number of spans in this trajectory."""
        return len(self.spans)

    @property
    def tool_sequence(self) -> list[str]:
        """Ordered list of tools called (with duplicates)."""
        return [s.tool_name for s in self.spans if s.tool_name]

    @property
    def last_thought(self) -> str | None:
        """The reasoning trace from the last span, if available."""
        for span in reversed(self.spans):
            if span.thoughts:
                return span.thoughts
        return None

    def summary(self) -> dict[str, Any]:
        """Compact summary for logging / display."""
        return {
            "task_id": self.task_id,
            "intent": self.intent[:80] + ("..." if len(self.intent) > 80 else ""),
            "app": self.app_name,
            "n_spans": self.n_spans,
            "n_tools": len(self.tools_used),
            "success": self.success,
        }
