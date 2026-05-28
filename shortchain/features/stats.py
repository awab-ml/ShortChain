"""Corpus-level statistics computed from training trajectories.

``CorpusStats`` is a typed, serialisable container that stores
precomputed statistics used by feature builders and negative samplers.
Computing these once and caching avoids redundant work.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from pydantic import BaseModel, Field


class CorpusStats(BaseModel):
    """Precomputed statistics from a trajectory corpus.

    Attributes
    ----------
    tool_frequency
        How many trajectories each tool appears in.
    co_occurrence
        ``co_occurrence[tool_a][tool_b]`` = number of trajectories
        where both tools appear together.
    app_tools
        ``app_tools[app_name]`` = set of tool names used in that app.
    app_tool_count
        ``app_tool_count[app_name]`` = number of distinct tools for that app.
    total_trajectories
        Number of trajectories the stats were computed from.
    """

    tool_frequency: dict[str, int] = Field(default_factory=dict)
    co_occurrence: dict[str, dict[str, int]] = Field(default_factory=dict)
    app_tools: dict[str, list[str]] = Field(default_factory=dict)
    app_tool_count: dict[str, int] = Field(default_factory=dict)
    total_trajectories: int = 0

    @classmethod
    def from_trajectories(cls, trajectories: list[Any]) -> "CorpusStats":
        """Compute corpus statistics from a list of Trajectory objects.

        Parameters
        ----------
        trajectories
            List of ``shortchain.ingest.schema.Trajectory`` objects.

        Returns
        -------
        CorpusStats
            Precomputed statistics ready for use by builders and samplers.
        """
        tool_freq: Counter[str] = Counter()
        co_occ: dict[str, Counter[str]] = defaultdict(Counter)
        app_tools_map: dict[str, set[str]] = defaultdict(set)

        for traj in trajectories:
            tools = list(traj.tools_used)
            for tool in tools:
                tool_freq[tool] += 1
                app_tools_map[traj.app_name].add(tool)

            # Pairwise co-occurrence within same trajectory
            for i, t1 in enumerate(tools):
                for t2 in tools[i + 1 :]:
                    co_occ[t1][t2] += 1
                    co_occ[t2][t1] += 1

        return cls(
            tool_frequency=dict(tool_freq),
            co_occurrence={k: dict(v) for k, v in co_occ.items()},
            app_tools={k: sorted(v) for k, v in app_tools_map.items()},
            app_tool_count={k: len(v) for k, v in app_tools_map.items()},
            total_trajectories=len(trajectories),
        )

    def get_same_app_tools(self, app_name: str) -> list[str]:
        """Return all tools associated with the given app."""
        return self.app_tools.get(app_name, [])

    def get_co_occurring_tools(self, tool_name: str) -> dict[str, int]:
        """Return tools that co-occur with the given tool and their counts."""
        return self.co_occurrence.get(tool_name, {})

    def get_tool_freq(self, tool_name: str) -> int:
        """Return the frequency of a tool in the corpus."""
        return self.tool_frequency.get(tool_name, 0)
