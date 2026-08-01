#!/usr/bin/env python3
"""Download and convert nvidia/Open-SWE-Traces to ShortChain JSONL format.

Downloads parquet shards directly via ``huggingface_hub`` and converts each
trajectory's conversation messages into ShortChain's ``Trajectory`` schema
(task_id, intent, spans, success, app_name).

Usage::

    python scripts/fetch_hf_traces.py \\
        --output data/open_swe/ \\
        --limit 5000 \\
        --success-only

Requires the ``datasets`` optional dependency group::

    uv pip install -e ".[datasets]"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from shortchain.utils.io import ensure_dir
from shortchain.utils.logging import get_logger

log = get_logger(__name__)

# HuggingFace dataset identifier
HF_DATASET = "nvidia/Open-SWE-Traces"


# ---------------------------------------------------------------------------
# Trajectory conversion
# ---------------------------------------------------------------------------

def _extract_tool_name(content: str) -> str | None:
    """Extract tool/function name from assistant content.

    SWE-agent and OpenHands use structured tool calls. Common patterns:
    - ``<function=tool_name>`` XML-style calls
    - ``bash`` / ``str_replace_editor`` / ``view_file`` etc.
    - Function call JSON blocks with ``"name": "tool_name"``
    """
    if not content:
        return None

    # Pattern 1: <function=tool_name> XML-style
    match = re.search(r"<function=(\w+)>", content)
    if match:
        return match.group(1)

    # Pattern 2: OpenHands-style tool_call blocks
    match = re.search(r'"name"\s*:\s*"(\w+)"', content)
    if match:
        return match.group(1)

    return None


def _extract_tool_name_from_call(tool_calls: list[dict[str, Any]] | None) -> str | None:
    """Extract tool name from structured tool_calls list."""
    if not tool_calls:
        return None
    for call in tool_calls:
        if isinstance(call, dict):
            # Standard OpenAI format: {"function": {"name": "..."}}
            func = call.get("function", {})
            if isinstance(func, dict) and "name" in func:
                return func["name"]
            # Direct name field
            if "name" in call:
                return call["name"]
    return None


def convert_trajectory(record: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a single Open-SWE-Traces record to ShortChain JSONL format.

    Parameters
    ----------
    record
        A row from the HuggingFace dataset with fields: instance_id, repo,
        trajectory, tools, resolved, metadata.

    Returns
    -------
    dict or None
        ShortChain-compatible trajectory dict, or None if conversion fails.
    """
    trajectory_messages = record.get("trajectory")
    if trajectory_messages is None:
        return None

    # Handle numpy / list / dict array conversions if needed
    if hasattr(trajectory_messages, "tolist"):
        trajectory_messages = trajectory_messages.tolist()

    if not isinstance(trajectory_messages, (list, tuple)) or not trajectory_messages:
        return None

    # Extract intent from the first user message
    intent = ""
    for msg in trajectory_messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in content
                )
            intent = str(content)[:2000]
            break

    if not intent:
        return None

    # Build spans from conversation turns
    spans: list[dict[str, Any]] = []
    pending_thoughts: str | None = None

    for msg in trajectory_messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")

        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in content
            )

        if role == "assistant":
            tool_calls = msg.get("tool_calls")
            if hasattr(tool_calls, "tolist"):
                tool_calls = tool_calls.tolist()
            tool_name = _extract_tool_name_from_call(tool_calls)

            if tool_name:
                spans.append({
                    "agent_name": "swe-agent",
                    "action": tool_name,
                    "thoughts": str(content)[:1000] if content else None,
                    "observation": None,
                })
            else:
                extracted = _extract_tool_name(str(content) if content else "")
                if extracted:
                    spans.append({
                        "agent_name": "swe-agent",
                        "action": extracted,
                        "thoughts": str(content)[:1000] if content else None,
                        "observation": None,
                    })
                else:
                    pending_thoughts = str(content)[:1000] if content else None

        elif role == "tool":
            if spans:
                spans[-1]["observation"] = str(content)[:1000] if content else None
            else:
                spans.append({
                    "agent_name": "swe-agent",
                    "action": "unknown_tool",
                    "thoughts": pending_thoughts,
                    "observation": str(content)[:1000] if content else None,
                })
                pending_thoughts = None

    if not spans:
        return None

    resolved = record.get("resolved", -1)
    success = int(resolved) == 1

    repo = str(record.get("repo", "unknown"))

    available_tools: list[str] = []
    raw_tools = record.get("tools")
    if hasattr(raw_tools, "tolist"):
        raw_tools = raw_tools.tolist()
    if isinstance(raw_tools, (list, tuple)):
        for tool_def in raw_tools:
            if isinstance(tool_def, str):
                try:
                    parsed = json.loads(tool_def)
                    func_info = parsed.get("function", {})
                    if isinstance(func_info, dict) and "name" in func_info:
                        available_tools.append(func_info["name"])
                except (json.JSONDecodeError, TypeError):
                    continue
            elif isinstance(tool_def, dict):
                func_info = tool_def.get("function", {})
                if isinstance(func_info, dict) and "name" in func_info:
                    available_tools.append(func_info["name"])

    tools_used = list({s["action"] for s in spans if s.get("action")})

    meta = record.get("metadata", {})
    if not isinstance(meta, dict):
        meta = {}

    return {
        "task_id": str(record.get("instance_id", record.get("trajectory_id", ""))),
        "intent": intent,
        "app_name": repo,
        "success": success,
        "spans": spans,
        "tools_used": tools_used,
        "metadata": {
            "trajectory_id": str(record.get("trajectory_id", "")),
            "repo": repo,
            "language": str(record.get("language", "")),
            "license": str(record.get("license", "")),
            "category": str(meta.get("category", "")),
            "available_tools": available_tools,
            "n_available_tools": len(available_tools),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download nvidia/Open-SWE-Traces parquet files and convert to ShortChain JSONL."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/open_swe",
        help="Output directory for converted trajectories.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Maximum number of trajectories to convert (default: 5000).",
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        default=False,
        help="Only include resolved (successful) trajectories.",
    )
    parser.add_argument(
        "--repo-filter",
        type=str,
        default=None,
        help="Filter to a specific repo (e.g., 'django/django').",
    )
    args = parser.parse_args()

    output_dir = ensure_dir(args.output)
    output_file = output_dir / "trajectories.jsonl"

    log.info(f"[bold]Fetching parquet list from HF repository {HF_DATASET}[/bold]")
    api = HfApi()
    all_files = api.list_repo_files(HF_DATASET, repo_type="dataset")
    parquet_files = [f for f in all_files if f.endswith(".parquet")]

    log.info(f"Found [bold]{len(parquet_files)}[/bold] parquet files in repository")
    log.info(f"Target limit: {args.limit} trajectories, Success-only: {args.success_only}")

    converted = 0
    skipped = 0
    repos_seen: dict[str, int] = {}

    with open(output_file, "w") as f:
        for pfile in parquet_files:
            if converted >= args.limit:
                break

            log.info(f"  Downloading parquet shard: [bold]{pfile}[/bold]")
            local_path = hf_hub_download(HF_DATASET, pfile, repo_type="dataset")
            df = pd.read_parquet(local_path)

            for _, row in df.iterrows():
                if converted >= args.limit:
                    break

                record = row.to_dict()

                if args.repo_filter and record.get("repo") != args.repo_filter:
                    continue

                if args.success_only and int(record.get("resolved", -1)) != 1:
                    skipped += 1
                    continue

                traj = convert_trajectory(record)
                if traj is None:
                    skipped += 1
                    continue

                f.write(json.dumps(traj, default=str) + "\n")
                converted += 1

                repo = traj.get("app_name", "unknown")
                repos_seen[repo] = repos_seen.get(repo, 0) + 1

                if converted % 1000 == 0:
                    log.info(f"    Converted {converted}/{args.limit} trajectories...")

    log.info(f"\n[bold green]✓ Conversion complete[/bold green]")
    log.info(f"  Output: {output_file}")
    log.info(f"  Converted: {converted}")
    log.info(f"  Skipped: {skipped}")
    log.info(f"  Unique Repos: {len(repos_seen)}")

    top_repos = sorted(repos_seen.items(), key=lambda x: -x[1])[:10]
    log.info(f"  Top repos distribution:")
    for repo, count in top_repos:
        log.info(f"    {repo}: {count}")

    stats = {
        "dataset": HF_DATASET,
        "limit": args.limit,
        "converted": converted,
        "skipped": skipped,
        "success_only": args.success_only,
        "repo_filter": args.repo_filter,
        "repos": repos_seen,
    }
    stats_path = output_dir / "fetch_stats.json"
    with open(stats_path, "w") as sf:
        json.dump(stats, sf, indent=2, default=str)
    log.info(f"  Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
