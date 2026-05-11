#!/usr/bin/env python3
"""Set up and validate ToolBench data for TabAgent benchmarking.

Usage::

    python scripts/setup_toolbench.py --data-dir data/toolbench
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tabagent.ingest.toolbench_catalog import ToolBenchCatalog
from tabagent.ingest.toolbench_loader import ToolBenchLoader
from tabagent.utils.logging import get_logger

log = get_logger(__name__)

DOWNLOAD_URL = "https://drive.google.com/drive/folders/1TysbSWYpP8EioFu9xPJtpbJZMLLmwAmL"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set up ToolBench data for TabAgent benchmarking."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/toolbench",
        help="Root directory for ToolBench data.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="G1",
        choices=["G1", "G2", "G3", "all"],
        help="Scenario to prepare (default: G1).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    log.info("[bold]ToolBench Setup for TabAgent[/bold]")
    log.info(f"Data directory: {data_dir}")
    log.info("")

    # ------------------------------------------------------------------
    # 1. Validate directory structure
    # ------------------------------------------------------------------
    log.info("[bold]Step 1:[/bold] Validating directory structure...")

    toolenv_dir = data_dir / "data" / "toolenv" / "tools"
    train_file = data_dir / "data" / "toolllama_G123_dfs_train.json"
    eval_file = data_dir / "data" / "toolllama_G123_dfs_eval.json"

    missing = []
    if not toolenv_dir.is_dir():
        missing.append(str(toolenv_dir))
    if not train_file.is_file():
        missing.append(str(train_file))
    if not eval_file.is_file():
        missing.append(str(eval_file))

    if missing:
        log.error("[bold red]Missing required files/directories:[/bold red]")
        for m in missing:
            log.error(f"  ✗ {m}")
        log.info("")
        log.info("[bold yellow]Download ToolBench data from:[/bold yellow]")
        log.info(f"  {DOWNLOAD_URL}")
        log.info("")
        log.info("Then unzip and place under your data directory:")
        log.info(f"  unzip data.zip -d {data_dir}")
        return

    log.info("[bold green]  ✓ All required files found[/bold green]")

    # ------------------------------------------------------------------
    # 2. Parse tool catalog
    # ------------------------------------------------------------------
    log.info("")
    log.info("[bold]Step 2:[/bold] Parsing tool environment catalog...")

    catalog = ToolBenchCatalog.from_toolenv(toolenv_dir)
    summary = catalog.summary()

    log.info(f"  APIs:       {summary['n_apis']:,}")
    log.info(f"  Tools:      {summary['n_tools']:,}")
    log.info(f"  Categories: {summary['n_categories']}")
    log.info("  Top categories:")
    for cat, count in summary["top_categories"][:5]:
        log.info(f"    {cat}: {count} APIs")

    # Cache catalog
    catalog_cache = data_dir / "catalog.json"
    with open(catalog_cache, "w") as f:
        json.dump(
            {"catalog": catalog.catalog, "category_map": catalog.category_map},
            f,
            indent=2,
        )
    log.info(f"  Cached catalog → {catalog_cache}")

    # ------------------------------------------------------------------
    # 3. Analyse preprocessed data
    # ------------------------------------------------------------------
    log.info("")
    log.info("[bold]Step 3:[/bold] Analysing preprocessed data...")

    loader = ToolBenchLoader(catalog=catalog, success_only=False)

    # Count instances
    with open(train_file) as f:
        train_data = json.load(f)
    with open(eval_file) as f:
        eval_data = json.load(f)

    log.info(f"  Train instances: {len(train_data):,}")
    log.info(f"  Eval instances:  {len(eval_data):,}")

    # ------------------------------------------------------------------
    # 4. Filter to scenario and save summary
    # ------------------------------------------------------------------
    log.info("")
    log.info(f"[bold]Step 4:[/bold] Loading {args.scenario} trajectories...")

    trajs = loader.load_with_filter(train_file, scenario=args.scenario)

    # App distribution
    from collections import Counter

    app_counts = Counter(t.app_name for t in trajs)
    log.info(f"  {args.scenario} trajectories: {len(trajs):,}")
    log.info(f"  Unique apps/categories: {len(app_counts)}")
    log.info("  Top apps:")
    for app, count in app_counts.most_common(5):
        log.info(f"    {app}: {count}")

    # Tool usage stats
    all_tools = set()
    for t in trajs:
        all_tools.update(t.tools_used)
    log.info(f"  Unique tools used: {len(all_tools)}")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    log.info("")
    log.info("[bold green]✓ ToolBench setup complete![/bold green]")
    log.info("")
    log.info("Next steps:")
    log.info("  python scripts/benchmark_toolbench.py --config configs/toolbench.yaml")


if __name__ == "__main__":
    main()
