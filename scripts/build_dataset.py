#!/usr/bin/env python3
"""Build a training dataset from agent execution trajectories.

Usage::

    python scripts/build_dataset.py \\
        --trajectories data/example/ \\
        --output data/datasets/ \\
        --config configs/example.yaml

Runtime (OTEL) data::

    python scripts/build_dataset.py \\
        --trajectories data/runtime/trajectories.jsonl \\
        --catalog data/runtime/catalog.json \\
        --output data/datasets/runtime \\
        --config configs/runtime.yaml

Concept
-------
Converts trajectories into the pointwise ``(context, tool, label)`` rows the
classifier learns from: for each trajectory, every tool actually used becomes
a positive row and every negative is sampled from the catalog. Context and
tool features are composed by the builders (see ``features/``); the output
is ``train.csv``/``test.csv`` split at the task level (no task leaks across
the split).

``--catalog`` supplies ``{tool_name: description}`` (e.g. the receiver's
``data/runtime/catalog.json``); without it the catalog is derived from the
trajectories themselves.
"""

from __future__ import annotations

import argparse

from shortchain.config import load_config
from shortchain.dataset.builder import DatasetBuilder
from shortchain.dataset.splitter import GroupStratifiedSplitter
from shortchain.ingest.loader import load_trajectories
from shortchain.telemetry.catalog import load_catalog_file
from shortchain.utils.io import ensure_dir
from shortchain.utils.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a training dataset from agent trajectories."
    )
    parser.add_argument(
        "--trajectories",
        type=str,
        required=True,
        help="Path to trajectory files (directory or single file).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/datasets",
        help="Output directory for the built dataset.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file (overrides defaults).",
    )
    parser.add_argument(
        "--catalog",
        type=str,
        default=None,
        help="Path to a tool catalog JSON ({tool_name: description}); "
        "runtime writes this to data/runtime/catalog.json. User entries "
        "win over any discovered during loading.",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Don't create train/test split; output a single dataset.",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # 1. Ingest trajectories
    log.info(f"[bold]Span 1:[/bold] Loading trajectories from {args.trajectories}")
    trajectories = load_trajectories(args.trajectories, config=cfg.ingest)
    if not trajectories:
        log.error("No trajectories loaded. Check path and format.")
        return

    # 1b. Optional user tool catalog (runtime-projected trajectories usually
    # ship a catalog.json next to them — descriptions feed TF-IDF/BM25).
    tool_catalog: dict[str, str] | None = None
    if args.catalog:
        tool_catalog = load_catalog_file(args.catalog)
        log.info(
            f"[bold]Catalog:[/bold] loaded {len(tool_catalog)} tools "
            f"from {args.catalog}"
        )

    # 2. Build dataset
    log.info("[bold]Span 2:[/bold] Building (context, tool, label) pairs")
    builder = DatasetBuilder(config=cfg.dataset, tool_catalog=tool_catalog)
    df = builder.build(trajectories)

    # 3. Split
    output_dir = ensure_dir(args.output)

    if args.no_split:
        out_path = output_dir / "full_dataset.csv"
        df.to_csv(out_path, index=False)
        log.info(f"Saved full dataset to {out_path}")
    else:
        log.info("[bold]Span 3:[/bold] Creating train/test split")
        splitter = GroupStratifiedSplitter(cfg.splitter)
        train_df, test_df = splitter.train_test_split(df)

        train_path = output_dir / "train.csv"
        test_path = output_dir / "test.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        log.info(f"Saved train ({len(train_df)} rows) → {train_path}")
        log.info(f"Saved test  ({len(test_df)} rows)  → {test_path}")

    log.info("[bold green]✓ Dataset build complete[/bold green]")


if __name__ == "__main__":
    main()
