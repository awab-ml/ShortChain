"""CLI for the ShortChain telemetry receiver.

    python -m shortchain.telemetry receive [--config configs/runtime.yaml]

Starts the thin OTLP/HTTP receiver with the mandated ``--workers 1``
(uvicorn multi-worker would split one trace_id across assemblers — each
would flush a fragment and the LRU would drop the rest).
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

from shortchain.config import RuntimeConfig, load_config
from shortchain.telemetry.assembler import (
    JsonlTrajectoryWriter,
    RuntimeMetrics,
    TraceAssembler,
)
from shortchain.utils.logging import get_logger

log = get_logger(__name__)


def _config_from_args(args: argparse.Namespace) -> RuntimeConfig:
    root = load_config(args.config)
    return root.runtime


def _start_background_tick(assembler: TraceAssembler) -> threading.Thread:
    """1s assembly ticker; flush loop terminates the process on SIGTERM."""

    def tick_loop() -> None:
        while True:
            try:
                assembler.tick()
            except Exception as exc:  # pragma: no cover
                log.exception(f"assembler tick failed: {exc}")
            time.sleep(1.0)

    thread = threading.Thread(target=tick_loop, name="assembler-tick", daemon=True)
    thread.start()
    return thread


def _make_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shortchain.telemetry",
        description="ShortChain production collection runtime.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    receive = subparsers.add_parser(
        "receive",
        help="Run the thin OTLP/HTTP receiver (workers=1 enforced).",
    )
    receive.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config (configs/runtime.yaml).",
    )
    receive.add_argument(
        "--out",
        type=str,
        default=None,
        help="Override runtime.output (Trajectory JSONL).",
    )
    receive.add_argument(
        "--bind",
        type=str,
        default=None,
        help="Override runtime.bind (host:port).",
    )
    receive.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Uvicorn workers. Must be 1 (multi-worker splits trace_ids).",
    )
    receive.add_argument(
        "--idle-timeout",
        type=float,
        default=None,
        help="Override runtime.idle_timeout_s.",
    )
    return parser


def _warn_if_workers_gt_1(workers: int) -> None:
    if workers > 1:
        log.error(
            "workers>1 splits one trace_id across assemblers; each would "
            "flush a fragment and the LRU would drop the rest. Refusing."
        )
        sys.exit(2)


def _validate_env_workers() -> None:
    env_workers = os.environ.get("WEB_CONCURRENCY")
    if env_workers and env_workers.strip().isdigit() and int(env_workers) > 1:
        log.error("WEB_CONCURRENCY>1 refused: trace assembly requires workers=1.")
        sys.exit(2)


def main(argv: list[str] | None = None) -> None:
    args = _make_args_parser().parse_args(argv)
    if args.command == "receive":
        run_receive(args)
    else:  # pragma: no cover
        _make_args_parser().print_help()
        sys.exit(2)


def run_receive(args: argparse.Namespace) -> None:
    """Start the assembled receiver with uvicorn (workers=1)."""
    import uvicorn

    from shortchain.telemetry.receiver import create_receiver_app

    config = _config_from_args(args)
    if args.out:
        config = config.model_copy(update={"output": args.out})
    if args.bind:
        config = config.model_copy(update={"bind": args.bind})
    if args.idle_timeout is not None:
        config = config.model_copy(update={"idle_timeout_s": args.idle_timeout})

    _warn_if_workers_gt_1(args.workers or config.workers)
    _validate_env_workers()

    out_path = Path(config.output)
    metrics = RuntimeMetrics()
    assembler = TraceAssembler(
        config,
        writer=JsonlTrajectoryWriter(out_path),
        metrics=metrics,
    )
    app = create_receiver_app(assembler, config)

    _start_background_tick(assembler)

    host, _, port = config.bind.partition(":")
    log.info(
        f"receiver listening on http://{config.bind}/v1/traces "
        f"-> {config.output}"
    )
    uvicorn.run(app, host=host, port=int(port or 4318), workers=1, log_level="info")


if __name__ == "__main__":
    main()