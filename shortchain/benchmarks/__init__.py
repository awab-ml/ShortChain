"""Benchmark adapters — pluggable dataset integrations.

Each benchmark (ToolBench, APIBank, Gorilla, …) implements the
``BenchmarkAdapter`` protocol and registers here.  The generic
``run_benchmark.py`` script resolves adapters by name.
"""

from __future__ import annotations

import importlib
from typing import Any

from shortchain.benchmarks.adapter import BenchmarkAdapter
from shortchain.config import ShortChainConfig

# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------
# Values are "module_path:ClassName" strings — lazy-imported on demand so
# that heavyweight benchmark dependencies aren't loaded at import time.

ADAPTERS: dict[str, str] = {
    "toolbench": "shortchain.benchmarks.toolbench:ToolBenchAdapter",
    # Future:
    # "apibank":   "shortchain.benchmarks.apibank:APIBankAdapter",
    # "gorilla":   "shortchain.benchmarks.gorilla:GorillaAdapter",
    # "webarena":  "shortchain.benchmarks.webarena:WebArenaAdapter",
}


def list_adapters() -> list[str]:
    """Return the names of all registered benchmark adapters."""
    return sorted(ADAPTERS)


def create_adapter(name: str, config: ShortChainConfig, **kwargs: Any) -> BenchmarkAdapter:
    """Create a benchmark adapter by name.

    Parameters
    ----------
    name
        Registered adapter name (e.g., ``"toolbench"``).
    config
        Root ShortChain configuration.  The adapter receives the relevant
        sub-configs it needs.
    **kwargs
        Additional keyword arguments forwarded to the adapter constructor
        (e.g., ``train_path``, ``eval_path``).

    Returns
    -------
    BenchmarkAdapter
        An initialised adapter instance.

    Raises
    ------
    ValueError
        If *name* is not a registered adapter.
    ImportError
        If the adapter module cannot be imported.
    """
    name = name.lower().strip()
    if name not in ADAPTERS:
        available = ", ".join(list_adapters())
        raise ValueError(
            f"Unknown benchmark adapter: {name!r}. "
            f"Available adapters: {available}"
        )

    module_path, class_name = ADAPTERS[name].rsplit(":", 1)
    module = importlib.import_module(module_path)
    adapter_cls = getattr(module, class_name)

    # Pass relevant config slices to the adapter
    return adapter_cls(
        benchmark_config=config.benchmark,
        ingest_config=config.ingest,
        **kwargs,
    )


__all__ = [
    "BenchmarkAdapter",
    "ADAPTERS",
    "create_adapter",
    "list_adapters",
]
