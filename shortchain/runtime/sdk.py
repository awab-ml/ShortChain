"""ShortChain SDK — production collection entry point (PR 5).

``ShortChain.init`` sets up our own ``TracerProvider`` (see
``shortchain/runtime/instrument.py``), enables installed OpenLLMetry
instrumentations, and exports OTLP to the training receiver. The API is
ours (K5): we never wrap ``Traceloop.init`` — no Traceloop branding, no
default ``api.traceloop.com``, no hard dependency on every instrumentation.

PR 6 adds the load-bearing task-root span (``set_task`` / ``end_task`` /
``set_success``) so traces carry the success signal on the same
``trace_id``. Until then ``init`` alone is enough to *collect* OTEL spans.

Env vars (SDK-owned):
- ``SHORTCHAIN_API_KEY``, ``SHORTCHAIN_ENDPOINT`` (``file://`` allowed),
  ``SHORTCHAIN_DISPLAY_ENDPOINT``, ``SHORTCHAIN_APP_NAME``,
  ``SHORTCHAIN_TRACING_ENABLED``, ``SHORTCHAIN_TRACE_CONTENT``.
"""

from __future__ import annotations

import atexit
import os
from collections.abc import Callable

from opentelemetry.sdk.trace import ReadableSpan

from shortchain.runtime.instrument import (
    attach_exporters,
    enable_instrumentors,
    setup_tracer_provider,
)
from shortchain.utils.logging import get_logger

log = get_logger(__name__)


class ShortChain:
    """Public SDK surface (one-line init + task-scoped helpers)."""

    _initialized = False

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    @staticmethod
    def init(
        *,
        api_key: str | None = None,
        app_name: str = "shortchain-app",
        endpoint: str = "http://127.0.0.1:4318",
        display_endpoint: str | None = None,
        enabled: bool = True,
        disable_batch: bool = False,
        instruments: set[str] | None = None,   # None = auto-detect installed
        block_instruments: set[str] | None = None,
        resource_attributes: dict | None = None,
        headers: dict[str, str] | None = None,
        content_tracing: bool = True,          # default ON (K12)
        span_postprocess: Callable[[ReadableSpan], None] | None = None,
    ) -> None:
        """Enable OpenLLMetry instrumentations and export OTLP traces.

        Parameters
        ----------
        api_key
            Bearer token for the training receiver (env
            ``SHORTCHAIN_API_KEY`` if omitted).
        app_name
            Resource ``service.name`` (env ``SHORTCHAIN_APP_NAME``).
        endpoint
            OTLP endpoint base; ``/v1/traces`` appended if missing.
            ``file://path`` enables a local JSONL span dump (dev only).
        display_endpoint
            Optional second exporter for a display backend (K8 hook).
        enabled
            Kill switch (env ``SHORTCHAIN_TRACING_ENABLED``).
        instruments
            Whitelist of instrumentor short names; ``None`` auto-detects.
        block_instruments
            Blacklist; always wins over ``instruments``.
        content_tracing
            Send prompts / tool calls (needed for intent+observation);
            also mirrors ``TRACELOOP_TRACE_CONTENT`` (K12).
        span_postprocess
            Called on each ended span (redaction hook; may mutate
            attributes — safe to ship now, implement later).
        """
        if not _env_enabled(enabled):
            log.warning("ShortChain.init: tracing disabled (enabled=False / env)")
            return

        api_key = api_key or os.environ.get("SHORTCHAIN_API_KEY")
        app_name = os.environ.get("SHORTCHAIN_APP_NAME", app_name)
        endpoint = os.environ.get("SHORTCHAIN_ENDPOINT", endpoint)
        display_endpoint = display_endpoint or os.environ.get("SHORTCHAIN_DISPLAY_ENDPOINT")
        content_tracing = _env_bool("SHORTCHAIN_TRACE_CONTENT", content_tracing)

        if not api_key and not _is_local_endpoint(endpoint):
            log.warning(
                "ShortChain.init: no API key and endpoint is not loopback — "
                "the receiver will reject unauthenticated spans."
            )

        provider = setup_tracer_provider(
            app_name=app_name,
            resource_attributes=resource_attributes,
        )
        n_exporters = len(
            attach_exporters(
                provider,
                endpoint=endpoint,
                api_key=api_key,
                display_endpoint=display_endpoint,
                disable_batch=disable_batch,
                span_postprocess=span_postprocess,
                headers=headers,
            )
        )
        enabled_instruments, skipped = enable_instrumentors(
            provider,
            instruments=instruments,
            block_instruments=block_instruments,
            content_tracing=content_tracing,
        )
        log.info(
            f"ShortChain.init: endpoint={endpoint} app_name={app_name} "
            f"exporters={n_exporters} instruments={sorted(enabled_instruments)}"
            + (f" skipped={sorted(skipped)}" if skipped else "")
        )
        ShortChain._initialized = True
        atexit.register(ShortChain.flush)

    @staticmethod
    def flush() -> None:
        """Flush all span processors (idempotent; atexit-registered)."""
        try:
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.trace import get_tracer_provider

            provider = get_tracer_provider()
            # Never build a new Resource here: at interpreter shutdown the
            # threading instrumentation can no longer schedule detectors.
            if isinstance(provider, TracerProvider):
                provider.force_flush()  # type: ignore[union-attr]
        except Exception:  # pragma: no cover
            log.exception("ShortChain.flush failed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no")


def _is_local_endpoint(endpoint: str) -> bool:
    return endpoint.startswith(("http://127.0.0.1", "http://localhost", "file://"))


def _env_enabled(enabled_flag: bool) -> bool:
    if not enabled_flag:
        return False
    return _env_bool("SHORTCHAIN_TRACING_ENABLED", True)