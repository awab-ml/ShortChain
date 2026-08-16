"""Association properties context + span injection (PR 6).

Implements the merge-not-replace association contract from ``ingest-fix-plan``
(K7 / K13):

- ``association_values()`` — the merged association dict held in a
  ``ContextVar``.
- ``set_association`` — merges ``{**current, **props}``, never replaces,
  and writes the new keys onto the current span if it is recording.
- ``AssociationInjectionSpanProcessor`` — copies the merged dict onto every
  *future* child span at ``on_start`` (``traceloop.association.properties.*``
  + ``shortchain.*`` aliases) so the projector's priority-2 rule can read
  association from any span, not just the task root.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan, Span
from opentelemetry.sdk.trace.export import SpanProcessor

_ASSOCIATION: ContextVar[dict[str, Any] | None] = ContextVar(
    "shortchain.association", default=None
)

_ROOT_SPAN_NAME = "shortchain.task"


def association_values() -> dict[str, Any]:
    """The current merged association dict (``{}`` when none)."""
    return dict(_ASSOCIATION.get() or {})


def set_association_values(props: dict[str, Any]) -> None:
    """Merge-not-replace the association context."""
    merged = {**(association_values()), **props}
    _ASSOCIATION.set(merged)


def reset_association() -> None:
    """Clear the association context (after ``end_task`` / ``set_success``)."""
    _ASSOCIATION.set(None)


def write_association_on_span(span: Span, props: dict[str, Any]) -> None:
    """Write association props onto a recording span (both key namespaces)."""
    for key, value in props.items():
        span.set_attribute(f"traceloop.association.properties.{key}", value)
        span.set_attribute(f"shortchain.{key}", value)


class AssociationInjectionSpanProcessor(SpanProcessor):
    """Inject the merged association dict onto every new child span."""

    def on_start(self, span: Span, parent_context) -> None:
        props = association_values()
        if not props:
            return
        # The task root carries its own association attrs; children need the
        # copy so priority-2 extraction ("same keys on any other span") works.
        if span.name == _ROOT_SPAN_NAME:
            return
        write_association_on_span(span, props)

    def on_end(self, span: ReadableSpan) -> None:
        pass

    def shutdown(self) -> None:  # pragma: no cover - trivial
        pass

    def force_flush(self, timeout_millis: int = 30_000) -> bool:  # pragma: no cover
        return True