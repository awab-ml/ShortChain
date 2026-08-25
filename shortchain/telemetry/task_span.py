"""SDK-owned ``shortchain.task`` root span lifecycle.

Why a root span at all: OpenLLMetry instrumentors end their workflow/agent/
tool spans *before* ``agent.run()`` returns, so a post-run ``set_success``
has nothing to write on (Traceloop's ``set_association_properties`` would
start a NEW trace). The SDK therefore starts an INTERNAL root span,
attaches it as the current span, merges association into context, and
writes success onto that still-open span at ``end_task`` / ``set_success``,
then ends it. Children started meanwhile (via the attached context) share
the same ``trace_id``.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from opentelemetry import context, trace
from opentelemetry.trace import Span, SpanKind, set_span_in_context

from shortchain.telemetry.association import (
    reset_association,
    set_association_values,
    write_association_on_span,
)
from shortchain.utils.logging import get_logger

log = get_logger(__name__)

_ROOT_SPAN_NAME = "shortchain.task"

_TASK_HANDLE: ContextVar["TaskHandle | None"] = ContextVar(
    "shortchain.task_handle", default=None
)


@dataclass
class TaskHandle:
    """The open task root + the context token that keeps it current."""

    span: Span
    context_token: object
    task_id: str


def current_task() -> TaskHandle | None:
    """The open task handle, or ``None`` when no task is active."""
    return _TASK_HANDLE.get()


def open_task(
    task_id: str,
    *,
    intent: str | None = None,
    app_name: str | None = None,
    **association: str,
) -> TaskHandle:
    """Start (or replace) the SDK task-root span and merge association.

    A nested ``set_task`` ends the previous root WITHOUT success
    (``success_source=unknown`` will make the assembler drop that trace —
    intentional; we do not leak handles).
    """
    previous = current_task()
    if previous is not None:
        log.warning(
            "set_task called while task %s is still open: ending it "
            "(success unknown)",
            previous.task_id,
        )
        _end_handle(previous, success=None)

    tracer = trace.get_tracer(__name__)
    span = tracer.start_span(
        _ROOT_SPAN_NAME,
        kind=SpanKind.INTERNAL,
        attributes={
            "shortchain.task_root": True,
            "traceloop.span.kind": "workflow",
        },
    )
    props: dict[str, Any] = {"task_id": task_id}
    if intent is not None:
        props["intent"] = intent
    if app_name is not None:
        props["app_name"] = app_name
    props.update(association)

    # Task root carries the full association itself (priority-1 extraction).
    write_association_on_span(span, props)

    # Children spawned while this is current inherit the same trace_id.
    token = context.attach(set_span_in_context(span))
    handle = TaskHandle(span=span, context_token=token, task_id=task_id)
    _TASK_HANDLE.set(handle)

    # Merged association context for every FUTURE child (on_start injection).
    set_association_values(props)
    return handle


def end_task(success: bool | None = None) -> None:
    """Write optional success onto the still-open root, then end it.

    ``success=None`` still ends the root but does NOT write
    ``shortchain.success`` — the assembler then sees no association success
    (``success_unknown``) and drops the trace under default gates. That is
    intentional: it beats training on an unlabelled failure.
    """
    handle = current_task()
    if handle is None:
        log.warning(
            "end_task / set_success called with no active task root "
            "(set_success_without_task); ignoring"
        )
        return
    _end_handle(handle, success=success)


def set_success(success: bool) -> None:
    """Alias of ``end_task``: write success on the STILL-OPEN root and end it.

    Never starts a new span — a post-run new span would be a new trace_id,
    and the assembler would see tools + ``success_unknown`` next to a lone
    success span.
    """
    end_task(success=bool(success))


def set_association(**props: str) -> None:
    """Merge-not-replace association onto the open (or future) task root."""
    handle = current_task()
    if handle is None:
        log.warning("set_association called with no active task root; ignoring")
        return
    set_association_values(props)
    write_association_on_span(handle.span, props)


def _end_handle(handle: TaskHandle, success: bool | None) -> None:
    span = handle.span
    if success is not None:
        span.set_attribute("shortchain.success", success)
        span.set_attribute("traceloop.association.properties.success", success)
    span.set_attribute("shortchain.complete", True)
    span.end()
    context.detach(handle.context_token)
    _TASK_HANDLE.set(None)
    reset_association()