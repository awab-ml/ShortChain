"""ShortChain — an observability backend for LLM applications that learns and adapts.

ShortChain sits on OpenTelemetry / OpenLLMetry: it collects execution traces,
learns the execution patterns they reveal, and adapts how the agent selects
tools — so repeated LLM decision calls shrink over time.

Core pipeline
-------------
- **Collect** — the SDK emits standard OTLP traces; a receiver assembles them.
- **Ingest** — traces normalize to a canonical ``Trajectory`` / ``Span`` schema.
- **Learn** — successful traces become a compact classifier of "which tool,
  given this context" (a pointwise ``(context, tool, label)`` problem).
- **Adapt** — at each decision the backend returns a ranked tool shortlist in
  ~1 ms, optionally hybridized with an LLM fallback when confidence is low.

The pipeline is agent-agnostic: it binds only to execution traces and typed
tool schemas, so the same core serves many source bindings (see ``adapters/``).
"""

__version__ = "0.1.0"
