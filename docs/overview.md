# Overview

ShortChain is an observability backend for LLM applications, built on
[OpenTelemetry](https://opentelemetry.io/) / [OpenLLMetry](https://github.com/OpenLLMetry/opentelemetry-openllmetry).

Most of an agent's operating cost and latency sits at a surprisingly simple
point: deciding *"which tool do I call next?"*. Each such call to a large
language model costs a few cents and adds hundreds of milliseconds. For agents
that make many decisions per task, these costs add up fast.

That decision is a **ranking** problem, not a generation problem. Given a fixed
tool catalog and the current execution context, the system mostly has to rank a
known set of candidates.

ShortChain treats the traces already flowing through your agent as the source
of truth for learning that ranking. It is not only a trace store:

- **Collect** — ShortChain.init() enables OpenLLMetry instrumentation and
  exports standard OTLP traces to a receiver that assembles them.
- **Learn** — successful trajectories become a compact classifier of "which
  tool, given this context".
- **Adapt** — at each decision the backend returns a ranked shortlist in ~1 ms
  (full replace, or hybrid with an LLM fallback when confidence is low), so
  repeated LLM tool-selection calls shrink over time.

## How it works

```
Live OTEL traces (SDK + OpenLLMetry)
        │
        ▼
Telemetry receiver (OTLP/HTTP, assembler, quality gate)
        │
        ▼
Canonical Trajectory / Span schema
        │
        ▼
Pointwise dataset  →  features  →  compact classifier  →  ranked tool shortlist (~1ms)
        │
        ▼
Optional hybrid: classifier when confident, LLM fallback when not
```

## When to use it

**Good fit:**

- Your agent has a mostly fixed or slowly changing tool catalog.
- You have execution traces (even a few dozen successful trajectories).
- Tool-selection decisions are repetitive across tasks.
- You want lower latency and cost at decision time.

**Not a good fit:**

- Your tool catalog changes every request.
- You have zero historical execution data (cold start — but see the cold-start
  guidance in the [integration guide](integration.md)).
- Tool selection requires open-ended reasoning that a fixed ranked catalog
  cannot capture.

## Representative numbers

On a standard tool-selection workload, ShortChain produces a ranked tool
shortlist in ~1 ms per decision and routes to the LLM only when its confidence
is low — yielding meaningful savings on both cost and latency while keeping
tool-selection accuracy on par with an LLM-only baseline. See
`examples/benchmarks/` for the validation harness that reproduces these
numbers on your own data.