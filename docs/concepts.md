# Concepts

Four ideas underpin ShortChain. Each is a short paragraph and a one-line
takeaway.

## Traces are the source of truth

OpenLLMetry spans — not hand-written JSONL — are the production input.
Instrumenting your agent with `ShortChain.init()` emits standard OTLP traces;
a receiver assembles them into complete executions. JSONL is the offline and
example path, so the pipeline is identical whether data arrives live or from
a dump.

> Instrument the agent, not the learning pipeline. The same learning code runs
> on live OTLP traces and on offline JSONL.

## Execution patterns

A successful trajectory is evidence: *given this intent and this observed
state, these tools followed in this order*. Across many tasks, those paths form
a distribution — which tools are used together, which tools belong to which
application, how often each tool appears. That distribution is learnable and
it is what the classifier captures.

> Successful traces are training data waiting to happen.

## Pointwise learning

The ranking problem — "which tool next?" — is reduced to a simple learning
problem. Each decision becomes `(context, candidate, label)`:

- **positive** rows for the tools the agent actually used, and
- **negative** rows sampled from the catalog for tools it did not.

A compact model scores `P(use this tool | context)` for every candidate and
ranks the catalog by that score. Because the model is small, scoring a full
catalog takes about a millisecond.

> Rank by probability, not by generation.

## Adapt, don't only display

The backend uses those scores to influence the agent at decision time — not
just to visualize past runs. Three modes:

- **Replace** — use the shortlist directly, no LLM call.
- **Shortlist** — narrow the catalog, let the LLM pick from a small set.
- **Hybrid** — use the classifier when calibrated confidence is high, and
  defer to the LLM when it is low.

That last mode is where the cost comes down: the agent stops paying for every
decision and only pays when the learned model is unsure.

> Confidence routing turns a trace store into a cost optimizer.