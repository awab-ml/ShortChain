# ShortChain

An observability backend for LLM applications that learns and adapts.

ShortChain sits on [OpenTelemetry](https://opentelemetry.io/) / [OpenLLMetry](https://github.com/OpenLLMetry/opentelemetry-openllmetry): it collects execution traces from your agentic system, learns the execution patterns already present in them, and adapts how that system chooses tools — so you can cut operating cost and latency without replacing your agent framework.

It is not just a trace store. Repeated LLM tool-selection calls shrink over time.

## Why

- **Collect** — instrument the agent with the ShortChain SDK; OpenLLMetry emits standard OTLP traces; ShortChain receives and assembles them.
- **Learn** — successful traces become a compact classifier of *"which tool, given this context"*.
- **Adapt** — at each decision the backend returns a ranked shortlist in ~1 ms (full replace, or hybrid with LLM fallback), so repeated LLM tool-selection calls shrink over time.

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

## Quick start

### 1. Install

```bash
pip install -e ".[dev]"          # core
pip install -e ".[sdk,receiver]" # telemetry collection
```

### 2. Instrument with `ShortChain.init`

```python
from shortchain.sdk import ShortChain

ShortChain.init(
    api_key="your-receiver-key",
    app_name="support-agent",
    endpoint="http://127.0.0.1:4318",
)
```

OpenLLMetry instrumentations are enabled automatically for your existing
LangChain / OpenAI / CrewAI / MCP code.

### 3. Run the receiver

```bash
shortchain receive --config configs/runtime.yaml
```

The receiver assembles live OTLP traces and writes them as projected
trajectories under `data/runtime/` (mode `0600`).

### 4. Train from collected traces

```bash
shortchain dataset --trajectories data/runtime/trajectories.jsonl \
    --catalog data/runtime/catalog.json --output data/datasets/runtime
shortchain train --dataset data/datasets/runtime --output models/shortchain.pkl
```

### 5. Adapt at the decision point

```python
from shortchain.model import InferenceEngine

engine = InferenceEngine(model_path="models/shortchain.pkl", top_k=5)

shortlist = engine.predict(
    context={"intent": "Refund order 9921", "app_name": "support-agent"},
    candidates=tool_catalog,  # [{tool_name, tool_description}, ...]
    top_k=5,
)
# [("refund_order", 0.94), ("lookup_order", 0.72), ...]  — in ~1ms
```

## Example (offline, no agent required)

The repo ships sample trajectories under `examples/traces/`:

```bash
python -m shortchain dataset --trajectories examples/traces \
    --config examples/configs/example.yaml --output /tmp/sc-ds
python -m shortchain train --dataset /tmp/sc-ds --output /tmp/sc-model.pkl
python -m shortchain evaluate --model /tmp/sc-model.pkl --dataset /tmp/sc-ds/test.csv
```

See `examples/README.md` for collect / train / adapt demos.

## Architecture

ShortChain is a linear pipeline of operation-named modules:

```
shortchain/
├── telemetry/    # SDK, instrumentors, OTLP receiver, assembler
├── ingest/       # Trajectory / Span schema, loaders, OTEL projection
├── features/     # context / tool / corpus-stat encoders
├── dataset/      # pointwise (context, tool, label) construction
├── model/        # classifier, trainer, inference engine
├── evaluation/   # ranking metrics, calibration, hybrid fallback
└── adapters/     # optional source / benchmark bindings
```

- [docs/architecture.md](docs/architecture.md) — modules and data flow
- [docs/concepts.md](docs/concepts.md) — traces, patterns, pointwise learning, adapt
- [docs/integration.md](docs/integration.md) — SDK and the three adaptation modes

## Documentation

| Doc | What it covers |
| --- | --- |
| [Overview](docs/overview.md) | What ShortChain is and the problem it solves |
| [Getting Started](docs/getting-started.md) | Install, collect, train, adapt |
| [Architecture](docs/architecture.md) | Modules, data flow, design decisions |
| [Configuration](docs/configuration.md) | YAML reference |
| [API Reference](docs/api-reference.md) | Public classes and functions |
| [Integration](docs/integration.md) | SDK + replace / shortlist / hybrid modes |
| [Concepts](docs/concepts.md) | The four core ideas in one page |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Tests mirror the package layout, so
`pytest tests/ingest/` runs the ingest suite and `pytest tests/` runs
everything. Ruff must stay clean.

## License

[MIT](LICENSE). Security issues: see [SECURITY.md](SECURITY.md).