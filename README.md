# ShortChain

> Replace expensive LLM decision components in agentic systems with compact tabular-textual classifiers.

**ShortChain** is an optimization layer that learns from successful agent execution traces to train lightweight classifiers (~1ms inference) that replace or augment expensive LLM tool-selection decisions.

## Why ShortChain?

| | LLM Tool Selection | ShortChain |
|---|---|---|
| **Latency** | 500–2000ms per decision | ~1ms per decision |
| **Cost** | $0.01–$0.10 per call | $0 (local inference) |
| **Accuracy** | Baseline | Comparable (R-Precision ≥ 0.85) |
| **Dependencies** | API key + network | NumPy + XGBoost (local) |

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Build dataset → Train → Evaluate (3 commands, example data)
python scripts/build_dataset.py --trajectories data/example/ --output data/datasets/
python scripts/train.py --dataset data/datasets/ --output models/shortchain.pkl
python scripts/evaluate.py --model models/shortchain.pkl --dataset data/datasets/test.csv
```

## Production: Collect Traces From Your Agent (SDK + OTEL)

Dump JSONL only for benchmarks / offline data. For production, ShortChain
enables the published OpenLLMetry instrumentations and projects live OTLP
traces onto the training schema server-side:

```python
# pip install "shortchain[sdk,receiver]"
import os
from shortchain.sdk import ShortChain

ShortChain.init(
    api_key=os.environ["SHORTCHAIN_API_KEY"],
    app_name="support-agent",
    endpoint=os.environ.get("SHORTCHAIN_ENDPOINT", "http://127.0.0.1:4318"),
)

def handle_request(req):
    ShortChain.set_task(task_id=req.id, intent=req.text)
    try:
        result = agent.run(req.text)
        ShortChain.end_task(success=bool(result.ok))
        return result
    except Exception:
        ShortChain.end_task(success=False)
        raise
```

Run the receiver, then train on its output:

```bash
python -m shortchain.runtime receive --config configs/runtime.yaml
python scripts/build_dataset.py \
    --trajectories data/runtime/trajectories.jsonl \
    --catalog data/runtime/catalog.json \
    --output data/datasets/runtime
python scripts/train.py --dataset data/datasets/runtime --output models/shortchain.pkl
```

## Use in Your Agent

```python
from shortchain.head.inference import InferenceEngine

engine = InferenceEngine(model_path="models/shortchain.pkl", top_k=5)

# At each agent decision point (~1ms):
shortlist = engine.predict(
    context={"intent": "Send an email to John", "app_name": "gmail", ...},
    candidates=[{"tool_name": "send_email", "tool_description": "..."}, ...],
)
# → [("send_email", 0.94), ("create_draft", 0.72), ...]
```

## Architecture

```
Trajectories → Ingestion → CorpusStats → DatasetBuilder → FeaturePipeline → Classifier → Inference
                   ↓             ↓              ↓                ↓               ↓            ↓
             OTEL/OTLP +  Frequencies     Pointwise pairs   TF-IDF or       XGBoost/RF    Top-K tools
             JSONL adapter Co-occurrence  (context, tool,   E5-small        with state-   ranked by
             (runtime      App-tool maps  label) + negs     encoding        aware feats   confidence
             receiver)
```

## Project Structure

```
ShortChain/
├── shortchain/                    # Core package (2,920 lines)
│   ├── config.py                # Pydantic config models (+ RuntimeConfig)
│   ├── ingest/                  # Source adapters: JSONL, OTEL→Trajectory projector, quality gate
│   ├── features/                # Feature pipeline (encoders, context, tool, stats)
│   ├── dataset/                 # Dataset construction & negative sampling
│   ├── head/                    # Classifier, trainer, inference engine
│   ├── evaluation/              # R-precision, Recall@k, F1, AUC
│   ├── runtime/                 # Production collection: SDK, OTLP receiver, assembler
│   ├── sdk.py                   # Public SDK façade (from shortchain.sdk import ShortChain)
│   └── utils/                   # I/O, logging
├── scripts/                     # CLI entry points
├── tests/                       # Test suite (~400 tests)
├── configs/                     # default.yaml, runtime.yaml, example.yaml
├── data/example/                # 15 example trajectories
└── docs/                        # Full documentation
```

## Key Features

- **Agent-agnostic** — works with any agent that produces JSON execution logs
- **Configurable field mapping** — map your log fields to ShortChain's schema via YAML
- **Modular feature pipeline** — context, tool, and encoding stages via `FeaturePipeline`
- **State-aware features** — span index, last action, history summary, tool diversity
- **Pluggable negative sampling** — random, hard (same-app, co-usage, similarity), or mixed
- **Hybrid text encoding** — TF-IDF (default) or E5-small dense embeddings
- **Group-aware splits** — no task-level data leakage in train/test/CV
- **Multiple backends** — XGBoost (default), Random Forest, Logistic Regression
- **Faithful metrics** — R-precision (P@R), Recall@k

## Documentation

| Document | Description |
|---|---|
| [Overview](docs/overview.md) | What ShortChain is, the problem it solves, the paper |
| [Getting Started](docs/getting-started.md) | Installation, quick start, data format |
| [Architecture](docs/architecture.md) | System design, data flow, module details |
| [Configuration](docs/configuration.md) | Complete YAML reference for every setting |
| [API Reference](docs/api-reference.md) | Python API for all public classes |
| [Integration Guide](docs/integration.md) | How to integrate into your agent system |
| [Development](docs/development.md) | Testing, contributing, extension patterns |

## Configuration

Override defaults by passing a YAML config:

```yaml
classifier:
  model_type: "random_forest"

negatives:
  strategy: "hard"

dataset:
  negative_ratio: 5
```

```bash
python scripts/train.py --dataset data/datasets/ --config my_config.yaml
```

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 — MVP | Core pipeline: ingest → train → evaluate → inference | ✅ Complete |
| 2 — Features | Modular FeaturePipeline, negative sampling, encoders | ✅ Complete |
| 3A — TabSchema | LLM-driven feature extraction (optional enhancement) | Planned |
| 3B — TabSynth | Synthetic data generation for rare patterns | Planned |
| 3C — Span-Level | Per-span decision modeling (experimental mode) | Planned |
| 4 — Benchmarks | AppWorld adapter, real-world evaluation | Planned |

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v          # 100 tests, ~1.5 seconds
```

## License

MIT

---

> Inspired by the research paper *"TabAgent"* (Levy et al., 2026).
