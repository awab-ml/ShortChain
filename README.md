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

# Build dataset → Train → Evaluate (3 commands)
python scripts/build_dataset.py --trajectories data/example/ --output data/datasets/
python scripts/train.py --dataset data/datasets/ --output models/shortchain.pkl
python scripts/evaluate.py --model models/shortchain.pkl --dataset data/datasets/test.csv
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
             JSON/JSONL    Frequencies     Pointwise pairs   TF-IDF or       XGBoost/RF    Top-K tools
             with field    Co-occurrence   (context, tool,   E5-small        with state-   ranked by
             mapping       App-tool maps   label) + negs     encoding        aware feats   confidence
```

## Project Structure

```
ShortChain/
├── shortchain/                    # Core package (2,920 lines)
│   ├── config.py                # 11 Pydantic config models
│   ├── ingest/                  # Trajectory loading & normalization
│   ├── features/                # Feature pipeline (encoders, context, tool, stats)
│   ├── dataset/                 # Dataset construction & negative sampling
│   ├── head/                    # Classifier, trainer, inference engine
│   ├── evaluation/              # R-precision, Recall@k, F1, AUC
│   └── utils/                   # I/O, logging
├── scripts/                     # CLI entry points
├── tests/                       # 100 tests (~1.5s)
├── configs/default.yaml         # Default configuration
├── data/example/                # 15 example trajectories
└── docs/                        # Full documentation
```

## Key Features

- **Agent-agnostic** — works with any agent that produces JSON execution logs
- **Configurable field mapping** — map your log fields to ShortChain's schema via YAML
- **Modular feature pipeline** — context, tool, and encoding stages via `FeaturePipeline`
- **State-aware features** — step index, last action, history summary, tool diversity
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
| 3C — Step-Level | Per-step decision modeling (experimental mode) | Planned |
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
