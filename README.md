# TabAgent

> Replace expensive LLM decision components in agentic systems with compact tabular-textual classifiers.

**TabAgent** is an optimization layer that learns from successful agent execution traces to train lightweight classifiers (~50M params) that can replace expensive generative LLM decisions. Based on [Levy et al., 2026](https://arxiv.org/abs/2602.16429).

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Build dataset from example trajectories
python scripts/build_dataset.py --trajectories data/example/ --output data/datasets/

# Train classifier (XGBoost, 5-fold CV)
python scripts/train.py --dataset data/datasets/ --output models/tabagent.pkl

# Evaluate
python scripts/evaluate.py --model models/tabagent.pkl --dataset data/datasets/test.csv
```

## Architecture

```
Trajectories → Ingestion → Dataset Builder → Classifier → Inference
                  ↓              ↓               ↓            ↓
            JSON/JSONL    Pointwise pairs    XGBoost/RF    Top-K tools
            with field    (context, tool,    with TF-IDF   ranked by
            mapping       label) + negs     encoding       confidence
```

## Project Structure

```
tabagent/
├── tabagent/                    # Core package
│   ├── config.py                # Pydantic configuration
│   ├── ingest/                  # Trajectory loading & validation
│   ├── dataset/                 # Dataset construction & splitting
│   ├── head/                    # Classifier training & inference
│   ├── evaluation/              # Metrics (R-precision, Recall@k)
│   └── utils/                   # I/O, logging
├── scripts/                     # CLI entry points
├── tests/                       # Test suite
├── configs/                     # YAML configurations
└── data/example/                # Example trajectories
```

## Key Features

- **Agent-agnostic**: Works with any agent system that produces JSON/JSONL execution logs
- **Configurable field mapping**: Map your log format to TabAgent's schema via YAML
- **Pointwise reduction**: Transforms ranking problems into binary classification
- **Group-aware splits**: No task-level data leakage in train/test/CV
- **Multiple backends**: XGBoost (default), Random Forest, Logistic Regression
- **Head-matched metrics**: R-precision, Recall@k from the paper

## Configuration

Override defaults by passing a YAML config:

```yaml
classifier:
  model_type: "random_forest"
  random_forest:
    n_estimators: 500

dataset:
  negative_ratio: 5
```

```bash
python scripts/train.py --dataset data/datasets/ --config my_config.yaml
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
