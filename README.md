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
Trajectories → Ingestion → CorpusStats → DatasetBuilder → FeaturePipeline → Classifier → Inference
                  ↓             ↓              ↓                ↓               ↓            ↓
            JSON/JSONL    Frequencies     Pointwise pairs   TF-IDF or       XGBoost/RF    Top-K tools
            with field    Co-occurrence   (context, tool,   E5-small        with state-   ranked by
            mapping       App-tool maps   label) + negs     encoding        aware feats   confidence
```

## Project Structure

```
tabagent/
├── tabagent/                    # Core package
│   ├── config.py                # Pydantic configuration
│   ├── features/                # Feature pipeline
│   │   ├── encoders.py          # Text encoders (TF-IDF, E5-small)
│   │   ├── pipeline.py          # FeaturePipeline orchestrator
│   │   ├── context.py           # State-aware context features
│   │   ├── tool.py              # Tool/candidate features
│   │   └── stats.py             # CorpusStats
│   ├── ingest/                  # Trajectory loading & validation
│   ├── dataset/                 # Dataset construction & splitting
│   │   ├── builder.py           # Pointwise pair construction
│   │   ├── negatives.py         # Negative sampling strategies
│   │   └── splitter.py          # Group-aware stratified splits
│   ├── head/                    # Classifier training & inference
│   ├── evaluation/              # Metrics (R-precision, Recall@k)
│   └── utils/                   # I/O, logging
├── scripts/                     # CLI entry points
├── tests/                       # Test suite (100 tests)
├── configs/                     # YAML configurations
└── data/example/                # Example trajectories
```

## Key Features

- **Agent-agnostic**: Works with any agent system that produces JSON/JSONL execution logs
- **Configurable field mapping**: Map your log format to TabAgent's schema via YAML
- **Modular feature pipeline**: Separate context, tool, and encoding stages with `FeaturePipeline`
- **State-aware features**: Step index, last action, history summary, tool diversity
- **Pluggable negative sampling**: Random, hard (same-app, co-usage, description-similar), or mixed
- **Hybrid text encoding**: TF-IDF (default) or E5-small dense embeddings (`pip install -e ".[embeddings]"`)
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

negatives:
  strategy: "hard"                # random | hard | mixed

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
