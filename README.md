# TabAgent

> Replace expensive LLM decision components in agentic systems with compact tabular-textual classifiers.

**TabAgent** is an optimization layer that learns from successful agent execution traces to train lightweight classifiers (~50M params) that can replace expensive generative LLM decisions. Based on [Levy et al., 2026](https://arxiv.org/abs/2602.16429).

## Quick Start

```bash
# Install (with uv)
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e ".[dev]"

# Or traditional pip
pip install -e ".[dev]"

# Optional: dense embeddings (E5-small)
pip install -e ".[embeddings]"

# Build dataset from example trajectories
python scripts/build_dataset.py --trajectories data/example/ --output data/datasets/

# Train classifier (XGBoost, 3-fold CV)
python scripts/train.py --dataset data/datasets/ --output models/tabagent.pkl --folds 3

# Evaluate
python scripts/evaluate.py --model models/tabagent.pkl --dataset data/datasets/test.csv
```

## Architecture

```
Trajectories → Ingestion → CorpusStats ─┐
                                         ├→ ContextFeatureBuilder ─┐
                                         ├→ ToolFeatureBuilder    ─┤
                                         └→ NegativeSampler       ─┤
                                                                    ↓
                                              DatasetBuilder → DataFrame
                                                                    ↓
                                              FeaturePipeline → np.ndarray
                                                                    ↓
                                              TabAgentClassifier → predictions
                                                                    ↓
                                              Metrics (P@R, R@K, F1)
```

## Project Structure

```
tabagent/
├── tabagent/                    # Core package
│   ├── config.py                # Pydantic configuration
│   ├── features/                # Feature pipeline (Phase 2)
│   │   ├── encoders.py          # TF-IDF & E5-small text encoders
│   │   ├── pipeline.py          # FeaturePipeline orchestrator
│   │   ├── context.py           # State-aware context features
│   │   ├── tool.py              # Tool/candidate features
│   │   └── stats.py             # CorpusStats (precomputed statistics)
│   ├── ingest/                  # Trajectory loading & validation
│   ├── dataset/                 # Dataset construction & splitting
│   │   ├── builder.py           # Pointwise (context, tool, label) pairs
│   │   ├── negatives.py         # Negative sampling strategies
│   │   └── splitter.py          # Group-aware stratified splits
│   ├── head/                    # Classifier training & inference
│   ├── evaluation/              # Metrics (R-precision, Recall@k)
│   └── utils/                   # I/O, logging
├── scripts/                     # CLI entry points
├── tests/                       # 100 unit tests
├── configs/                     # YAML configurations
└── data/example/                # Example trajectories
```

## Key Features

- **Agent-agnostic**: Works with any agent system that produces JSON/JSONL execution logs
- **Configurable field mapping**: Map your log format to TabAgent's schema via YAML
- **Modular feature pipeline**: Separate context, tool, and encoding stages — swap encoders or add features without touching the classifier
- **State-aware features**: Step index, last action, history summary, tool diversity, and app-tool counts
- **Pluggable negative sampling**: Random (default), hard (same-app, co-usage, description-similar), or mixed strategies
- **Hybrid text encoding**: TF-IDF (default) or E5-small dense embeddings (optional)
- **Pointwise reduction**: Transforms ranking problems into binary classification
- **Group-aware splits**: No task-level data leakage in train/test/CV
- **Multiple backends**: XGBoost (default), Random Forest, Logistic Regression
- **Head-matched metrics**: R-precision, Recall@k from the paper
- **CorpusStats**: Precomputed tool frequency, co-occurrence, and app-tool mappings
- **Dual input support**: `FeaturePipeline` accepts both `pd.DataFrame` and `list[dict]`

## Configuration

Override defaults by passing a YAML config:

```yaml
# Feature pipeline
features:
  text_encoder: "tfidf"              # tfidf | e5-small | auto
  include_state_features: true
  include_dependency_features: true

# Negative sampling
negatives:
  strategy: "hard"                    # random | hard | mixed
  same_app_weight: 0.4
  co_usage_weight: 0.3
  similarity_weight: 0.3

# Classifier
classifier:
  model_type: "xgboost"
  xgboost:
    n_estimators: 300
    max_depth: 8

# Dataset
dataset:
  negative_ratio: 5
```

```bash
python scripts/train.py --dataset data/datasets/ --config my_config.yaml
```

## Development

```bash
# Setup
uv venv --python 3.10 && source .venv/bin/activate
uv pip install -e ".[dev]"

# Run tests (100 tests, ~2s)
pytest tests/ -v

# Run specific test suites
pytest tests/test_features.py -v    # Feature pipeline tests
pytest tests/test_negatives.py -v   # Negative sampling tests
```

## Roadmap

| Phase | Status | Description |
|---|---|---|
| **Phase 1** — MVP | ✅ Complete | Core pipeline: ingestion → dataset → classifier → evaluation |
| **Phase 2** — Feature Pipeline | ✅ Complete | Modular features, hard negatives, dense encoding, FeaturePipeline |
| **Phase 3** — TabSchema | 🔜 Next | LLM-driven feature extraction (analyzer, judges, code executor) |
| **Phase 4** — TabSynth | ⬜ Planned | Synthetic data generation with schema-aligned augmentation |
| **Phase 5** — Baselines | ⬜ Planned | BM25, Dense Semantic Retrieval, LLM controller baselines |
| **Phase 6** — AppWorld | ⬜ Planned | Benchmark adapter for real agent trajectories |

## License

MIT
