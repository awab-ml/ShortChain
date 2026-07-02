# ShortChain: Final Implementation Plan (MVP — Phase 1)

## Philosophy

> **ShortChain is an optimization layer that evolves on top of an existing agentic system.**
> Start with a working classifier, validate the idea, then scale complexity.

---

## Project Structure

```
shortchain/
├── pyproject.toml
├── README.md
├── configs/
│   └── default.yaml
├── shortchain/
│   ├── __init__.py
│   ├── config.py                     # Pydantic config models
│   ├── ingest/                       # Trajectory ingestion
│   │   ├── __init__.py
│   │   ├── base.py                   # TrajectoryLoader protocol
│   │   ├── loader.py                 # Generic JSON/JSONL loader
│   │   └── schema.py                 # Pydantic trajectory data models
│   ├── features/                     # Feature pipeline (separated from classifier)
│   │   ├── __init__.py
│   │   ├── context.py                # Context feature builders (state-aware)
│   │   ├── tool.py                   # Tool/candidate feature builders
│   │   ├── encoders.py               # Text encoders (TF-IDF, embeddings)
│   │   └── pipeline.py              # FeaturePipeline orchestrator
│   ├── dataset/                      # Dataset construction
│   │   ├── __init__.py
│   │   ├── builder.py                # Build (context, tool, label) pairs
│   │   ├── negatives.py              # Negative sampling (random + hard)
│   │   └── splitter.py               # Group-aware stratified splits
│   ├── head/                         # Classifier
│   │   ├── __init__.py
│   │   ├── classifier.py             # Unified classifier interface
│   │   ├── trainer.py                # Training with CV
│   │   └── inference.py              # Inference with confidence + hybrid mode
│   ├── evaluation/                   # Metrics & logging
│   │   ├── __init__.py
│   │   ├── metrics.py                # P@R, Recall@k, replacement rate
│   │   └── logger.py                 # Online decision logging hook
│   └── utils/
│       ├── __init__.py
│       ├── io.py
│       └── logging.py
├── scripts/
│   ├── build_dataset.py
│   ├── train.py
│   └── evaluate.py
├── tests/
│   ├── __init__.py
│   ├── test_ingest.py
│   ├── test_features.py
│   ├── test_dataset.py
│   ├── test_classifier.py
│   └── test_metrics.py
└── data/
    └── example/                      # Shipped example trajectories
```

---

## Key Design Decisions

### 1. State-Aware Context Features
Context features capture the agent's *current execution state*, not just the initial intent:
- `intent` — user's original goal
- `span_index` — current position in the execution plan
- `history_summary` — compact aggregation of prior actions and outcomes
- `last_action` — most recent tool/API called
- `last_observation` — result of the last action
- `app_name` — current application context

### 2. Hard Negative Sampling
Beyond random negatives, include tools that are semantically/functionally similar:
- Same-app tools (share the application context)
- Description-similar tools (cosine similarity on embeddings)
- Co-usage tools (tools that frequently appear together but are distinct)

### 3. Hybrid Text Encoding
- TF-IDF as baseline (no extra dependencies)
- E5-small embeddings when `sentence-transformers` is installed
- Feature pipeline auto-selects based on available dependencies

### 4. Confidence-Based Hybrid Decision Mode
- Classifier predicts with calibrated confidence
- High confidence → use classifier (fast path)
- Low confidence → fallback to LLM (safe path)
- Configurable threshold (default: 0.7)

### 5. Online Decision Logging
Every decision is logged:
- Classifier prediction + confidence
- Whether LLM fallback was triggered
- Final outcome (success/failure if available)
- Creates a feedback loop for continuous improvement

### 6. LLM Replacement Rate Metric
`replacement_rate = decisions_by_classifier / total_decisions`
where classifier decisions maintain comparable success rate to LLM-only baseline.
