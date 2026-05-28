# Configuration Reference

ShortChain is fully configurable via YAML. All settings have sensible defaults in `configs/default.yaml`. Override any setting by creating a custom YAML file and passing it with `--config`:

```bash
python scripts/train.py --dataset data/datasets/ --config my_config.yaml
```

Custom configs are **deep-merged** on top of defaults — you only specify what you want to change.

---

## Complete Configuration

```yaml
# ─── Ingestion ────────────────────────────────────────────────────────────────
ingest:
  format: "jsonl"                     # "json" or "jsonl"
  success_only: true                  # Only use successful trajectories
  field_map:                          # Map your log fields → ShortChain fields
    task_id: "task_id"
    intent: "intent"
    steps: "steps"
    success: "success"
    agent_name: "agent_name"
    action: "action"
    observation: "observation"
    thoughts: "thoughts"

# ─── Feature Pipeline ────────────────────────────────────────────────────────
features:
  text_encoder: "tfidf"              # "tfidf" | "e5-small" | "auto"
  e5_model_name: "intfloat/e5-small-v2"  # HuggingFace model for dense encoding
  tfidf_max_features: 5000           # Max vocabulary size per text column
  context_fields:                    # Context fields to extract
    - "intent"
    - "app_name"
    - "n_steps"
    - "previous_tools"
    - "last_thought"
  include_state_features: true       # step_index, last_action, last_observation,
                                     # unique_tools_so_far, history_summary
  include_dependency_features: true  # tool_diversity, app_tool_count

# ─── Negative Sampling ───────────────────────────────────────────────────────
negatives:
  strategy: "random"                 # "random" | "hard" | "mixed"
  hard_negative_ratio: 0.5           # Fraction of hard negatives in "mixed" mode
  same_app_weight: 0.4               # Hard neg weight: tools from same app
  co_usage_weight: 0.3               # Hard neg weight: co-occurring tools
  similarity_weight: 0.3             # Hard neg weight: description-similar tools
  # random_state: 42                 # Seed for reproducible sampling

# ─── Dataset Construction ─────────────────────────────────────────────────────
dataset:
  negative_ratio: 3                  # Negative samples per positive (3:1 default)

# ─── Train/Test Splitting ─────────────────────────────────────────────────────
splitter:
  n_folds: 5                         # Number of CV folds
  test_size: 0.2                     # Test set fraction (for train/test split)
  group_by: "task_id"                # Column to group by (prevents leakage)
  stratify_by:
    - "app_name"                     # Stratification columns

# ─── Classifier ───────────────────────────────────────────────────────────────
classifier:
  model_type: "xgboost"             # "xgboost" | "random_forest" | "logistic"
  xgboost:
    n_estimators: 300
    max_depth: 8
    learning_rate: 0.1
    subsample: 0.8
    colsample_bytree: 0.8
    min_child_weight: 3
    eval_metric: "logloss"
    early_stopping_rounds: 20
  random_forest:
    n_estimators: 200
    max_depth: 12
    min_samples_leaf: 5
  logistic:
    C: 1.0
    max_iter: 1000

# ─── Inference ────────────────────────────────────────────────────────────────
inference:
  top_k: 7                          # Default shortlist size
  confidence_threshold: 0.5         # Threshold for binary predictions

# ─── Evaluation ───────────────────────────────────────────────────────────────
evaluation:
  k_values: [3, 5, 7, 9]            # K values for Recall@K
  metrics:
    - "r_precision"
    - "recall_at_k"
    - "accuracy"
    - "f1"
```

---

## Common Configuration Scenarios

### Use Hard Negative Sampling

```yaml
negatives:
  strategy: "hard"
```

### Use Mixed Negatives (50/50 Random + Hard)

```yaml
negatives:
  strategy: "mixed"
  hard_negative_ratio: 0.5
```

### Higher Negative Ratio for Large Catalogs

```yaml
dataset:
  negative_ratio: 5    # 5 negatives per positive
```

### Switch to Random Forest

```yaml
classifier:
  model_type: "random_forest"
  random_forest:
    n_estimators: 500
    max_depth: 16
```

### Use Dense Embeddings (E5-small)

```bash
pip install -e ".[embeddings]"
```

```yaml
features:
  text_encoder: "e5-small"
```

### Custom Field Mapping (for non-standard logs)

```yaml
ingest:
  field_map:
    task_id: "request_id"
    intent: "user_query"
    steps: "execution_trace"
    action: "function_call"
    observation: "function_result"
    thoughts: "chain_of_thought"
```

### Reduce CV Folds for Faster Iteration

```yaml
splitter:
  n_folds: 3
```

### Disable State Features (minimal mode)

```yaml
features:
  include_state_features: false
  include_dependency_features: false
```

---

## Loading Config in Python

```python
from shortchain.config import load_config, ShortChainConfig

# Load defaults
cfg = load_config()

# Load with overrides
cfg = load_config("my_config.yaml")

# Access any setting
print(cfg.classifier.model_type)       # "xgboost"
print(cfg.negatives.strategy)          # "random"
print(cfg.features.text_encoder)       # "tfidf"
print(cfg.splitter.n_folds)            # 5

# Create config programmatically
from shortchain.config import ClassifierConfig, FeaturesConfig

clf_config = ClassifierConfig(model_type="random_forest")
feat_config = FeaturesConfig(text_encoder="e5-small")
```
