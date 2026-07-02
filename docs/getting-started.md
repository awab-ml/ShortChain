# Getting Started

## Prerequisites

- Python 3.10+
- pip or uv package manager

## Installation

```bash
# Clone the repository
git clone https://github.com/awab-ml/ShortChain.git
cd ShortChain

# Install with development dependencies
pip install -e ".[dev]"

# Optional: install dense embeddings (E5-small)
pip install -e ".[embeddings]"
```

## Quick Start: Full Pipeline in 3 Commands

```bash
# 1. Build dataset from example trajectories
python scripts/build_dataset.py \
    --trajectories data/example/ \
    --output data/datasets/

# 2. Train classifier with cross-validation
python scripts/train.py \
    --dataset data/datasets/ \
    --output models/shortchain.pkl

# 3. Evaluate on test set
python scripts/evaluate.py \
    --model models/shortchain.pkl \
    --dataset data/datasets/test.csv
```

## What Just Happened?

### Span 1: Build Dataset

The `build_dataset.py` script:

1. **Loaded** 15 example trajectories from `data/example/trajectories.jsonl`
2. **Extracted** context features (intent, app, tools used, thoughts) and tool features (name, description, frequency) for each trajectory
3. **Created positive pairs** (tools actually used → label=1) and **negative pairs** (randomly sampled tools → label=0) at a 3:1 negative ratio
4. **Split** into train/test sets with no task leakage (all rows from one task stay in the same split)
5. **Saved** `train.csv` and `test.csv` to `data/datasets/`

### Span 2: Train

The `train.py` script:

1. **Loaded** `train.csv`
2. **Ran k-fold cross-validation** (default 5 folds) with group-stratified splits
3. **Trained** an XGBoost classifier using `FeaturePipeline` to encode text (TF-IDF), categorical (label encoding), and numeric features
4. **Trained a final model** on all training data
5. **Saved** the model to `models/shortchain.pkl` and CV results to `models/cv_results.json`

### Span 3: Evaluate

The `evaluate.py` script:

1. **Loaded** the trained model and test data
2. **Scored** all test candidates using the classifier
3. **Computed** ranking metrics (R-precision, Recall@k) and classification metrics (F1, accuracy, AUC)

## Expected Output

```
Evaluation Results:
          accuracy: 0.9286
               auc: 1.0000
                f1: 0.8333
         precision: 1.0000
       r_precision: 1.0000
            recall: 0.7143
       recall_at_3: 1.0000
       recall_at_5: 1.0000
       recall_at_7: 1.0000
       recall_at_9: 1.0000
```

## Using Your Own Data

### Trajectory Format

ShortChain reads JSON or JSONL files. Each record represents one agent execution:

```json
{
  "task_id": "task_001",
  "intent": "Send an email to John about the meeting tomorrow at 3pm",
  "app_name": "gmail",
  "success": true,
  "spans": [
    {
      "agent_name": "TaskAnalyzer",
      "thoughts": "User wants to compose and send an email.",
      "action": null,
      "observation": null
    },
    {
      "agent_name": "ShortlisterAgent",
      "thoughts": "Need email composition and sending tools",
      "action": "search_contacts",
      "observation": "Found contact: John Smith <john@example.com>"
    },
    {
      "agent_name": "CoderAgent",
      "thoughts": "Composing email with subject and body",
      "action": "create_draft",
      "observation": "Draft created with id=draft_123"
    },
    {
      "agent_name": "CoderAgent",
      "thoughts": "Sending the drafted email",
      "action": "send_email",
      "observation": "Email sent successfully to john@example.com"
    }
  ]
}
```

**Required fields:**
- `task_id` — unique identifier for this task
- `intent` — the user's original goal (natural language)
- `spans` — list of agent execution spans, each with at least an `action` field

**Optional fields:**
- `app_name` — application context (used for negative sampling and features)
- `success` — whether the task completed successfully (default: `true`)
- `spans[].thoughts` — agent reasoning trace (improves feature quality)
- `spans[].observation` — result of the action

### Custom Field Names

If your logs use different field names, map them in YAML:

```yaml
# my_config.yaml
ingest:
  field_map:
    task_id: "id"             # your field → ShortChain field
    intent: "instruction"
    spans: "actions"
    success: "completed"
    action: "tool_call"
    observation: "result"
    thoughts: "reasoning"
```

```bash
python scripts/build_dataset.py \
    --trajectories path/to/your/logs/ \
    --output data/datasets/ \
    --config my_config.yaml
```

### Providing a Tool Catalog

By default, ShortChain derives the tool catalog from the trajectories (every tool name that appears becomes a catalog entry). For better negative sampling and features, provide explicit tool descriptions:

```python
from shortchain.dataset.builder import DatasetBuilder
from shortchain.ingest.loader import load_trajectories

trajectories = load_trajectories("path/to/logs/")

catalog = {
    "send_email": "Send an email to a recipient",
    "search_contacts": "Search for contacts by name or email",
    "create_draft": "Create an email draft",
    "search_flights": "Search for available flights",
    # ... all your tools
}

builder = DatasetBuilder(tool_catalog=catalog)
df = builder.build(trajectories)
```

## Running Tests

```bash
# All tests (100 tests, ~2 seconds)
pytest tests/ -v

# Specific module
pytest tests/test_features.py -v
pytest tests/test_negatives.py -v
pytest tests/test_classifier.py -v
```
