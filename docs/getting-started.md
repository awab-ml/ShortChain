# Getting Started

## Prerequisites

- Python 3.10+
- pip (or uv)

## Installation

```bash
git clone https://github.com/awab-ml/ShortChain.git
cd ShortChain

# Core package + dev tools
pip install -e ".[dev]"

# Telemetry collection (SDK + receiver)
pip install -e ".[sdk,receiver]"

# Optional: dense embeddings (E5-small) for feature / baseline encoding
pip install -e ".[embeddings]"
```

## Quick start on the shipped example

The repo ships 15 sample trajectories under `examples/traces/`. Run the full
pipeline without an agent or a receiver:

```bash
# 1. Build a pointwise dataset from the example trajectories
python -m shortchain dataset \
    --trajectories examples/traces/ \
    --config examples/configs/example.yaml \
    --output /tmp/sc-ds

# 2. Train a classifier with group-aware cross-validation
python -m shortchain train \
    --dataset /tmp/sc-ds \
    --output /tmp/sc-model.pkl

# 3. Evaluate on the held-out test set
python -m shortchain evaluate \
    --model /tmp/sc-model.pkl \
    --dataset /tmp/sc-ds/test.csv
```

Expected tail of the evaluation:

```
recall_at_5: 1.0000
recall_at_7: 1.0000
recall_at_9: 1.0000
✓ Evaluation complete
```

### What just happened

1. **dataset** loaded 15 example trajectories, extracted context/tool features,
   created positive pairs for tools the agent used and 3:1 negatives from the
   catalog, then split at the *task* level (no task appears in both train and
   test).
2. **train** ran 5-fold group-aware cross-validation, then trained a final
   XGBoost classifier over the `FeaturePipeline` encoding.
3. **evaluate** scored every held-out candidate and reported ranking metrics
   (R-Precision, Recall@k) plus classification metrics (F1, AUC).

## Production: collect, then train

For live OTEL traces, instrument the agent and run the receiver:

```python
# python -m pip install "shortchain[sdk,receiver]"
import os
from shortchain.sdk import ShortChain

ShortChain.init(
    api_key=os.environ["SHORTCHAIN_API_KEY"],
    app_name="support-agent",
    endpoint=os.environ.get("SHORTCHAIN_ENDPOINT", "http://127.0.0.1:4318"),
)
```

```bash
shortchain receive --config configs/runtime.yaml
```

The receiver writes projected trajectories to `data/runtime/trajectories.jsonl`
(created with `0600`) plus a tool catalog at `data/runtime/catalog.json`. Then
the same three commands train on that output:

```bash
python -m shortchain dataset \
    --trajectories data/runtime/trajectories.jsonl \
    --catalog data/runtime/catalog.json \
    --output data/datasets/runtime
python -m shortchain train \
    --dataset data/datasets/runtime \
    --output models/shortchain.pkl
```

## Using your own JSONL data

ShortChain reads JSON / JSONL. Each record is one agent execution:

```json
{
  "task_id": "task_001",
  "intent": "Send an email to John about the meeting tomorrow at 3pm",
  "app_name": "gmail",
  "success": true,
  "spans": [
    {"agent_name": "ShortlisterAgent", "thoughts": "Need email tools",
     "action": "search_contacts", "observation": "Found John Smith"},
    {"agent_name": "CoderAgent", "thoughts": "Composing email",
     "action": "create_draft", "observation": "Draft created"},
    {"agent_name": "CoderAgent", "thoughts": "Sending email",
     "action": "send_email", "observation": "Sent"}
  ]
}
```

**Required:** `task_id`, `intent`, `spans` (each span needs `action`).
**Recommended:** `app_name`, `success`, `spans[].thoughts`, `spans[].observation`.

If your logs use different field names, map them in YAML:

```yaml
# my_config.yaml
ingest:
  field_map:
    task_id: "id"
    intent: "instruction"
    spans: "actions"
    success: "completed"
    action: "tool_call"
    observation: "result"
    thoughts: "reasoning"
```

```bash
python -m shortchain dataset \
    --trajectories path/to/your/logs/ \
    --config my_config.yaml \
    --output /tmp/my-ds
```

## Programmatic use

```python
from shortchain.ingest import load_trajectories
from shortchain.dataset import DatasetBuilder

trajectories = load_trajectories("path/to/logs/")

builder = DatasetBuilder(tool_catalog={
    "send_email": "Send an email to a recipient",
    "search_contacts": "Search for contacts by name",
})
df = builder.build(trajectories)
```

## Next steps

- [Integration guide](integration.md) — SDK, task root, and the three
  adaptation modes.
- [Architecture](architecture.md) — how the modules connect.
- [Configuration](configuration.md) — full YAML reference.