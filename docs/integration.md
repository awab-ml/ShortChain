# Integration Guide

This guide explains how to integrate ShortChain into your existing agent system as an optimization layer.

## Integration Patterns

ShortChain supports three deployment modes, ranging from conservative to aggressive cost savings:

```
                 ┌──────────────────────────────────────────┐
                 │          YOUR AGENT SYSTEM               │
                 │                                          │
                 │   User Request → Agent Controller        │
                 │                    │                     │
                 │           "Which tool next?"             │
                 │                    │                     │
                 │        ┌───────────┼───────────┐        │
                 │        ▼           ▼           ▼        │
                 │   ┌─────────┐ ┌─────────┐ ┌─────────┐  │
                 │   │ Mode A  │ │ Mode B  │ │ Mode C  │  │
                 │   │ Replace │ │ Hybrid  │ │Adaptive │  │
                 │   │  ~95%   │ │  ~85%   │ │  ~70%   │  │
                 │   │ savings │ │ savings │ │ savings │  │
                 │   └─────────┘ └─────────┘ └─────────┘  │
                 └──────────────────────────────────────────┘
```

---

## Mode A: Full Replacement

ShortChain replaces the LLM entirely for tool selection. Use when you have high confidence in the classifier (well-trained model, tools are familiar).

```python
from shortchain.head.inference import InferenceEngine

# Load once at startup (~50ms)
engine = InferenceEngine(model_path="models/shortchain.pkl", top_k=5)


def choose_tool(context: dict, tool_catalog: list[dict]) -> str:
    """Replace LLM tool selection with ShortChain."""
    shortlist = engine.predict(context, tool_catalog, top_k=1)
    return shortlist[0][0]  # top tool name
```

| Metric | Value |
|---|---|
| Latency | ~1ms per decision |
| Cost | $0 per decision |
| Savings | ~95% cost, ~99% latency |
| Risk | Model may pick wrong tool for unseen patterns |

---

## Mode B: Hybrid (Shortlist + LLM)

ShortChain narrows candidates from N tools to 5, then the LLM picks from the shortlist. The LLM processes 5 tool descriptions instead of 100+, reducing prompt size and cost.

```python
from shortchain.head.inference import InferenceEngine

engine = InferenceEngine(model_path="models/shortchain.pkl", top_k=5)


def choose_tool(context: dict, tool_catalog: list[dict], llm) -> str:
    """ShortChain shortlists, LLM makes final decision."""
    # Step 1: ShortChain narrows to top-5 (~1ms)
    shortlist = engine.predict(context, tool_catalog, top_k=5)
    
    # Step 2: LLM picks from shortlist (cheaper — 5 tools, not 100+)
    shortlisted_tools = [
        {"name": name, "score": score} for name, score in shortlist
    ]
    tool_name = llm.select_tool(
        intent=context["intent"],
        candidates=shortlisted_tools,
    )
    return tool_name
```

| Metric | Value |
|---|---|
| Latency | ~1ms (ShortChain) + ~200ms (LLM on small prompt) |
| Cost | ~80-85% reduction (LLM sees 5 tools, not 100+) |
| Risk | Low — LLM has final say, ShortChain only filters |

---

## Mode C: Adaptive (Confidence-Based Routing)

Use ShortChain when it's confident, fall back to the LLM when it's not. Best balance of cost and accuracy.

```python
from shortchain.head.inference import InferenceEngine

engine = InferenceEngine(model_path="models/shortchain.pkl", top_k=5)

# Confidence thresholds (tune these on your eval set)
HIGH_CONFIDENCE = 0.85
LOW_CONFIDENCE = 0.50


def choose_tool(context: dict, tool_catalog: list[dict], llm) -> str:
    """Route to ShortChain or LLM based on confidence."""
    shortlist = engine.predict(context, tool_catalog, top_k=5)
    top_tool, top_score = shortlist[0]
    
    if top_score >= HIGH_CONFIDENCE:
        # ShortChain is confident → use directly (no LLM call)
        return top_tool
    
    elif top_score >= LOW_CONFIDENCE:
        # Medium confidence → LLM picks from shortlist
        return llm.select_tool(
            intent=context["intent"],
            candidates=[{"name": n, "score": s} for n, s in shortlist],
        )
    
    else:
        # Low confidence → full LLM decision (all tools)
        return llm.select_tool(
            intent=context["intent"],
            candidates=tool_catalog,
        )
```

| Metric | Value |
|---|---|
| Latency | Variable: 1ms (high conf) to ~500ms (low conf) |
| Cost | ~70% reduction on average (depends on confidence distribution) |
| Risk | Lowest — graceful fallback to LLM |

---

## Step-by-Step Integration

### 1. Collect Agent Logs

Your agent must log its execution traces. At minimum, each log entry needs:

```json
{
  "task_id": "unique_id",
  "intent": "what the user asked",
  "steps": [
    {"action": "tool_name", "observation": "result"}
  ]
}
```

Optional but recommended: `app_name`, `thoughts`, `success`.

### 2. Map Your Fields

If your log format uses different field names, create a config:

```yaml
# shortchain_config.yaml
ingest:
  field_map:
    task_id: "request_id"        # your field name → ShortChain field
    intent: "user_query"
    steps: "execution_trace"
    action: "function_call"
    observation: "function_result"
```

### 3. Build Dataset

```bash
python scripts/build_dataset.py \
    --trajectories /path/to/your/logs/ \
    --output data/datasets/ \
    --config shortchain_config.yaml
```

### 4. Train Model

```bash
python scripts/train.py \
    --dataset data/datasets/ \
    --output models/my_agent.pkl
```

### 5. Evaluate

```bash
python scripts/evaluate.py \
    --model models/my_agent.pkl \
    --dataset data/datasets/test.csv
```

Check the metrics. Key thresholds:

| Metric | Good | Acceptable | Poor |
|---|---|---|---|
| R-Precision | > 0.85 | 0.70 – 0.85 | < 0.70 |
| Recall@5 | > 0.90 | 0.80 – 0.90 | < 0.80 |
| F1 | > 0.80 | 0.65 – 0.80 | < 0.65 |

### 6. Deploy

```python
from shortchain.head.inference import InferenceEngine

engine = InferenceEngine(model_path="models/my_agent.pkl")

# In your agent's decision loop:
shortlist = engine.predict(context, candidates, top_k=5)
```

---

## Programmatic Pipeline (No CLI)

For full control, use the Python API directly:

```python
from shortchain.config import load_config
from shortchain.ingest.loader import load_trajectories
from shortchain.dataset.builder import DatasetBuilder
from shortchain.dataset.splitter import GroupStratifiedSplitter
from shortchain.head.trainer import Trainer
from shortchain.head.inference import InferenceEngine
from shortchain.evaluation.metrics import compute_metrics

# 1. Load config
cfg = load_config("my_config.yaml")

# 2. Ingest trajectories
trajectories = load_trajectories("logs/", config=cfg.ingest)

# 3. Build dataset
builder = DatasetBuilder(
    config=cfg.dataset,
    features_config=cfg.features,
    negatives_config=cfg.negatives,
    tool_catalog={
        "send_email": "Send an email to a recipient",
        "search_contacts": "Search for contacts by name",
        # ... your full tool catalog
    },
)
df = builder.build(trajectories)

# 4. Split
splitter = GroupStratifiedSplitter(cfg.splitter)
train_df, test_df = splitter.train_test_split(df)

# 5. Train
trainer = Trainer(
    classifier_config=cfg.classifier,
    splitter_config=cfg.splitter,
    eval_config=cfg.evaluation,
)
cv_results = trainer.train_with_cv(train_df)
clf = trainer.train_final(train_df, save_path="models/my_agent.pkl")

# 6. Evaluate
y_proba = clf.predict_proba(test_df.drop(columns=["label"]))
metrics = compute_metrics(
    y_true=test_df["label"].values,
    y_proba=y_proba,
    X_val=test_df.drop(columns=["label"]),
)
print(metrics)

# 7. Deploy
engine = InferenceEngine(classifier=clf, top_k=5)
```

---

## Context Dict Format

When calling `engine.predict()`, provide a context dict with these fields:

| Field | Type | Required | Description |
|---|---|---|---|
| `intent` | str | Yes | User's original goal |
| `app_name` | str | Recommended | Application context |
| `n_steps` | int | Recommended | Number of steps taken so far |
| `previous_tools` | str | Recommended | Pipe-separated tools used: `"search_contacts\|create_draft"` |
| `last_thought` | str | Optional | Last agent reasoning trace |

And a candidates list:

```python
candidates = [
    {"tool_name": "send_email", "tool_description": "Send an email to a recipient"},
    {"tool_name": "create_draft", "tool_description": "Create an email draft"},
    # ... all available tools
]
```

---

## Retraining

Models should be retrained when:

- New tools are added to the catalog
- Agent behavior patterns change
- Performance metrics degrade below acceptable thresholds
- Every 1–4 weeks as a maintenance cycle

Retraining is fast (~5 seconds for 1000 trajectories) and can be automated:

```bash
# Automated retraining script
python scripts/build_dataset.py --trajectories logs/latest/ --output data/datasets/
python scripts/train.py --dataset data/datasets/ --output models/shortchain_v2.pkl
python scripts/evaluate.py --model models/shortchain_v2.pkl --dataset data/datasets/test.csv

# If metrics are good, swap the model:
mv models/shortchain_v2.pkl models/shortchain.pkl
```

---

## Cold Start Strategy

When deploying ShortChain for the first time with no training data:

1. **Week 1–2**: Run your agent with full LLM decisions. Log all traces.
2. **Week 2**: Once you have 50+ successful trajectories, train the first model.
3. **Week 2–4**: Deploy in **Mode C (Adaptive)** with conservative thresholds.
4. **Week 4+**: As the model improves, lower thresholds or switch to **Mode B**.
5. **Ongoing**: Retrain periodically as you accumulate more data.
