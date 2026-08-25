# API Reference

## Ingestion

### `Span`

```python
from shortchain.ingest.schema import Span

span = Span(
    agent_name="CoderAgent",
    action="send_email(to='john@example.com')",
    observation="Email sent successfully",
    thoughts="Need to send the composed email",
)

span.tool_name   # "send_email" — auto-extracted from action
```

| Attribute | Type | Description |
|---|---|---|
| `agent_name` | `str` | Name of the agent that took this span |
| `action` | `str \| None` | Tool/API called (e.g., `"send_email"` or `"send_email(args)"`) |
| `observation` | `str \| None` | Result of the action |
| `thoughts` | `str \| None` | Agent reasoning trace |
| `metadata` | `dict` | Extra fields from the raw record |
| `tool_name` | `str \| None` | **Property** — extracted tool name (strips arguments) |

---

### `Trajectory`

```python
from shortchain.ingest.schema import Trajectory

traj = Trajectory(
    task_id="task_001",
    intent="Send an email to John",
    spans=[span1, span2, span3],
    success=True,
    app_name="gmail",
)

traj.tools_used     # {"search_contacts", "create_draft", "send_email"}
traj.tool_sequence  # ["search_contacts", "create_draft", "send_email"]
traj.n_spans        # 3
traj.last_thought   # "Sending the drafted email"
traj.summary()      # {"task_id": "task_001", "intent": "Send an email...", ...}
```

| Attribute / Property | Type | Description |
|---|---|---|
| `task_id` | `str` | Unique task identifier |
| `intent` | `str` | User's original goal |
| `spans` | `list[Span]` | Execution spans |
| `success` | `bool` | Whether the task completed successfully |
| `app_name` | `str` | Application context |
| `tools_used` | `set[str]` | **Auto-derived** — unique tools called |
| `tool_sequence` | `list[str]` | **Property** — ordered tool calls |
| `n_spans` | `int` | **Property** — number of spans |
| `last_thought` | `str \| None` | **Property** — last reasoning trace |

---

### `load_trajectories()`

```python
from shortchain.ingest.loader import load_trajectories

# Load from directory (reads all .json/.jsonl files)
trajs = load_trajectories("examples/traces/")

# Load from single file
trajs = load_trajectories("examples/traces/trajectories.jsonl")

# With custom config
from shortchain.config import IngestConfig
trajs = load_trajectories("logs/", config=IngestConfig(success_only=False))
```

---

## Features

### `CorpusStats`

```python
from shortchain.features.stats import CorpusStats

stats = CorpusStats.from_trajectories(trajectories)

stats.tool_frequency              # {"send_email": 5, "search_contacts": 3, ...}
stats.co_occurrence               # {"send_email": {"create_draft": 4, ...}, ...}
stats.app_tools                   # {"gmail": ["create_draft", "send_email", ...]}
stats.app_tool_count              # {"gmail": 5, "spotify": 8}
stats.total_trajectories          # 15

stats.get_same_app_tools("gmail")          # ["create_draft", "send_email", ...]
stats.get_co_occurring_tools("send_email") # {"create_draft": 4, ...}
stats.get_tool_freq("send_email")          # 5
```

---

### `ContextFeatureBuilder`

```python
from shortchain.features.context import ContextFeatureBuilder

builder = ContextFeatureBuilder(
    corpus_stats=stats,          # Optional
    include_state=True,          # span_index, last_action, etc.
    include_dependencies=True,   # tool_diversity, app_tool_count
)

# Trajectory-level features (default)
features = builder.build(traj, span_index=None)
# Returns: {"task_id": "...", "intent": "...", "n_spans": 4, "span_index": 4,
#           "last_action": "send_email", "tool_diversity": 0.75, ...}

# Span-level features (for future use)
features = builder.build(traj, span_index=2)
# Returns features as-of span 2 only
```

---

### `ToolFeatureBuilder`

```python
from shortchain.features.tool import ToolFeatureBuilder

builder = ToolFeatureBuilder(corpus_stats=stats)

features = builder.build(
    tool_name="send_email",
    tool_meta={"description": "Send an email to a recipient"},
    context=context_features,  # from ContextFeatureBuilder
)
# Returns: {"tool_name": "send_email", "tool_description": "...",
#           "tool_name_length": 10, "has_description": True,
#           "tool_app_match": 1, "tool_frequency": 5,
#           "tool_co_occurrence": 3.5}
```

---

### `FeaturePipeline`

```python
from shortchain.features.pipeline import FeaturePipeline
from shortchain.config import FeaturesConfig

pipeline = FeaturePipeline(config=FeaturesConfig())

# Fit and transform (training)
X = pipeline.fit_transform(train_df)          # np.ndarray
X = pipeline.fit_transform(list_of_dicts)     # also works

# Transform (inference)
X = pipeline.transform(test_df)

# Persistence
pipeline.save("models/pipeline.pkl")
pipeline = FeaturePipeline.load("models/pipeline.pkl")
```

---

## Dataset

### `DatasetBuilder`

```python
from shortchain.dataset.builder import DatasetBuilder, build_dataset

# Full control
builder = DatasetBuilder(
    config=dataset_config,
    features_config=features_config,
    negatives_config=negatives_config,
    tool_catalog={"send_email": "Send an email", ...},  # Optional
)
df = builder.build(trajectories)

# Convenience
df = build_dataset(trajectories)

# Access corpus stats after build
builder.corpus_stats  # CorpusStats object
```

**Output DataFrame columns:**

| Column | Type | Source |
|---|---|---|
| `task_id` | str | Context |
| `intent` | str | Context |
| `app_name` | str | Context |
| `n_spans` | int | Context |
| `previous_tools` | str | Context |
| `last_thought` | str | Context |
| `span_index` | int | State feature |
| `last_action` | str | State feature |
| `last_observation` | str | State feature |
| `unique_tools_so_far` | int | State feature |
| `history_summary` | str | State feature |
| `tool_diversity` | float | Dependency feature |
| `app_tool_count` | int | Dependency feature |
| `tool_name` | str | Tool feature |
| `tool_description` | str | Tool feature |
| `tool_name_length` | int | Tool feature |
| `has_description` | bool | Tool feature |
| `tool_app_match` | int | Tool feature |
| `tool_frequency` | int | Tool feature (corpus) |
| `tool_co_occurrence` | float | Tool feature (corpus) |
| `label` | int | 1 = positive, 0 = negative |

---

### `NegativeSampler`

```python
from shortchain.dataset.negatives import create_sampler
from shortchain.config import NegativeSamplingConfig

sampler = create_sampler(
    config=NegativeSamplingConfig(strategy="hard"),
    catalog={"send_email": "...", "search_contacts": "...", ...},
    corpus_stats=stats,
)

negatives = sampler.sample(
    positive_tools={"send_email", "create_draft"},
    app_name="gmail",
    n=6,  # number of negatives to sample
)
# Returns: ["search_flights", "play_tracks", ...]
```

---

### `GroupStratifiedSplitter`

```python
from shortchain.dataset.splitter import GroupStratifiedSplitter
from shortchain.config import SplitterConfig

splitter = GroupStratifiedSplitter(SplitterConfig(n_folds=5, test_size=0.2))

# Single train/test split
train_df, test_df = splitter.train_test_split(df)

# K-fold cross-validation
for train_fold, val_fold in splitter.kfold_split(df):
    # All rows from one task_id stay in the same fold
    ...
```

---

## Classifier

### `ShortChainClassifier`

```python
from shortchain.model.classifier import ShortChainClassifier
from shortchain.config import ClassifierConfig

clf = ShortChainClassifier(ClassifierConfig(model_type="xgboost"))

# Train
clf.fit(X_train, y_train)                    # X is raw DataFrame

# Predict
probas = clf.predict_proba(X_test)           # np.ndarray, shape (n,)
preds = clf.predict(X_test)                  # np.ndarray, binary
shortlists = clf.shortlist(X_test, top_k=5)  # list of (tool, score) per task

# Persistence
clf.save("models/shortchain.pkl")
clf = ShortChainClassifier.load("models/shortchain.pkl")  # Supports v1 and v2
```

---

### `InferenceEngine`

```python
from shortchain.model.inference import InferenceEngine

# Load from disk
engine = InferenceEngine(model_path="models/shortchain.pkl", top_k=5)

# Or from an existing classifier
engine = InferenceEngine(classifier=clf, top_k=5)

# Single-context inference
shortlist = engine.predict(
    context={
        "intent": "Send an email to John",
        "app_name": "gmail",
        "n_spans": 2,
        "previous_tools": "search_contacts",
        "last_thought": "Found John's email address",
    },
    candidates=[
        {"tool_name": "send_email", "tool_description": "Send an email"},
        {"tool_name": "create_draft", "tool_description": "Create a draft"},
        {"tool_name": "play_tracks", "tool_description": "Play music tracks"},
    ],
    top_k=3,
)
# Returns: [("send_email", 0.94), ("create_draft", 0.72), ("play_tracks", 0.11)]

# Batch inference
results = engine.predict_batch(test_df, top_k=5)
# Returns: {"task_001": [("send_email", 0.94), ...], "task_002": [...]}
```

---

### `Trainer`

```python
from shortchain.model.trainer import Trainer

trainer = Trainer(
    classifier_config=ClassifierConfig(),
    splitter_config=SplitterConfig(n_folds=5),
    eval_config=EvaluationConfig(),
)

# Cross-validation
cv_results = trainer.train_with_cv(train_df)
# Returns: {"fold_metrics": [...], "aggregate": {"r_precision": 0.87, ...}}

# Final model
clf = trainer.train_final(train_df, save_path="models/shortchain.pkl")
```

---

## Evaluation

### `compute_metrics()`

```python
from shortchain.evaluation.metrics import compute_metrics, format_metrics

metrics = compute_metrics(
    y_true=y_test,                     # np.ndarray, binary labels
    y_proba=clf.predict_proba(X_test), # np.ndarray, probabilities
    X_val=X_test,                      # DataFrame with task_id, tool_name
    k_values=[3, 5, 7, 9],
)
# Returns: {"accuracy": 0.93, "f1": 0.83, "r_precision": 1.0,
#           "recall_at_3": 1.0, "recall_at_5": 1.0, ...}

print(format_metrics(metrics))
```
