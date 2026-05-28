# Architecture

## System Overview

ShortChain is organized into six modules that form a linear pipeline:

```
┌───────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌────────────┐
│  Ingest   │───▶│ Features │───▶│  Dataset  │───▶│   Head   │───▶│ Evaluation │
│           │    │          │    │           │    │          │    │            │
│ Load logs │    │ Encode   │    │ Build     │    │ Train /  │    │ R-Precision│
│ Normalize │    │ features │    │ pairs     │    │ Predict  │    │ Recall@k   │
└───────────┘    └──────────┘    └───────────┘    └──────────┘    └────────────┘
      │                │               │               │
      ▼                ▼               ▼               ▼
   schema.py      pipeline.py     builder.py     classifier.py
   loader.py      encoders.py     negatives.py   trainer.py
                  context.py      splitter.py    inference.py
                  tool.py
                  stats.py
```

## Data Flow

### Training Flow

```
1. Raw JSON/JSONL logs
       │
       ▼
2. JSONLTrajectoryLoader.load()
   - Reads files, maps fields via FieldMapConfig
   - Filters by success
   - Output: list[Trajectory]
       │
       ▼
3. DatasetBuilder.build()
   - Computes CorpusStats (frequencies, co-occurrence, app-tool maps)
   - For each trajectory:
     a. ContextFeatureBuilder.build() → context dict
     b. For each tool_used: ToolFeatureBuilder.build() → positive row
     c. NegativeSampler.sample() → negative tool names
     d. For each negative: ToolFeatureBuilder.build() → negative row
   - Output: pd.DataFrame (168 rows for 15 trajectories at 3:1 ratio)
       │
       ▼
4. GroupStratifiedSplitter.train_test_split()
   - Groups by task_id (no leakage)
   - Output: train.csv, test.csv
       │
       ▼
5. Trainer.train_with_cv()
   - K-fold cross-validation (group-aware)
   - Each fold:
     a. FeaturePipeline.fit_transform() on train fold
     b. model.fit() (XGBoost/RF/Logistic)
     c. FeaturePipeline.transform() on val fold
     d. compute_metrics() → fold results
   - Output: CV metrics dict
       │
       ▼
6. Trainer.train_final()
   - Train on all data
   - ShortChainClassifier.save() → models/shortchain.pkl
   - Pickle contains: model + FeaturePipeline + config
```

### Inference Flow

```
1. Context dict + list of candidate tools
       │
       ▼
2. InferenceEngine.predict()
   - Builds DataFrame: one row per candidate
   - Calls classifier.predict_proba()
     a. FeaturePipeline.transform() → np.ndarray
     b. model.predict_proba() → scores
   - Ranks by score, returns top-K
   - Output: [("tool_name", confidence), ...]
   - Latency: ~1ms
```

## Module Details

### Ingest (`shortchain/ingest/`)

**Purpose**: Normalize any agent log format into typed `Trajectory` objects.

```
schema.py
├── Step          — Single agent action (action, observation, thoughts)
│   └── .tool_name  — Extracts tool name from "send_email(to='x')" → "send_email"
└── Trajectory    — Complete execution trace
    ├── .tools_used     — Auto-derived set of tool names
    ├── .tool_sequence  — Ordered list with duplicates
    ├── .n_steps        — Step count
    └── .last_thought   — Last reasoning trace

loader.py
└── JSONLTrajectoryLoader
    ├── .load(path)        — Load from file or directory
    └── FieldMapConfig     — Maps your field names to ShortChain's
```

**Design decisions:**
- `FieldMapConfig` means zero code changes to ingest new log formats — just update YAML
- `Step.tool_name` handles both `"send_email"` and `"send_email(to='x', subject='...')"` formats
- `tools_used` is auto-derived via `@model_validator` — no manual extraction needed

---

### Features (`shortchain/features/`)

**Purpose**: Transform raw (context, tool) pairs into the numeric matrix the classifier consumes.

```
pipeline.py
└── FeaturePipeline                — Orchestrator
    ├── .fit_transform(data)       — Fit encoders + transform
    ├── .transform(data)           — Transform with fitted encoders
    ├── .save(path) / .load(path)  — Persistence
    └── Encodes 4 column types:
        ├── TEXT_COLS:  intent, previous_tools, last_thought,
        │              tool_name, tool_description,
        │              history_summary, last_observation
        │              → TfidfEncoder or DenseEncoder (per column)
        ├── CAT_COLS:  app_name → LabelEncoder
        ├── NUM_COLS:  n_steps, step_index, unique_tools_so_far,
        │              tool_diversity, app_tool_count,
        │              tool_name_length, tool_frequency, tool_co_occurrence
        └── BOOL_COLS: has_description, tool_app_match

encoders.py
├── TfidfEncoder    — sklearn TfidfVectorizer (default, no deps)
├── DenseEncoder    — E5-small via sentence-transformers (opt-in)
│   └── Falls back to TfidfEncoder if dependency unavailable
└── create_encoder()  — Factory: "tfidf" | "e5-small" | "auto"

context.py
└── ContextFeatureBuilder
    ├── Core: task_id, intent, app_name, n_steps, previous_tools, last_thought
    ├── State: step_index, last_action, last_observation, unique_tools_so_far,
    │          history_summary
    └── Dependencies: tool_diversity, app_tool_count
    └── Supports step_index=None (trajectory-level) or step_index=int (step-level)

tool.py
└── ToolFeatureBuilder
    ├── Core: tool_name, tool_description, tool_name_length, has_description
    ├── Cross: tool_app_match (does tool belong to same app?)
    └── Corpus: tool_frequency, tool_co_occurrence

stats.py
└── CorpusStats                    — Precomputed from training trajectories
    ├── tool_frequency             — How many trajectories use each tool
    ├── co_occurrence              — Pairwise tool co-occurrence counts
    ├── app_tools                  — Which tools belong to which app
    └── .from_trajectories()       — Class method to compute from data
```

**Design decisions:**
- `FeaturePipeline` accepts both `pd.DataFrame` and `list[dict]` — DataFrame for training, dicts for production inference
- Text columns get individual encoders (not one shared encoder) so vocabulary is column-specific
- `CorpusStats` is computed once and shared by feature builders and negative samplers
- `step_index=None` vs `step_index=int` enables future step-level features without API changes

---

### Dataset (`shortchain/dataset/`)

**Purpose**: Convert trajectories into supervised training data via pointwise reduction.

```
builder.py
└── DatasetBuilder
    ├── .build(trajectories)       — Main entry point
    ├── ._resolve_catalog()        — Derive or validate tool catalog
    └── ._build_pairs()            — Create pos/neg rows per trajectory

negatives.py
├── RandomSampler          — Uniform random from catalog \ positives
├── HardNegativeSampler    — Weighted mix of:
│   ├── Same-app tools (40%)       — Tools from the same application
│   ├── Co-usage tools (30%)       — Tools that co-occur with positives
│   └── Description-similar (30%)  — Token overlap with positive tools
├── MixedSampler           — Configurable random + hard mix
└── create_sampler()       — Factory: "random" | "hard" | "mixed"

splitter.py
└── GroupStratifiedSplitter
    ├── .train_test_split()        — Single split (GroupShuffleSplit)
    └── .kfold_split()             — K-fold CV (GroupKFold)
    └── Groups by task_id → no task appears in both train and test
```

**Design decisions:**
- Negative ratio is configurable (default 3:1) — higher ratios help with large tool catalogs
- `HardNegativeSampler` precomputes candidate pools at construction time for fast `sample()` calls
- `GroupStratifiedSplitter` uses sklearn's `GroupKFold` — all rows from one task stay together

---

### Head (`shortchain/head/`)

**Purpose**: Train, persist, and run inference with the classifier.

```
classifier.py
└── ShortChainClassifier
    ├── .fit(X, y)                 — Train (creates FeaturePipeline internally)
    ├── .predict_proba(X)          — Score candidates
    ├── .predict(X)                — Binary predictions
    ├── .shortlist(X, top_k)       — Ranked results per task
    ├── .save(path) / .load(path)  — Persistence (v1/v2 format compat)
    └── Backends:
        ├── "xgboost"      → XGBClassifier (default)
        ├── "random_forest" → RandomForestClassifier
        └── "logistic"      → LogisticRegression

trainer.py
└── Trainer
    ├── .train_with_cv()           — K-fold CV, returns aggregate metrics
    └── .train_final()             — Train on all data, optionally save

inference.py
└── InferenceEngine
    ├── .predict(context, candidates, top_k)
    │   — Score one context against multiple tools
    └── .predict_batch(df, top_k)
        — Score multiple tasks at once
```

**Design decisions:**
- `ShortChainClassifier.save()` stores the model, pipeline, and config together — one file to deploy
- v1 models (Phase 1, without FeaturePipeline) are loaded via a legacy compatibility adapter
- `InferenceEngine` is the production API — takes plain dicts, returns `(tool_name, confidence)` tuples

---

### Evaluation (`shortchain/evaluation/`)

**Purpose**: Faithful ranking metrics.

```
metrics.py
├── r_precision()      — P@R: adapts cutoff to each task's relevant-set size
├── recall_at_k()      — R@k: fixed-budget recovery at k ∈ {3, 5, 7, 9}
├── compute_metrics()  — All metrics at once (accuracy, precision, recall, F1, AUC, P@R, R@k)
└── format_metrics()   — Pretty-print for display
```

**Design decisions:**
- R-precision (P@R) from the methodology: if a task uses 3 tools, retrieve the top 3 and measure precision
- All ranking metrics are macro-averaged across tasks
- `compute_metrics()` requires `task_id` column for ranking metrics but works without it for classification metrics

---

### Config (`shortchain/config.py`)

**Purpose**: Single source of truth for all settings. YAML-based with deep-merge.

```
ShortChainConfig (root)
├── IngestConfig
│   └── FieldMapConfig
├── FeaturesConfig
├── NegativeSamplingConfig
├── DatasetConfig
├── SplitterConfig
├── ClassifierConfig
│   ├── XGBoostParams
│   ├── RandomForestParams
│   └── LogisticParams
├── InferenceConfig
└── EvaluationConfig
```

`load_config(path)` loads `configs/default.yaml` as the base and deep-merges any user overrides on top. You only specify what you want to change.

---

### Utils (`shortchain/utils/`)

```
io.py      — read_json, read_jsonl, write_json, write_jsonl, find_files, ensure_dir
logging.py — Rich-based structured logging, get_logger(), setup_file_logging()
```

## Project Structure

```
ShortChain/
├── shortchain/                     # Core package (2,920 lines)
│   ├── __init__.py
│   ├── config.py                 # 11 Pydantic config models (220 lines)
│   ├── ingest/                   # Trajectory loading (307 lines)
│   │   ├── schema.py             #   Step + Trajectory models
│   │   ├── loader.py             #   JSONLTrajectoryLoader
│   │   └── base.py               #   Abstract loader base
│   ├── features/                 # Feature pipeline (834 lines)
│   │   ├── pipeline.py           #   FeaturePipeline orchestrator
│   │   ├── encoders.py           #   TF-IDF + E5-small encoders
│   │   ├── context.py            #   ContextFeatureBuilder
│   │   ├── tool.py               #   ToolFeatureBuilder
│   │   └── stats.py              #   CorpusStats
│   ├── dataset/                  # Dataset construction (617 lines)
│   │   ├── builder.py            #   DatasetBuilder (pointwise reduction)
│   │   ├── negatives.py          #   Random/Hard/Mixed negative samplers
│   │   └── splitter.py           #   GroupStratifiedSplitter
│   ├── head/                     # Classifier (609 lines)
│   │   ├── classifier.py         #   ShortChainClassifier
│   │   ├── trainer.py            #   Trainer (CV + final)
│   │   └── inference.py          #   InferenceEngine
│   ├── evaluation/               # Metrics (197 lines)
│   │   └── metrics.py            #   R-precision, Recall@k, F1, AUC
│   └── utils/                    # Utilities (136 lines)
│       ├── io.py                 #   File I/O helpers
│       └── logging.py            #   Rich logging
├── scripts/                      # CLI entry points (307 lines)
│   ├── build_dataset.py
│   ├── train.py
│   └── evaluate.py
├── tests/                        # Test suite (1,216 lines, 100 tests)
│   ├── test_features.py          #   32 tests
│   ├── test_negatives.py         #   14 tests
│   ├── test_ingest.py            #   20 tests
│   ├── test_dataset.py           #   8 tests
│   ├── test_classifier.py        #   6 tests
│   └── test_metrics.py           #   8 tests + 12 other
├── configs/
│   └── default.yaml              # Default configuration
├── data/example/
│   └── trajectories.jsonl        # 15 example trajectories
├── models/                       # Trained model artifacts
│   ├── shortchain.pkl
│   └── cv_results.json
├── docs/                         # Documentation
├── pyproject.toml                # Package metadata + dependencies
└── README.md
```

**Total**: ~4,458 lines of Python, 100 tests, 88-line YAML config.
