# Architecture

ShortChain is a linear pipeline of modules named by the operation they
perform. The package is `shortchain/`, and `tests/` mirrors it one-to-one.

```
Live OTEL traces (SDK + OpenLLMetry)
        │
        ▼
shortchain/telemetry/   collect + assemble (receiver)
        │
        ▼
shortchain/ingest/      normalize → Trajectory / Span
        │
        ▼
shortchain/features/    encode context, tool, corpus stats
        │
        ▼
shortchain/dataset/     pointwise (context, tool, label) rows
        │
        ▼
shortchain/model/       compact classifier (train + inference)
        │
        ▼
shortchain/evaluation/  ranking metrics, calibration, hybrid fallback
```

`shortchain/adapters/` holds optional source / benchmark bindings and is not
part of the core pipeline.

## Data flow

### Collection (live)

```
User agent (SDK)                    ShortChain receiver
OpenLLMetry spans ──OTLP HTTP──▶ POST /v1/traces
                                    TraceAssembler (buffer by trace_id,
                                    explicit/root + settle, idle, max_age)
                                    OtelTraceProjector → Trajectory
                                    TrajectoryQualityGate → drop reasons
                                    data/runtime/trajectories.jsonl (0600)
```

`ShortChain.init()` enables the installed OpenLLMetry instrumentations, starts
our own `TracerProvider`, and exports OTLP. A task-root span
(`set_task` / `end_task` / `set_success`) carries the success signal on the
same `trace_id` so the receiver can label traces: a trace without success is
dropped under default quality gates rather than trained on as a silent failure.

### Training

```
1. Ingest         JSONL / OTEL → list[Trajectory]
2. Features       ContextFeatureBuilder / ToolFeatureBuilder / CorpusStats
3. Dataset        positive (used) + negative (sampled) pairs, task-level split
4. Model          FeaturePipeline → XGBoost classifier (group-aware CV)
5. Evaluate       R-Precision, Recall@k, calibration, hybrid metrics
```

### Inference (adapt)

A context dict plus the candidate tool catalog becomes one row per candidate;
the classifier scores each and ranks by probability. `InferenceEngine` wraps
this and returns `(tool_name, confidence)` shortlists in ~1 ms.

## Module details

### Telemetry (`shortchain/telemetry/`)

Production collection: SDK init, OpenLLMetry instrumentor enablement, task-root
span, association injection, OTLP/HTTP receiver, in-process assembler, tool
catalog merge, JSONL writer.

```
sdk.py         ShortChain.init / set_task / set_success / end_task
instrument.py  Own TracerProvider + OpenLLMetry instrumentor enablement
task_span.py   SDK-owned "shortchain.task" root span carrying success
association.py merge-not-replace association injection onto child spans
assembler.py   TraceAssembler: buffer by trace_id, completion rules, bounds
receiver.py    Starlette POST /v1/traces (protobuf + JSON + gzip)
cli.py         shortchain receive (workers locked to 1)
catalog.py     tool-catalog extraction from OTEL traces
```

**Design decisions**

- The receiver is a single worker; multi-worker uvicorn would split one
  `trace_id` across assemblers and flush fragments.
- Success is required for a trainable trace. The quality gate drops traces
  with unknown success (`set_task`…`end_task` is the contract).

### Ingest (`shortchain/ingest/`)

Canonical `Span` / `Trajectory`, JSONL loader + field map, OTEL projector,
quality gate, and reusable transforms (e.g. span-level expansion).

```
schema.py     Span / Trajectory (+ tool_name extraction, derived sets)
base.py       abstract loader
loader.py     JSONLTrajectoryLoader, load_trajectories()
otel.py       OtelSpan / OtelTrace / OtelTraceProjector / OtelTrajectoryLoader
quality.py    TrajectoryQualityGate — drop reasons (missing_intent, …)
transforms.py expand_to_span_trajectories() — span-level expansion
```

**Design decisions**

- `field_map` (in YAML) means new log formats need zero code changes.
- `Span.tool_name` handles both `"send_email"` and `"send_email(args…)"`.
- Offline and live paths land here: the receiver projects to the same schema.

### Features (`shortchain/features/`)

Encode context (intent / state), tool (schema / description), and corpus
statistics.

```
pipeline.py   FeaturePipeline — orchestrator (fit/transform/save/load)
encoders.py   TfidfEncoder (default), DenseEncoder (E5, opt-in), factory
context.py    ContextFeatureBuilder (core + state + dependency fields)
tool.py       ToolFeatureBuilder (name, description, corpus cross-features)
stats.py      CorpusStats — tool frequency / co-occurrence / app-tool maps
```

**Design decisions**

- Text columns get individual encoders so each vocabulary is column-specific.
- `CorpusStats` is computed once and frozen on the training set; it is shared
  by feature builders and negative samplers, and never recomputed from
  evaluation data (leak-free invariant).
- `span_index=None` (trajectory-level) vs `span_index=int` (span-level) selects
  whether context includes decisions made so far.

### Dataset (`shortchain/dataset/`)

Convert trajectories into supervised `(context, tool, label)` rows.

```
builder.py   DatasetBuilder — pointwise reduction
negatives.py RandomSampler / HardNegativeSampler / MixedSampler
splitter.py  GroupStratifiedSplitter — task-grouped train/test and k-fold
```

**Design decisions**

- Negative ratio is configurable (default 3:1).
- Splits group by `task_id`; a task's rows never appear in both train and test.
- Per-decision context never looks ahead to the current or future span
  (no-lookahead contract).

### Model (`shortchain/model/`)

Train, persist, and run the compact classifier.

```
classifier.py  ShortChainClassifier (xgboost / random_forest / logistic)
trainer.py     Trainer — group-aware CV + final fit
inference.py   InferenceEngine — predict() / predict_batch()
```

**Design decisions**

- `ShortChainClassifier.save()` stores the model, `FeaturePipeline`, and config
  in one file.
- Model format is versioned: `v2` (pipeline-based, current) and `v1` (legacy
  inline vectorizers). `load()` inspects the version and reconstructs either.
- `InferenceEngine` is the production API: plain dicts in, ranked
  `(tool_name, confidence)` tuples out.

### Evaluation (`shortchain/evaluation/`)

Rank metrics, calibration, and hybrid fallback — how the backend decides when
to adapt vs. defer.

```
metrics.py       r_precision, recall_at_k, compute_metrics, format_metrics
calibration.py   per-decision confidence scaling (Platt / isotonic)
hybrid.py        selective prediction + LLM fallback
statistics.py    paired-bootstrap CIs, pairwise contrasts, per-metric control
```

### Adapters (`shortchain/adapters/`)

Optional source / benchmark bindings (AppWorld, HALO). Not required to use the
product. They only produce trajectories and tool schemas that flow through the
normal ingest path.

### Config (`shortchain/config.py`)

Single source of truth. `load_config(path)` loads `configs/default.yaml` and
deep-merges user overrides. Config models: `IngestConfig` (+ `FieldMapConfig`),
`FeaturesConfig`, `NegativeSamplingConfig`, `DatasetConfig`, `SplitterConfig`,
`ClassifierConfig`, `InferenceConfig`, `EvaluationConfig`, and runtime settings.

## Repository layout

```
ShortChain/
├── README.md, LICENSE, CONTRIBUTING.md, SECURITY.md
├── pyproject.toml
├── configs/             # default.yaml, runtime.yaml (product configs)
├── docs/                # index, overview, concepts, getting-started, …
├── examples/            # README + collect / train / adapt demos + traces
├── shortchain/          # the package (operation-named modules)
├── scripts/             # maintainer utilities (dataset/train/evaluate)
└── tests/               # mirrors the package: one suite per module
```

`data/` and `models/` are gitignored local working directories for receiver
output, generated datasets, and trained artifacts.