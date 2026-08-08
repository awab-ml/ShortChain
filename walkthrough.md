# ShortChain — Walkthrough

> Replace expensive LLM decision components in agentic systems with a compact,
> tabular-textual classifier trained on your own execution traces. This
> walkthrough explains *what* it is, *how* it is architected, and — with a live
> run on **AppWorld** — *what it actually produces*.

---

## 1. Scope & Core Concept

**The problem.** Agents make decisions by repeatedly calling an LLM to
*select* from a closed set of options — which tool to call next, which app to
use, which sub-goal to route to. Every call costs latency and tokens, and the
same decision is made thousands of times.

**The idea.** A single compact classifier (`XGBoost`/`RandomForest`, TF-IDF +
structured features) can learn the *same decision boundary* from your **own
successful traces**: for every decision, build `(context, candidate, label)`
rows and learn `p(correct | context, tool)`. At runtime you score all cand
candidates and **rank by probability** — a single pass, ~1 ms, ~$0.

**Scope.**
- **In:** JSON/JSONL agent trajectories (any format, via a configurable field
  map) + a typed tool catalog (names, descriptions, argument schemas).
- **Out:** a shortlist-ranked tool recommendation per decision point.
- **Out of scope by design:** the agent’s planning/execution itself. ShortChain
  replaces the *selection* component; it is not an agent framework.

The pipeline is divided into five stages that map 1:1 to directories:

```
 Trajectories ──► Ingest ──► Dataset ──► Features ──► Head ──► Evaluate
   (logs)        schema map   pointwise    builders    classifier  metrics
                               (+negatives) (context/    +trainer   (P@R,R@k)
                                             tool/stats) +inference
```

## 2. Architecture Overview

```
shortchain/
├── ingest/            # trajectory schema (Span/Trajectory) + format loaders
├── dataset/           # DatasetBuilder (pointwise rows), negative sampling, splits
├── features/          # context/state, tool/schema, corpus-stats builders + encoder pipeline
├── head/              # ShortChainClassifier (XGBoost/RF/LR), Trainer (group-CV), InferenceEngine
├── evaluation/        # P@R & Recall@k metrics, bootstrap CI/Holm, calibration, selective/hybrid
└── integrations/      # HALO/OpenInference adapter, AppWorld function_calling spec
scripts/               # build_dataset.py · train.py · evaluate.py · run_validation.py · run_llm_baseline.py
configs/               # default.yaml · example.yaml · validation.yaml
```

**Data flow (what each stage does):**

| Stage | Module | Produces |
|---|---|---|
| Ingest | `ingest/schema.py`, `integrations/halo.py` | `Trajectory` (task, intent, ordered spans, derived `tools_used`) |
| Dataset | `dataset/builder.py` | rows `(context, tool, label)` — positives = tools used, negatives sampled |
| Features | `features/context.py`, `features/tool.py`, `features/pipeline.py` | numeric matrix: intent + state + schema + corpus features, vectorised |
| Head | `head/classifier.py`, `head/inference.py` | `p(relevant | context, tool)` — rank by score |
| Evaluate | `evaluation/metrics.py`, `evaluation/statistics.py` | R-Precision, Recall@k, paired-bootstrap 95% CI, Holm |

## 3. Core Techniques (see the comments in each file)

1. **Pointwise reduction** (`dataset/builder.py`) — every decision becomes
   `(context, candidate tool, correct/incorrect)`. The model ranks candidates
   by probability; no per-step LLM calls.
2. **Behavior-grounded features** (`features/`) — context (intent, prior
   state), tool (description + typed argument schema), and corpus statistics
   (frequency, co-occurrence) all come from traces, not hand-tuning.
3. **Leak-free evaluation** — the invariants that make numbers honest:
   - `CorpusStats` are **frozen on the training set** (`builder.py`); eval rows
     never recompute them.
   - Splits group by **task** (`dataset/splitter.py`); the same task never
     spans train and validation.
   - Per-decision context reads **only the steps before** the decision
     (`features/context.py`, `span_index=k` → `spans[:k]`).
4. **Faithful baselines** (`scripts/run_validation.py`) — random, popularity,
   BM25 (TF-IDF), and DSR-E5 (zero-shot dense) are scored on the **same
   candidate rows** as the model; per-method latency is measured per decision.
5. **Head-matched metrics** (`evaluation/metrics.py`) — P@R adapts the cutoff
   to the relevant-set size (completeness claim); Recall@k is the fixed-budget
   view an agent actually consumes. Every table is anchored by the **random**
   baseline.
6. **Beyond the core** (`evaluation/calibration.py`, `hybrid.py`;
   `scripts/run_llm_baseline.py`) — cross-fold group-aware calibration (ECE)
   and selective prediction with a cost-bound LLM fallback, plus cost-bound
   LLM tool-selection baselines. See `docs/p4-benchmark-results.md`.

## 4. Quick Start — the small example (5 apps, 15 traces)

The shipped example logs name their step list `steps`, so we pass
`configs/example.yaml` to point the field map at it.

```bash
# 1) build a pointwise (context, tool, label) dataset
python scripts/build_dataset.py --trajectories data/example \
    --config configs/example.yaml --output /tmp/example_ds

# 2) train (group-aware 3-fold CV) and save the model
python scripts/train.py --dataset /tmp/example_ds \
    --config configs/example.yaml --output /tmp/example_ds/model.pkl --folds 3

# 3) evaluate on the held-out split
python scripts/evaluate.py --model /tmp/example_ds/model.pkl \
    --dataset /tmp/example_ds/test.csv
```

Example evaluation metrics (illustrative):

```
       accuracy: 0.8214        r_precision: 0.4444
            auc: 0.7959       recall_at_3: 0.6111
             f1: 0.7368       recall_at_5: 0.8889
```

## 5. Live Demo — the core pipeline on AppWorld

Setup: AppWorld `test_normal` traces (Gemini agent, `data/traces.jsonl`,
HALO/OpenInference format) + the official `function_calling` API spec
(`data/appworld_api/function_calling`), which supplies tool **descriptions
and argument schemas**.

### 5.1 Task-level benchmark

Run the full honest benchmark (5 seeds, leak-free, baselines on the same
candidate rows, bootstrap CIs + Holm significance):

```bash
python scripts/run_validation.py --config configs/validation.yaml \
    --level task --seeds 5 --output-dir models/validation/task
```

Real output (catalog-wide and app-scoped candidate pools, 57 tasks):

```
method       r_precision  recall_at_1  recall_at_3  recall_at_7  recall_at_9  mrr  ndcg_at_5
model            0.856       0.233       0.611       0.920       0.935     1.000   0.905
bm25             0.389       0.153       0.306       0.466       0.522     0.782   0.479
dsr_e5           0.375       0.152       0.311       0.437       0.469     0.744   0.443
popularity       0.351       0.133       0.293       0.415       0.475     0.720   0.406
random           0.054       0.015       0.035       0.069       0.091     0.173   0.060
latency (ms / decision):  model 5.44 · bm25 11.80 · dsr_e5 16.96
```

Reading the table

- The trained classifier **more than doubles** lexical and dense baselines on
  R-Precision and reaches near-perfect recovery (`R@9 0.935`) over the 91-tool
  catalog — and is **~2–3× faster per decision** than either baseline.
- `recall_at_1 ≈ 0.23` looks low but is correct: tasks use ≈4.7 relevant tools,
  so a perfect top-1 only recovers 1/R; `MRR = 1.000` shows the top-ranked tool
  is *always* one of the relevant tools.
- Every model-vs-baseline contrast is significant after Holm.

### 5.2 Inference “in action” — one real query

Train on all-but-three tasks, then rank the 91-tool catalog for a held-out
task and compare against its real tool usage:

```python
import pandas as pd
from shortchain.config import load_config
from shortchain.dataset.builder import DatasetBuilder
from shortchain.head.classifier import ShortChainClassifier
from shortchain.integrations.halo import load_appworld_traces, reconstruct_catalog
from shortchain.integrations.appworld_api import build_catalog_and_schemas

cfg = load_config("configs/validation.yaml")
traces  = load_appworld_traces("data/traces.jsonl")
catalog = reconstruct_catalog("data/traces.jsonl")
catalog, specs = build_catalog_and_schemas("data/appworld_api/function_calling", catalog)

# Train on all but three held-out tasks (frozen train corpus stats).
train, held = traces[:-3], traces[-3:]
builder = DatasetBuilder(config=cfg.dataset, features_config=cfg.features,
                         negatives_config=cfg.negatives,
                         tool_catalog=catalog, tool_specs=specs)
train_df = builder.build(train)
clf = ShortChainClassifier(cfg.classifier, features_config=cfg.features)
clf.fit(train_df.drop(columns=["label"]), train_df["label"])

# Faithful ranking: score every catalog candidate for the task context.
task = held[0]
cands = [{"tool_name": n, "tool_description": catalog[n]} for n in sorted(catalog)]
eval_builder = DatasetBuilder(config=cfg.dataset, features_config=cfg.features,
                              negatives_config=cfg.negatives, tool_catalog=catalog,
                              tool_specs=specs, corpus_stats=builder.corpus_stats)
rows = eval_builder.build_candidates(task, cands, relevant_tools=task.tools_used)
df = pd.DataFrame(rows)
proba = clf.predict_proba(df.drop(columns=["label"]))
ranked = sorted(zip(df["tool_name"], proba), key=lambda x: x[1], reverse=True)[:5]

print("TASK:", task.intent[:100])
print("RELEVANT:", sorted(task.tools_used))
for tool, p in ranked:
    print(f"  {tool:<44} {p:.3f}{'  *' if tool in task.tools_used else ''}")
```

Output:

```
TASK: Like all the songs from the artists I follow on Spotify.
RELEVANT (ground truth): ['spotify__like_song','spotify__login',
                          'spotify__search_songs','spotify__show_following_artists']
RANKED SHORTLIST (tool -> P(relevant))   [*= in ground truth]
   spotify__login                0.983  *
   spotify__show_liked_songs     0.849
   spotify__download_song        0.647
   spotify__follow_artist        0.554
   spotify__show_liked_albums    0.510
```

The engine opens with the *authenticated entry point* (`spotify__login`) at
~0.98 — exactly the first call any Spotify flow requires — and then proposes
the surrounding actions. Shortlists are scores over the real catalog; the
benchmark tables above are the aggregate truth.

*(The snippet builds rows with `DatasetBuilder.build_candidates` and ranks by
`classifier.predict_proba` — the same faithful path the benchmark uses.)*

### 5.3 Per-decision (span) benchmark — the state-aware view

The same pipeline also models *each step*: given the state before decision
`k` (previous tools, last observation), rank the tool that should be called
next.

```bash
python scripts/run_validation.py --config configs/validation.yaml \
    --level span --seeds 1 --output-dir models/validation/span
```

Real output (decision = next-tool correct; R=1):

```
method          r_precision   recall_at_3    mrr     (ms/decision)
model_state        0.575         0.885      0.731       7.96
model_nostate      0.309         0.677      0.542       5.60      (no state = ablation)
dsr_e5             0.207         0.482      0.372      17.25
bm25               0.110         0.450      0.348      12.24
random             0.012         0.033      0.056
```

Here the **state features** are the story: adding the per-step state
(`previous_tools` / `last_observation` / step index) roughly **doubles**
next-tool accuracy over the state-free control (`0.575 vs 0.309`, significant)
— the same lever the source paper isolates. (The `repeat_prev` row is a
trace-pattern baseline, not a real model; see the code comment.)

## 6. Going further

- **Calibration & selective/LLM fallback** (P4):
  `python scripts/run_validation.py --config configs/validation.yaml --level task --seeds 1 --calibrate --hybrid`
  — cross-fold ECE (≈0.45 → 0.15) and a risk–coverage curve, driven by a
  cached cost-bound LLM baseline (`scripts/run_llm_baseline.py`).
- **Dense baselines & latency** (P5): the DSR-E5 rows above come from the same
  table; per-method `ms/decision` is recorded for every run.
- **Docs**: per-module design notes are in the source comments (see §3 links);
  the P4 benchmark write-up lives in `docs/p4-benchmark-results.md`.

## 7. Reproduce everything

```bash
pip install -e ".[dev,embeddings]"

# example path
python scripts/build_dataset.py --trajectories data/example --config configs/example.yaml --output /tmp/example_ds
python scripts/train.py --dataset /tmp/example_ds --config configs/example.yaml --output /tmp/example_ds/model.pkl --folds 3

# AppWorld task-level + span-level benchmark
python scripts/run_validation.py --config configs/validation.yaml --level task --seeds 5 --output-dir models/validation/task
python scripts/run_validation.py --config configs/validation.yaml --level span  --seeds 1 --output-dir models/validation/span

# tests + lint
python -m pytest tests/ -q
python -m ruff check shortchain/ scripts/ tests/
```
