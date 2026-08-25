# Integration Guide

This guide explains how to integrate ShortChain into an existing agent system:
collect traces, train, then adapt tool selection at the decision point.

## 1. Collect

Production collection should not be hand-dumped JSONL. ShortChain ships an SDK
that enables the published OpenLLMetry instrumentations (LangChain, OpenAI
Agents, CrewAI, Agno, MCP, Anthropic, LiteLLM) and exports standard OTLP
traces to a small receiver that projects them onto the training schema
automatically.

```python
# pip install "shortchain[sdk,receiver]"
import os
from shortchain.sdk import ShortChain

ShortChain.init(
    api_key=os.environ["SHORTCHAIN_API_KEY"],
    app_name="support-agent",
    endpoint=os.environ.get("SHORTCHAIN_ENDPOINT", "http://127.0.0.1:4318"),
)
# existing LangChain / OpenAI / CrewAI / MCP code still runs — instrumentors
# are enabled automatically.
```

Mark each request's task root so the receiver can label success:

```python
def handle_request(req):
    ShortChain.set_task(task_id=req.id, intent=req.text, app_name="support-agent")
    try:
        result = agent.run(req.text)          # OpenLLMetry children nest under the root
        ShortChain.end_task(success=bool(result.ok))
        return result
    except Exception:
        ShortChain.end_task(success=False)
        raise
```

Start the receiver (workers are locked to 1; `data/runtime/` is secret
material):

```bash
shortchain receive --config configs/runtime.yaml
```

Then train on the collected traces:

```bash
python -m shortchain dataset \
    --trajectories data/runtime/trajectories.jsonl \
    --catalog data/runtime/catalog.json \
    --output data/datasets/runtime
python -m shortchain train \
    --dataset data/datasets/runtime --output models/shortchain.pkl
```

### Quality rules you must know

| Rule | Why |
|---|---|
| `set_task`…`end_task(success=...)` is **required** for trainable traces | Without it the success signal is unknown and the quality gate drops the trace (`success_status=unknown`). Training on unlabelled spans would teach the failing policy. |
| `end_task` must be called for human-in-the-loop / slow tools | The receiver's idle timeout is a safety net, not a contract. |
| Content tracing is ON by default | Prompts / tool results are needed to extract `intent` and `observation`. Only run the receiver on infrastructure you trust. |
| `data/runtime/trajectories.jsonl` is created with mode `0600` | Treat it like a dump of production logs. |

## 2. Adapt

Three deployment modes, from aggressive to conservative. All of them run the
same `InferenceEngine` at the decision point.

```python
from shortchain.model import InferenceEngine

# Load once at startup
engine = InferenceEngine(model_path="models/shortchain.pkl", top_k=5)
```

### Mode A: Replace

ShortChain returns the rank-1 tool directly. Use when the model is well trained
on familiar tools. $0 cost, ~1 ms latency per decision.

```python
def choose_tool(context: dict, tool_catalog: list[dict]) -> str:
    shortlist = engine.predict(context, tool_catalog, top_k=1)
    return shortlist[0][0]
```

| | |
|---|---|
| Latency | ~1 ms per decision |
| Cost | $0 per decision |
| Risk | Model may pick wrong for unseen patterns (mitigate with calibration) |

### Mode B: Shortlist + LLM

ShortChain narrows the catalog from N tools to 5; the LLM picks from that
shortlist. The LLM sees a handful of tool descriptions instead of the whole
catalog, so prompt size and cost drop.

```python
def choose_tool(context: dict, tool_catalog: list[dict], llm) -> str:
    shortlist = engine.predict(context, tool_catalog, top_k=5)
    return llm.select_tool(
        intent=context["intent"],
        candidates=[{"name": n, "score": s} for n, s in shortlist],
    )
```

| | |
|---|---|
| Latency | ~1 ms (ShortChain) + LLM on a small prompt |
| Cost | Reduced — the LLM sees 5 tools, not 100+ |
| Risk | Low — the LLM keeps final say |

### Mode C: Hybrid (confidence routing)

ShortChain decides when it is confident and defers to the LLM otherwise. This
is where the biggest savings sit: the agent stops paying for decisions the
learned model already knows how to make.

```python
HIGH_CONFIDENCE = 0.85
LOW_CONFIDENCE = 0.50


def choose_tool(context: dict, tool_catalog: list[dict], llm) -> str:
    shortlist = engine.predict(context, tool_catalog, top_k=5)
    top_tool, top_score = shortlist[0]

    if top_score >= HIGH_CONFIDENCE:
        return top_tool                       # no LLM call
    if top_score >= LOW_CONFIDENCE:
        return llm.select_tool(               # LLM picks from the shortlist
            intent=context["intent"],
            candidates=[{"name": n, "score": s} for n, s in shortlist],
        )
    return llm.select_tool(                   # low confidence → full LLM
        intent=context["intent"], candidates=tool_catalog,
    )
```

Confidence comes from the calibrated `shortchain.evaluation.calibration`
module; tune the thresholds on your own evaluation set.

| | |
|---|---|
| Latency | 1 ms (high confidence) to LLM latency (low confidence) |
| Cost | Reduced on average, depending on the confidence distribution |
| Risk | Lowest — graceful fallback to the LLM |

## 3. Context and candidates

`engine.predict()` takes a context dict and a candidate list:

```python
context = {
    "intent": "Send an email to John",
    "app_name": "gmail",
    "n_spans": 2,
    "previous_tools": "search_contacts",
    "last_thought": "Found John's email address",
}

candidates = [
    {"tool_name": "send_email", "tool_description": "Send an email to a recipient"},
    {"tool_name": "create_draft", "tool_description": "Create an email draft"},
    # ... the full tool catalog
]
```

## 4. Offline / benchmark data

For benchmarks and offline dumps, ShortChain also reads JSON/JSONL directly
(`examples/traces/` is a small example). Map non-standard field names in YAML
via `ingest.field_map`, then build / train / evaluate with the same commands
(`python -m shortchain dataset|train|evaluate`).

## 5. Programmatic pipeline

```python
from shortchain.config import load_config
from shortchain.ingest import load_trajectories
from shortchain.dataset import DatasetBuilder, GroupStratifiedSplitter
from shortchain.model import Trainer, InferenceEngine
from shortchain.evaluation import compute_metrics

cfg = load_config("my_config.yaml")

trajectories = load_trajectories("logs/", config=cfg.ingest)

builder = DatasetBuilder(
    config=cfg.dataset,
    features_config=cfg.features,
    negatives_config=cfg.negatives,
    tool_catalog={"send_email": "Send an email", "search_contacts": "…", },
)
df = builder.build(trajectories)

train_df, test_df = GroupStratifiedSplitter(cfg.splitter).train_test_split(df)

trainer = Trainer(
    classifier_config=cfg.classifier,
    splitter_config=cfg.splitter,
    eval_config=cfg.evaluation,
)
cv_results = trainer.train_with_cv(train_df)
trainer.train_final(train_df, save_path="models/shortchain.pkl")

y_proba = clf.predict_proba(test_df.drop(columns=["label"]))
metrics = compute_metrics(
    y_true=test_df["label"].values,
    y_proba=y_proba,
    X_val=test_df.drop(columns=["label"]),
)
```

## Retraining and cold start

- Retrain when tools are added, behavior changes, or metrics degrade. It takes
  seconds and can be automated with the same `dataset` / `train` commands.
- Cold start: run with the LLM alone for 1–2 weeks, collect traces, train the
  first model from ~50 successful trajectories, deploy in **Mode C** with
  conservative thresholds, then tighten as the model improves.