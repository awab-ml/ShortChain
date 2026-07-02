# Overview

## The Problem

Modern AI agent systems (AutoGPT, LangChain agents, enterprise assistants) rely on a large language model (LLM) at every decision point: *"Which tool should I call next?"* Each LLM call costs \$0.01–\$0.10 and adds 500–2000ms of latency. For agents that make dozens of decisions per task across thousands of daily requests, this creates a "double burden" of high cost and high latency.

But this is a **ranking problem**, not a generation problem. Given a fixed catalog of tools and the current execution context, the agent just needs to rank candidates — it doesn't need to *generate* a tool name from scratch.

## The Solution

**ShortChain** trains a lightweight tabular classifier (~1ms inference) on the agent's own successful execution traces. At runtime, it replaces or augments LLM calls with a single forward pass through the classifier, achieving:

- **~95% latency reduction**: ~1ms vs ~500–2000ms per decision
- **~85–91% cost reduction**: no API calls for tool selection
- **Comparable accuracy**: maintained task-level success rates

## How It Works

```
Phase 1: TRAINING (offline, one-time)
──────────────────────────────────────
Agent execution logs → Trajectories → (context, tool, label) pairs → XGBoost classifier

Phase 2: INFERENCE (online, per-decision)
─────────────────────────────────────────
Current context + candidate tools → Classifier → Ranked shortlist (in ~1ms)
```

### Key Concept: Pointwise Reduction

The core technique transforms a **ranking** problem into **binary classification**:

For each trajectory (a successful agent execution):
1. **Positive pairs** (label=1): For every tool the agent actually used, create a row `(context, tool_name, 1)`
2. **Negative pairs** (label=0): Sample tools from the catalog that were *not* used, create rows `(context, tool_name, 0)`

The classifier learns to predict: *"Given this context, is this tool likely to be used?"*

At inference time, score all candidate tools and return the top-K.

## Core Concepts

ShortChain introduces three components:

| Component | Description | Status in This Repo |
|---|---|---|
| **TabSchema** | LLM-driven extraction of schema, state, and dependency features from trajectories | Planned (Phase 3A) |
| **TabSynth** | Synthetic data generation for rare tool combinations | Planned (Phase 3B) |
| **TabHead** | Lightweight classifier that replaces LLM decisions | ✅ Implemented |

Results on the AppWorld benchmark (457 APIs across 9 apps), achieving ~95% latency reduction while maintaining task-level success.

## When to Use ShortChain

**Good fit:**
- Your agent has a fixed or slowly-changing tool catalog
- You have successful execution logs (even a few dozen trajectories)
- Tool selection decisions are repetitive across tasks
- You need lower latency or lower cost at inference time

**Not a good fit:**
- Your tool catalog changes every request
- You have zero historical execution data (cold start)
- Tool selection requires deep multi-span reasoning that can't be captured in features
