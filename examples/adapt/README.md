# Adapt

Use `InferenceEngine` to rank candidate tools at a decision point, then decide
how to route: replace fully, shortlist for the LLM, or use a confidence-based
hybrid.

```python
from shortchain.model import InferenceEngine

engine = InferenceEngine(model_path="/tmp/sc-model.pkl", top_k=5)

shortlist = engine.predict(
    context={
        "intent": "Refund order 9921",
        "app_name": "support-agent",
        "n_spans": 2,
        "previous_tools": "lookup_order",
    },
    candidates=[
        {"tool_name": "refund_order", "tool_description": "Issue a refund"},
        {"tool_name": "lookup_order", "tool_description": "Look up an order"},
        # ... the full tool catalog
    ],
    top_k=5,
)
# [("refund_order", 0.94), ("lookup_order", 0.61), ...]  — in ~1 ms
```

Three ways to use the shortlist:

1. **Replace** — use `shortlist[0][0]` directly, no LLM call.
2. **Shortlist** — pass the shortlist to the LLM to make the final pick.
3. **Hybrid** — route to the LLM only when calibrated top-1 confidence is low.

See `docs/integration.md` for the full adapter code for each mode, and
`shortchain/evaluation/calibration.py` for turning raw scores into actionable
confidence thresholds.