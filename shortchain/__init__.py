"""ShortChain — replace expensive LLM decision components with compact classifiers.

ShortChain learns a lightweight classifier from agent execution traces and uses
it at every decision point where an LLM would otherwise *select* from a closed
set of options (tools, applications, …).

Core techniques
---------------
- **Pointwise reduction** — each decision becomes (context, candidate tool,
  correct/incorrect) rows; the classifier scores and we rank by probability.
- **Behavior-grounded features** — context (intent/state), tool (schema),
  and corpus statistics are derived from successful traces, not hand-tuned.
- **Leak-free evaluation** — corpus statistics are frozen on the training set,
  splits group by task, and per-decision context never looks ahead.
- **Faithful baselines** — random, popularity, BM25, DSR-E5, and cost-bound
  LLMs are measured on identical candidate rows with bootstrap CIs.

The pipeline is agent-agnostic: it binds only to execution traces and typed
tool schemas, so the same core serves many integrations (see ``integrations/``).
"""

__version__ = "0.0.2"
