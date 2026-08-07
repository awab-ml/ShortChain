# P4 — Calibration, Selective/LLM-Fallback & Cost-Bound LLM Baselines (AppWorld, task level)

> Status: complete · Honest, 5-seed, leak-free evaluation on AppWorld (`test_normal`, 57 tasks, Gemini 3 Flash agent traces).

## Objective

Show that ShortChain is deployable as a **calibrated, cost-aware engine**: handle routine tool-selection decisions locally (sub-millisecond, ~$0) and only escalate when confidence is low — and quantify, with real numbers, how it compares against cost-bound LLM tool shortlisters (the generative component it is meant to replace).

## Setup

- **Local model**: ShortChain (XGBoost, pointwise ranking) with the full P0–P3 feature set — frozen train corpus stats, task-level 5-fold, 5 training seeds, catalog-wide candidate pools (91 tools). Task headline R-Precision **0.856**.
- **Decision under study**: *full-shortlist correctness* — the top-R predicted shortlist is exactly the relevant set (**P@R == 1**, R = |relevant|). Confidence = the highest predicted probability inside the top-R shortlist. (Top-1 is already always a relevant tool — MRR = 1.0 — so it saturates and leaves nothing to "advise" on.)
- **Calibration**: group-aware **cross-fold** OOF fitting — each fold's calibrator is fit on the *other* folds' out-of-fold (confidence, outcome) pairs and applied to the held-out fold (no leakage), pinned seed. Method: Platt.
- **LLM baselines** (cost-bound, out-of-band, cached; inputs = intent + the `function_calling` tool definitions the agent sees; no labels/solutions):
  - `deepseek/deepseek-v4-flash-0731` — OpenRouter, interactive route. 57 tasks, **~$0.03 total**.
  - `google/gemini-3.6-flash:batch` — OpenRouter **async Batch API**. 57 tasks, **~$0.028 total** (batch billing).
  - API key read from `OPENROUTER_API_KEY` at runtime; never written to disk.

## Results — ShortChain vs cost-bound LLM baselines

| Metric | **ShortChain (local)** | **deepseek-v4-flash-0731** | **gemini-3.6-flash (batch)** |
|---|---|---|---|
| Tool R-Precision | **0.856** | 0.471 | 0.650 |
| MRR | **1.000** (top-1 always relevant) | 0.604 | 0.974 |
| Recall@1 | — | 0.140 | 0.218 |
| Recall@5 | — | 0.487 | 0.659 |
| Recall@9 | — | 0.535 | 0.773 |
| Full-shortlist correct (**P@R == 1**) | **52.6%** | 17.5% | 0.0% |
| Latency / decision | **0.3 ms** | ≈ 20.9 s | async (batch) |
| Cost / decision | **$2.0×10⁻⁷** | $5.3×10⁻⁴ | $4.9×10⁻⁴ |

Cost/throughput gap vs interactive deepseek: **~2,600× cheaper**, **~70,000× lower latency**.

## Calibration

- Expected Calibration Error (ECE) for the P@R==1 decision: **0.451 → 0.150** (Platt). Confidence is calibrated well enough to be a trustworthy deferral signal.

## Selective / hybrid honesty

- **Coverage at LLM-parity risk: 96.5%** — we can resolve ~96.5% of decisions locally and still be no worse than an LLM-only policy.
- **But deferring to the LLM does not reduce risk at task level**: both baselines are a *weaker* oracle here (full-shortlist completeness: deepseek 17.5%, gemini 0.0%), so escalation only adds cost. The honest operating point at task level is **local-only**.
- The paper's justification for LLM fallback targets *span-level novel compositions* — out of this task-level scope (a natural follow-up in P4.1/span mode).

## Dense retrieval baseline (P5 — DSR-E5, zero-shot)

Zero-shot dense semantic retrieval (`intfloat/e5-small-v2`, E5 asymmetric `query:`/`passage:` prefixes) added as a first-class baseline using the same candidate rows, metrics, and significance/CI machinery.

| Metric | **ShortChain (local)** | **DSR-E5 (zero-shot)** | BM25 |
|---|---|---|---|
| Tool R-Precision | **0.856** | 0.375 | 0.389 |
| MRR | **1.000** | 0.744 | 0.782 |
| Recall@5 | — | 0.487 | 0.522 |
| Recall@9 | 0.935 | 0.469 | 0.522 |

- **Honest finding:** zero-shot E5 ≈ BM25 on this data (0.375 vs 0.389) — the descriptive tool names/argument hints already give lexical matching most of the dense signal, and neither approaches the trained model (significantly worse, † [+0.41, +0.56]).
- **Latency / decision** (measured in the same harness): ShortChain **6.2 ms** < BM25 13.4 ms < DSR-E5 19.1 ms — the local classifier is the fastest *and* the most accurate.
- Note: the harness pins `OMP_NUM_THREADS=1` / `KMP_DUPLICATE_LIB_OK=TRUE` so E5 (torch) and XGBoost coexist without segfaulting; absolute latencies are higher than the README's single-request figure because they include per-decision feature encoding, but the ordering is the honest comparison.

## Takeaways

1. The cheap local model **beats cost-bound LLM baselines** on task-level tool shortlisting — in quality (R-Precision, MRR, and especially full shortlist completeness) and by orders of magnitude in cost/latency.
2. Confidence **calibration works** (ECE 0.451 → 0.150), so thresholds are meaningful.
3. **Selective deferral is unnecessary at task level** on this data — the pipeline's value is local; fallback should be evaluated (later) at the decision/span level where out-of-distribution risk actually exists.
4. **Dense retrieval (zero-shot E5) adds no advantage over a cheap TF-IDF/BM25 baseline** here and both remain far below the trained ShortChain model — training on the traces is what carries the signal, not a better text encoder.

## Reproduce

```bash
# (0) Dense retrieval baseline (E5, downloaded & cached on first run)
python scripts/run_validation.py --config configs/validation.yaml --level task --seeds 5 \
    --output-dir models/validation/p5

# (1) Cost-bound LLM baselines (cached; replace key via env)
OPENROUTER_API_KEY=... python scripts/run_llm_baseline.py --config configs/validation.yaml \
    --model "deepseek/deepseek-v4-flash-0731" --output models/validation/llm_results.json
OPENROUTER_API_KEY=... python scripts/run_llm_baseline.py --config configs/validation.yaml \
    --model "google/gemini-3.6-flash:batch" --output models/validation/llm_results_gemini_batch.json

# (2) Calibration + hybrid with a given LLM cache
python scripts/run_validation.py --config configs/validation.yaml --level task --seeds 5 \
    --calibrate --hybrid [--llm-results models/validation/llm_results_gemini_batch.json] \
    --output-dir models/validation/p4
```
