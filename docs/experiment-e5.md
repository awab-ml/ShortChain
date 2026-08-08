# Experiment — E5 text encoding vs TF-IDF (context/tool features)

Branch: `experiment/e5-context` · Status: not merged (results under review)

## Setup

Swap the model's text features from TF-IDF to E5 embeddings
(`configs/experiment_e5.yaml` → `features.text_encoder: "e5-small"`,
`intfloat/e5-small-v2`). Every textual column the `FeaturePipeline` encodes
(intent, previous tools, last thought, tool name/description, history) becomes
a 384-d embedding (2707 features/fold vs 793 with TF-IDF). Baselines
(random / popularity / BM25 / DSR-E5), pooling, and the leakage invariants are
unchanged, so the comparison is model(E5-text) vs model(TF-IDF-text) on
identical rows.

Task-level benchmark, AppWorld (5-fold, single seed, catalog-wide pool):

| Model | P@R | R@3 | R@7 | R@9 | MRR | nDCG@5 | ms/decision |
|---|---|---|---|---|---|---|---|
| **TF-IDF** (current default) | **0.852** | 0.605 | **0.924** | **0.936** | 1.000 | **0.903** | **6.9** |
| E5-text (experiment) | 0.849 | 0.624 | 0.887 | 0.922 | 1.000 | 0.898 | ≈16,000 |

## Conclusion

- **No accuracy gain.** R-Precision is unchanged within noise (0.849 vs 0.852);
  E5 improves R@3 (+0.02) at the expense of R@7/R@9 (−0.03); MRR identical
  (both always pick a relevant top-1). Nothing here is close to significant.
- **Impractical cost.** Treating every text feature as a per-column E5
  embedding makes the model **≈2,300× slower per decision** (~16 s vs ~7 ms)
  and **~130× slower to train** (≈44 min vs ≈20 s for a single seed) —
  defeating the ~1ms decision engine this system exists to provide.
- **Recommendation:** keep TF-IDF as the default text encoding. The strong
  tool-selection signal comes from training on the traces + the structured
  features, not from a fancier text encoder. (The separate DSR-E5 *retrieval*
  baseline remains a meaningful comparison point; making E5 a *feature* does
  not pay off here.)
