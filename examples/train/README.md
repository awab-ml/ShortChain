# Train

Build a pointwise dataset from trajectories, train a classifier, and evaluate.

On the shipped sample traces (no agent or receiver needed):

```bash
python -m shortchain dataset \
    --trajectories examples/traces/ \
    --config examples/configs/example.yaml \
    --output /tmp/sc-ds

python -m shortchain train --dataset /tmp/sc-ds --output /tmp/sc-model.pkl

python -m shortchain evaluate \
    --model /tmp/sc-model.pkl \
    --dataset /tmp/sc-ds/test.csv
```

What each step does:

- **dataset** — each trajectory becomes `(context, tool, label)` rows: positive
  rows for tools the agent used, 3:1 negative rows sampled from the catalog,
  split at the task level (no task leaks across the split).
- **train** — 5-fold group-aware cross-validation, then a final XGBoost model
  over the `FeaturePipeline` encoding.
- **evaluate** — R-Precision, Recall@k, and classification metrics on the
  held-out test rows.

To train on traces collected by the receiver instead:

```bash
python -m shortchain dataset \
    --trajectories data/runtime/trajectories.jsonl \
    --catalog data/runtime/catalog.json \
    --output data/datasets/runtime
python -m shortchain train --dataset data/datasets/runtime --output models/shortchain.pkl
```