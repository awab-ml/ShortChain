# Examples

Three short demos built on the shipped sample traces in `examples/traces/`
(no agent, receiver, or API key required).

| Directory | What it shows |
| --- | --- |
| [`collect/`](collect/README.md) | Instrument with the SDK and run the receiver |
| [`train/`](train/README.md) | Build a dataset, train, and evaluate on the sample traces |
| [`adapt/`](adapt/README.md) | Use `InferenceEngine` at a decision point (replace / shortlist / hybrid) |

Quick pipeline on the sample traces:

```bash
python -m shortchain dataset \
    --trajectories examples/traces/ \
    --config examples/configs/example.yaml \
    --output /tmp/sc-ds
python -m shortchain train --dataset /tmp/sc-ds --output /tmp/sc-model.pkl
python -m shortchain evaluate --model /tmp/sc-model.pkl --dataset /tmp/sc-ds/test.csv
```

`benchmarks/validation.yaml` is the AppWorld validation harness configuration;
it is an experiment config, not a product default.