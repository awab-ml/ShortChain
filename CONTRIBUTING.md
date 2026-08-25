# Contributing to ShortChain

Thanks for your interest in ShortChain! This is a small, focused project and
we want contributions to be easy. This guide covers how to set up your
environment, run the tests, and what we look for in a good change.

## Welcome

ShortChain is an observability backend for LLM applications. It collects
OpenTelemetry traces, learns the execution patterns in them, and adapts tool
selection to cut cost and latency. You do not need to know any of that to
contribute — the code is organized so each part is self-contained.

## Prerequisites

- Python 3.10+
- `pip` (or `uv`)

```bash
# Core package + dev tools (pytest, ruff)
pip install -e ".[dev]"

# Optional extras for telemetry / SDK / receiver tests
pip install -e ".[sdk,receiver]"
```

The core training tests run with just `.[dev]`. Telemetry and ingest tests
that touch OpenTelemetry skip gracefully when the `otel` extras are not
installed.

## Running tests

```bash
# Everything
pytest tests/ -q

# One operation's suite (tests mirror the package layout)
pytest tests/ingest/       # trace normalization / OTEL projection
pytest tests/telemetry/    # SDK, receiver, assembler, task root
pytest tests/dataset/      # pointwise dataset construction
pytest tests/model/        # classifier + trainer
pytest tests/evaluation/   # ranking metrics, calibration, hybrid
pytest tests/features/     # feature builders + encoders
pytest tests/adapters/     # benchmark / source adapters

# Lint
ruff check shortchain/ tests/ scripts/
```

## Running the example

The repo ships sample trajectories under `examples/traces/` so you can run the
whole pipeline without an agent or a receiver:

```bash
python -m shortchain dataset --trajectories examples/traces \
    --config examples/configs/example.yaml --output /tmp/sc-ds
python -m shortchain train --dataset /tmp/sc-ds --output /tmp/sc-model.pkl
python -m shortchain evaluate --model /tmp/sc-model.pkl \
    --dataset /tmp/sc-ds/test.csv
```

See `examples/README.md` for the collect / train / adapt demos.

## Making a change

1. Create a branch off `main` (e.g. `fix/assembler-eviction`).
2. Make a small, focused change. Prefer several small PRs over one large one.
3. Add or update tests that cover the change.
4. Run the relevant suite and `ruff check`.
5. Open a pull request with a short description of what and why.

### What we look for

- Tests pass and cover the change.
- No regressions: `pytest tests/ -q` stays green.
- Ruff is clean.
- No generated artifacts are committed: `data/`, `models/`, and experiment
  outputs stay gitignored. Experiment write-ups do not belong in `docs/`.

## Code layout

The package is named by operation, and `tests/` mirrors it one-to-one:

```
shortchain/
├── telemetry/    # SDK, instrumentors, OTLP receiver, assembler
├── ingest/       # Trajectory / Span schema, loaders, OTEL projection
├── features/     # context / tool / corpus-stat encoders
├── dataset/      # pointwise (context, tool, label) construction
├── model/        # classifier, trainer, inference engine
├── evaluation/   # ranking metrics, calibration, hybrid fallback
└── adapters/     # optional source / benchmark bindings
```

See [docs/architecture.md](docs/architecture.md) for the data flow.

## Code of conduct

Be kind, be specific, and assume good intent. Disagreement about design is
welcome; personal attacks are not. This project follows the community standard
of respectful, constructive collaboration.