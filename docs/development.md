# Development Guide

## Setup

```bash
# Clone and install in development mode
git clone https://github.com/awab-ml/ShortChain.git
cd ShortChain
pip install -e ".[dev]"

# Optional: dense embeddings
pip install -e ".[embeddings]"
```

### Dependencies

**Core** (required):
| Package | Version | Purpose |
|---|---|---|
| pandas | ≥ 2.0 | DataFrames for dataset and features |
| numpy | ≥ 1.24 | Numeric operations |
| scikit-learn | ≥ 1.3 | TF-IDF, LabelEncoder, splitting, metrics |
| xgboost | ≥ 2.0 | Default classifier backend |
| pydantic | ≥ 2.0 | Config models and data schemas |
| pyyaml | ≥ 6.0 | YAML config loading |
| rich | ≥ 13.0 | Structured console logging |

**Optional**:
| Package | Extra | Purpose |
|---|---|---|
| sentence-transformers | `.[embeddings]` | E5-small dense text encoding |
| pytest | `.[dev]` | Test runner |
| pytest-cov | `.[dev]` | Coverage reporting |
| ruff | `.[dev]` | Linting |

---

## Testing

### Running Tests

```bash
# Run all tests (100 tests, ~2 seconds)
pytest tests/ -v

# Run a specific test file
pytest tests/test_features.py -v
pytest tests/test_negatives.py -v
pytest tests/test_classifier.py -v
pytest tests/test_dataset.py -v
pytest tests/test_ingest.py -v
pytest tests/test_metrics.py -v

# Run a specific test
pytest tests/test_features.py::TestContextFeatureBuilder::test_build_basic -v

# With coverage
pytest tests/ --cov=shortchain --cov-report=term-missing
```

### Test Structure

```
tests/
├── test_features.py      # 32 tests — pipeline, encoders, context, tool, stats
├── test_negatives.py      # 14 tests — random, hard, mixed samplers
├── test_ingest.py         # 20 tests — schema, loader, field mapping
├── test_dataset.py        #  8 tests — builder, pairs, catalog
├── test_classifier.py     #  6 tests — fit, predict, save/load, backends
└── test_metrics.py        #  8+ tests — r_precision, recall_at_k, compute_metrics
```

### Test Conventions

- All tests run without external dependencies (no API keys, no downloads)
- Tests create fixtures inline — no shared test data files
- Each test is independent and can run in isolation
- Deterministic seeding where randomness is involved
- Naming: `test_<module>.py` → `Test<Class>` → `test_<behavior>`

### Writing New Tests

```python
# tests/test_my_module.py
import pytest
from shortchain.ingest.schema import Step, Trajectory


def _make_trajectory(**overrides) -> Trajectory:
    """Helper to create test trajectories."""
    defaults = {
        "task_id": "test_001",
        "intent": "Test intent",
        "app_name": "testapp",
        "success": True,
        "steps": [
            Step(action="tool_a", observation="result_a"),
            Step(action="tool_b", observation="result_b"),
        ],
    }
    defaults.update(overrides)
    return Trajectory(**defaults)


class TestMyFeature:
    def test_basic_behavior(self):
        traj = _make_trajectory()
        # ...assert expected behavior...
    
    def test_edge_case_empty_input(self):
        traj = _make_trajectory(steps=[])
        # ...assert graceful handling...
```

---

## Linting

```bash
# Check style
ruff check shortchain/ tests/

# Auto-fix
ruff check --fix shortchain/ tests/
```

Configuration is in `pyproject.toml`:
```toml
[tool.ruff]
target-version = "py310"
line-length = 100
```

---

## Adding a New Feature Column

To add a new feature to the pipeline:

### 1. Add to the appropriate builder

**Context feature** (depends on trajectory state):
```python
# shortchain/features/context.py → ContextFeatureBuilder.build()
features["my_new_feature"] = self._compute_my_feature(traj)
```

**Tool feature** (depends on the candidate tool):
```python
# shortchain/features/tool.py → ToolFeatureBuilder.build()
features["my_tool_feature"] = self._compute_tool_feature(tool_name)
```

### 2. Register in the pipeline

Add the column name to the appropriate list in `shortchain/features/pipeline.py`:

```python
# For numeric features:
_NUM_COLS = [..., "my_new_feature"]

# For boolean features:
_BOOL_COLS = [..., "my_bool_feature"]

# For text features:
_TEXT_COLS = [..., "my_text_feature"]

# For categorical features:
_CAT_COLS = [..., "my_cat_feature"]
```

### 3. Add tests

```python
# tests/test_features.py
def test_my_new_feature(self):
    traj = _make_trajectory()
    builder = ContextFeatureBuilder()
    features = builder.build(traj)
    assert "my_new_feature" in features
    assert isinstance(features["my_new_feature"], (int, float))
```

### 4. Update the default config (if configurable)

```yaml
# configs/default.yaml
features:
  include_my_feature: true
```

---

## Adding a New Classifier Backend

### 1. Add config model

```python
# shortchain/config.py
class MyModelParams(BaseModel):
    n_estimators: int = 100
    # ...

class ClassifierConfig(BaseModel):
    # ...
    my_model: MyModelParams = Field(default_factory=MyModelParams)
```

### 2. Add to the model factory

```python
# shortchain/head/classifier.py → _create_model()
elif model_type == "my_model":
    from my_library import MyClassifier
    return MyClassifier(**self.config.my_model.model_dump())
```

### 3. Update the YAML config

```yaml
# configs/default.yaml
classifier:
  model_type: "my_model"
  my_model:
    n_estimators: 100
```

---

## Adding a New Negative Sampling Strategy

### 1. Create the sampler class

```python
# shortchain/dataset/negatives.py
class MyCustomSampler(NegativeSampler):
    def sample(self, positive_tools, app_name, n):
        pool = [t for t in self.catalog if t not in positive_tools]
        # ...your custom logic...
        return selected[:n]
```

### 2. Register in the factory

```python
# shortchain/dataset/negatives.py → create_sampler()
elif strategy == "my_custom":
    return MyCustomSampler(catalog, corpus_stats, config.random_state)
```

### 3. Add tests

```python
# tests/test_negatives.py
class TestMyCustomSampler:
    def test_basic_sampling(self):
        sampler = MyCustomSampler(catalog={"a": "", "b": "", "c": ""})
        result = sampler.sample(positive_tools={"a"}, app_name="test", n=2)
        assert len(result) == 2
        assert "a" not in result
```

---

## Adding a New Ingestion Format

### 1. Create a loader class

```python
# shortchain/ingest/my_format.py
from shortchain.ingest.base import TrajectoryLoader

class CSVTrajectoryLoader(TrajectoryLoader):
    def load(self, path: str | Path) -> list[Trajectory]:
        # Read CSV, map fields, return Trajectory objects
        ...
```

### 2. Register in the loader module

```python
# shortchain/ingest/__init__.py
from shortchain.ingest.my_format import CSVTrajectoryLoader
```

---

## End-to-End Pipeline Test

Run the full pipeline manually to verify everything works:

```bash
# Build dataset
python scripts/build_dataset.py \
    --trajectories data/example/ \
    --output data/datasets/

# Train with cross-validation
python scripts/train.py \
    --dataset data/datasets/ \
    --model xgboost \
    --folds 3 \
    --output models/shortchain.pkl

# Evaluate
python scripts/evaluate.py \
    --model models/shortchain.pkl \
    --dataset data/datasets/test.csv \
    --output models/eval_results.json
```

Expected: all commands succeed, R-Precision ≥ 0.85, Recall@5 = 1.0 on example data.

---

## Versioning and Releases

- **Package version**: defined in `pyproject.toml` and `shortchain/__init__.py`
- **Model format version**: `v2` (Phase 2). Stored in pickle alongside the model.
- **Backward compatibility**: v1 models (Phase 1) are loaded via the legacy adapter in `classifier.py`

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 — MVP | Core pipeline: ingest → train → evaluate → inference | ✅ Complete |
| 2 — Features | Modular FeaturePipeline, negative sampling, encoders | ✅ Complete |
| 3A — TabSchema | LLM-driven feature extraction (opt-in enhancement) | Planned |
| 3B — TabSynth | Synthetic data generation for rare patterns | Planned |
| 3C — Step-Level | Per-step decisions (experimental mode) | Planned |
| 4 — Benchmarks | AppWorld adapter, real-world evaluation | Planned |
