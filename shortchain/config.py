"""Configuration management for ShortChain.

Loads YAML configs and exposes them as validated Pydantic models.
Every module reads its config slice from the root `ShortChainConfig`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Config sub-models
# ---------------------------------------------------------------------------

class FieldMapConfig(BaseModel):
    """Maps external log field names to internal ShortChain schema names."""

    task_id: str = "task_id"
    intent: str = "intent"
    spans: str = "spans"
    success: str = "success"
    agent_name: str = "agent_name"
    action: str = "action"
    observation: str = "observation"
    thoughts: str = "thoughts"


class IngestConfig(BaseModel):
    """Trajectory ingestion settings."""

    format: str = "jsonl"
    success_only: bool = True
    field_map: FieldMapConfig = Field(default_factory=FieldMapConfig)


class ProjectionConfig(BaseModel):
    """OTEL/OpenLLMetry → Trajectory projection settings.

    Single source of truth: imported by ``shortchain/ingest/otel.py``,
    ``shortchain/ingest/quality.py``, and ``RuntimeConfig``. Do not redeclare
    it in those modules.
    """

    intent_strategy: str = "first_user"      # first_user | last_user_before_tools
    accept_gen_ai_task_id: bool = False
    accept_task_status: bool = False
    success_tools: list[str] = Field(default_factory=list)
    drop_tools: list[str] = Field(default_factory=list)
    max_observation_chars: int = 2000
    max_thought_chars: int = 2000
    require_intent: bool = True
    require_tool_spans: bool = True
    require_known_success: bool = True       # matches success_only training
    max_spans: int = 200                     # training-side cap after project


class FeaturesConfig(BaseModel):
    """Feature pipeline settings."""

    text_encoder: str = "tfidf"                     # tfidf | e5-small | auto
    e5_model_name: str = "intfloat/e5-small-v2"
    tfidf_max_features: int = 5000
    context_fields: list[str] = Field(
        default_factory=lambda: [
            "intent",
            "app_name",
            "n_spans",
            "previous_tools",
            "last_thought",
        ]
    )
    include_state_features: bool = True
    include_dependency_features: bool = True


class NegativeSamplingConfig(BaseModel):
    """Negative sampling strategy settings."""

    strategy: str = "random"                        # random | hard | mixed
    hard_negative_ratio: float = 0.5                # fraction of hard negs in mixed
    same_app_weight: float = 0.4
    co_usage_weight: float = 0.3
    similarity_weight: float = 0.3
    random_state: int | None = None                 # seed for reproducibility


class DatasetConfig(BaseModel):
    """Dataset construction settings."""

    mode: str = "intent"  # "intent" | "span"
    negative_ratio: int = 3
    # --- Kept for backward compatibility; prefer FeaturesConfig/NegativeSamplingConfig ---
    negative_strategy: str = "random"
    context_fields: list[str] = Field(
        default_factory=lambda: [
            "intent",
            "app_name",
            "n_spans",
            "previous_tools",
            "last_thought",
        ]
    )
    text_encoder: str = "tfidf"
    tfidf_max_features: int = 5000


class SplitterConfig(BaseModel):
    """Train/test splitting settings."""

    n_folds: int = 10
    test_size: float = 0.2
    group_by: str = "task_id"
    stratify_by: list[str] = Field(default_factory=lambda: ["app_name"])


class XGBoostParams(BaseModel):
    """XGBoost hyper-parameters."""

    n_estimators: int = 300 
    max_depth: int = 8
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    eval_metric: str = "logloss"
    early_stopping_rounds: int = 20


class RandomForestParams(BaseModel):
    """Random-forest hyper-parameters."""

    n_estimators: int = 200
    max_depth: int = 12
    min_samples_leaf: int = 5


class LogisticParams(BaseModel):
    """Logistic-regression hyper-parameters."""

    C: float = 1.0
    max_iter: int = 1000


class ClassifierConfig(BaseModel):
    """Classifier training settings."""

    model_type: str = "xgboost"
    xgboost: XGBoostParams = Field(default_factory=XGBoostParams)
    random_forest: RandomForestParams = Field(default_factory=RandomForestParams)
    logistic: LogisticParams = Field(default_factory=LogisticParams)


class InferenceConfig(BaseModel):
    """Inference settings."""

    top_k: int = 7
    confidence_threshold: float = 0.5


class EvaluationConfig(BaseModel):
    """Evaluation settings."""

    k_values: list[int] = Field(default_factory=lambda: [3, 5, 7, 9])
    metrics: list[str] = Field(
        default_factory=lambda: [
            "r_precision",
            "recall_at_k",
            "accuracy",
            "f1",
        ]
    )



class RuntimeConfig(BaseModel):
    """OTLP HTTP receiver / trace assembler settings."""

    bind: str = "127.0.0.1:4318"
    output: str = "data/runtime/trajectories.jsonl"
    idle_timeout_s: float = 30.0
    settle_timeout_s: float = 2.0
    max_trace_age_s: float = 300.0
    max_inflight_traces: int = 512
    max_spans_in: int = 500
    max_body_bytes: int = 16_777_216
    workers: int = 1
    require_success_true: bool = True   # ingest-level gate (success_only)
    projection: ProjectionConfig = Field(default_factory=ProjectionConfig)


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class ShortChainConfig(BaseModel):
    """Root configuration for the ShortChain pipeline."""

    ingest: IngestConfig = Field(default_factory=IngestConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    negatives: NegativeSamplingConfig = Field(default_factory=NegativeSamplingConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    splitter: SplitterConfig = Field(default_factory=SplitterConfig)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base* (returns new dict)."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path | None = None) -> ShortChainConfig:
    """Load and validate a ShortChain configuration.

    Parameters
    ----------
    config_path
        Path to a YAML config file.  If ``None``, the shipped
        ``configs/default.yaml`` is used.  If a path is provided, it is
        deep-merged on top of the defaults so that you only need to
        specify overrides.

    Returns
    -------
    ShortChainConfig
        Validated configuration object.
    """
    # Load defaults
    base: dict[str, Any] = {}
    if _DEFAULT_CONFIG_PATH.exists():
        with open(_DEFAULT_CONFIG_PATH) as f:
            base = yaml.safe_load(f) or {}

    # Merge overrides if provided
    if config_path is not None:
        with open(config_path) as f:
            overrides = yaml.safe_load(f) or {}
        base = _deep_merge(base, overrides)

    return ShortChainConfig.model_validate(base)
