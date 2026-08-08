#!/usr/bin/env python3
"""ShortChain P0 validation on AppWorld (HALO Gemini traces).

Protocol (paper-aligned, leakage-free):
- Task-level N-fold split (GroupKFold by task_id).
- Train one ShortChain classifier per fold on the fold-train trajectories;
  CorpusStats are frozen on the TRAIN set and reused for evaluation rows
  (build_candidates never recomputes stats from evaluation data).
- Evaluation is faithful ranking over explicit candidate pools built from the
  train-side tool catalog: catalog-wide (all tools) and app-scoped (the
  task's apps).
- Baselines scored on the SAME candidate rows: random, popularity (train-side
  tool frequency), BM25 (TF-IDF cosine, fit on catalog documents only), and
  DSR-E5 (zero-shot dense retrieval).
- Per-task metric values are pooled over test folds and macro-averaged with
  paired-bootstrap 95% CIs; model vs. baseline contrasts get Holm-Bonferroni
  control within each metric. Per-method latency is measured per decision.

Techniques worth knowing
------------------------
- Every table row is anchored by the RANDOM baseline so a number is never read
  in isolation (tiny candidate pools can make even mediocre rankings look
  strong when random is high).
- Dense (E5) baselines load torch; XGBoost loads libomp — together they can
  segfault on duplicate OpenMP runtimes, so the harness pins
  ``KMP_DUPLICATE_LIB_OK`` and ``OMP_NUM_THREADS`` before either loads.
- ``--calibrate``/``--hybrid`` add the P4 analysis: cross-fold, group-aware
  calibration (ECE) and selective / LLM-fallback metrics driven by a cached,
  cost-bound LLM baseline (see ``run_llm_baseline.py``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any
from collections import defaultdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# torch (sentence-transformers) and XGBoost both dispatch OpenMP; loading E5 then
# training XGBoost in one process can segfault on duplicate libomp. These must be
# set before either native runtime initializes.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupKFold

from shortchain.config import ClassifierConfig, DatasetConfig, FeaturesConfig, NegativeSamplingConfig
from shortchain.dataset.builder import DatasetBuilder
from shortchain.evaluation.calibration import create_calibrator, expected_calibration_error
from shortchain.evaluation.hybrid import (
    area_under_risk_coverage,
    coverage_at_target_risk,
    coverage_risk_curve,
    hybrid_curve,
)
from shortchain.evaluation.metrics import task_level_scores
from shortchain.evaluation.statistics import (
    bootstrap_mean_ci,
    holm_bonferroni,
    paired_bootstrap_p_and_ci,
)
from shortchain.head.classifier import ShortChainClassifier
from shortchain.integrations.halo import (
    catalog_app_index,
    load_appworld_traces,
    reconstruct_catalog,
)
from shortchain.integrations.appworld_api import build_catalog_and_schemas
from shortchain.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _deep_merge_dict(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (nested dicts are merged)."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_validation_config(path: str | Path | None) -> dict:
    """Load the validation config, deep-merging over ``configs/validation.yaml``.

    Experiment configs can therefore override only what they change
    (e.g. ``features.text_encoder``) and inherit the rest of the protocol
    settings.
    """
    default = Path(__file__).resolve().parent.parent / "configs" / "validation.yaml"
    with open(default) as f:
        base = yaml.safe_load(f) or {}
    if path is None:
        return base
    with open(path) as f:
        override = yaml.safe_load(f) or {}
    return _deep_merge_dict(base, override)


def _build_model_configs(cfg: dict):
    negatives = NegativeSamplingConfig(**cfg["negatives"], random_state=42)
    dataset = DatasetConfig(**cfg["dataset"])
    classifier = ClassifierConfig(**cfg["classifier"])
    features = FeaturesConfig(**(cfg.get("features") or {}))
    return negatives, dataset, classifier, features


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def _normalize_doc(name: str) -> str:
    return name.lower().replace("__", " ").replace("_", " ").strip()


def build_bm25_scorer(catalog: dict[str, str]):
    """Fit a TF-IDF lexical scorer on catalog tool documents (name + the
    candidate text/description; no labels involved, so a legitimate
    unsupervised baseline)."""
    docs = {name: f"{_normalize_doc(name)} {_normalize_doc(desc)}"
            for name, desc in catalog.items()}
    vec = TfidfVectorizer(lowercase=True, analyzer="word")
    all_docs = list(docs.values())
    mat = vec.fit_transform(all_docs)  # rows align with sorted(catalog)
    names = sorted(docs)
    name_to_row = {n: i for i, n in enumerate(names)}
    return vec, mat, name_to_row


def bm25_score_task(vec, mat, name_to_row, query: str, candidates: list[dict]) -> list[float]:
    """Cosine similarity between the task query and each candidate doc."""
    q = vec.transform([query])
    scores = []
    for cand in candidates:
        name = cand["tool_name"]
        row = name_to_row.get(name)
        if row is None:
            scores.append(0.0)
            continue
        sim = float(cosine_similarity(q, mat[row])[0, 0])
        scores.append(sim)
    return scores


def build_dsr_scorer(catalog: dict[str, str], model_name: str = "intfloat/e5-small-v2"):
    """Build a dense (E5, sentence-transformers) retrieval scorer.

    Zero-shot DSR baseline: tool documents are encoded once (no labels used),
    queries are encoded per task, and candidates are ranked by cosine
    similarity. Uses the E5 asymmetric ``passage:``/``query:`` prefixes.
    """
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model_name)
    names = sorted(catalog)
    docs = [catalog.get(n, "") or n for n in names]
    embs = encoder.encode(
        [f"passage: {d}" for d in docs],
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )
    mat = np.asarray(embs, dtype=np.float32)
    name_to_row = {n: i for i, n in enumerate(names)}
    return encoder, mat, name_to_row


def dsr_score_task(encoder, mat, name_to_row, query: str, candidates: list[dict]) -> list[float]:
    """Cosine similarity between the task query embedding and each tool doc."""
    q = np.asarray(
        encoder.encode([f"query: {query}"], normalize_embeddings=True,
                       show_progress_bar=False)[0],
        dtype=np.float32,
    )
    scores = []
    for cand in candidates:
        row = name_to_row.get(cand["tool_name"])
        if row is None:
            scores.append(0.0)
        else:
            scores.append(float(np.dot(q, mat[row])))
    return scores


def _timed(fn, *args, **kwargs) -> tuple[Any, float]:
    """Run ``fn`` and return ``(result, elapsed_ms)``."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - t0) * 1000.0


def build_candidates_for_task(
    traj, pool_name: str, catalog: dict[str, str], app_index: dict[str, list[str]]
) -> list[dict]:
    if pool_name == "catalog_wide":
        names = sorted(catalog)
    else:  # app_scoped
        apps = set(traj.metadata.get("apps") or [])
        names = []
        for app in apps:
            names.extend(app_index.get(app, []))
        names = sorted(set(names))
    return [{"tool_name": n, "tool_description": catalog.get(n, "")} for n in names]


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def _score_methods(
    clf,
    eval_builder,
    traj,
    candidates,
    train_freq: dict[str, int],
    rng,
):
    rows = eval_builder.build_candidates(traj, candidates, relevant_tools=traj.tools_used)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    y_true = df["label"].values
    task_ids = df["task_id"].values
    tool_names = df["tool_name"].values

    # Leak guard: every feature row must carry the train-frozen tool_frequency
    # (never recomputed from evaluation data). build_candidates enforces this by
    # refusing to run without frozen stats; this assert documents the invariant.
    for _, r in df.iterrows():
        assert float(r["tool_frequency"]) == float(train_freq.get(r["tool_name"], 0)), (
            "tool_frequency not train-derived (leak)"
        )

    ordered = {c["tool_name"] for c in candidates}
    relevant = traj.tools_used & ordered
    strictly_unseen = bool(relevant) and not (relevant & set(train_freq))

    model_p, lat_model = _timed(clf.predict_proba, df.drop(columns=["label"]))
    random_p, lat_random = _timed(rng.random, len(df))
    pop_p, lat_pop = _timed(
        lambda: np.array([train_freq.get(t, 0) for t in tool_names], dtype=float)
    )

    out = {
        "model": model_p,
        "random": random_p,
        "popularity": pop_p,
    }
    latency = {"model": lat_model, "random": lat_random, "popularity": lat_pop}
    return y_true, task_ids, tool_names, out, int(strictly_unseen), latency


def _run_seed(
    seed: int,
    negatives_cfg,
    dataset_cfg,
    classifier_cfg,
    features_cfg,
    tasks,
    catalog,
    app_index,
    tool_specs,
    pools_on,
    baselines_on,
    k_values,
    n_folds,
    dsr_model: str = "intfloat/e5-small-v2",
):
    """Run the grouped CV + scoring for one training seed.

    Returns
    -------
    tuple[dict, dict, dict, int]
        ``(task_scores, unseen_flags, app_of_task, n_folds_used)``.
    """
    negatives_cfg.random_state = seed  # deterministic negative sampling per seed
    group_kfold = GroupKFold(n_splits=n_folds)
    groups = [t.task_id for t in tasks]

    task_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    unseen_flags = defaultdict(lambda: defaultdict(dict))
    app_of_task: dict[str, str] = {}
    latency_sum: dict[str, float] = {}
    latency_n: dict[str, int] = {}
    rng = np.random.default_rng(2024 + seed)

    bm25 = None
    if "bm25" in baselines_on:
        bm25 = build_bm25_scorer(catalog)
    dsr = None
    if "dsr_e5" in baselines_on:
        dsr = build_dsr_scorer(catalog, dsr_model)

    fold_idx = 0
    for train_idx, test_idx in group_kfold.split(tasks, groups=groups):
        fold_idx += 1
        train_trajs = [tasks[i] for i in train_idx]
        test_trajs = [tasks[i] for i in test_idx]
        log.info(f"  [seed={seed}] fold {fold_idx}/{n_folds}: train={len(train_trajs)} test={len(test_trajs)}")

        train_builder = DatasetBuilder(
            config=dataset_cfg,
            features_config=features_cfg,
            negatives_config=negatives_cfg,
            tool_catalog=catalog or None, tool_specs=tool_specs,
        )
        train_df = train_builder.build(train_trajs)
        train_freq = train_builder.corpus_stats.tool_frequency

        clf = ShortChainClassifier(classifier_cfg, features_config=features_cfg)
        clf.fit(train_df.drop(columns=["label"]), train_df["label"])

        eval_builder = DatasetBuilder(
            config=dataset_cfg,
            features_config=features_cfg,
            negatives_config=negatives_cfg,
            tool_catalog=catalog or None, tool_specs=tool_specs,
            corpus_stats=train_builder.corpus_stats,  # frozen train stats
        )

        for traj in test_trajs:
            app_of_task[traj.task_id] = traj.app_name
            for pool_name in pools_on:
                candidates = build_candidates_for_task(traj, pool_name, catalog, app_index)
                res = _score_methods(clf, eval_builder, traj, candidates, train_freq, rng)
                if res is None:
                    continue
                y_true, tids, tools, method_probas, strictly_unseen, lat = res
                unseen_flags[pool_name][traj.task_id] = strictly_unseen
                for method, ms in lat.items():
                    latency_sum[method] = latency_sum.get(method, 0.0) + ms
                    latency_n[method] = latency_n.get(method, 0) + 1

                if "bm25" in baselines_on:
                    proba, ms = _timed(bm25_score_task, bm25[0], bm25[1], bm25[2], traj.intent, candidates)
                    method_probas["bm25"] = np.array(proba)
                    latency_sum["bm25"] = latency_sum.get("bm25", 0.0) + ms
                    latency_n["bm25"] = latency_n.get("bm25", 0) + 1
                if "dsr_e5" in baselines_on:
                    proba, ms = _timed(dsr_score_task, dsr[0], dsr[1], dsr[2], traj.intent, candidates)
                    method_probas["dsr_e5"] = np.array(proba)
                    latency_sum["dsr_e5"] = latency_sum.get("dsr_e5", 0.0) + ms
                    latency_n["dsr_e5"] = latency_n.get("dsr_e5", 0) + 1
                for method, proba in method_probas.items():
                    scores = task_level_scores(y_true, proba, tids, k_values=k_values)
                    for tid, metric_map in scores.items():
                        for metric, value in metric_map.items():
                            task_scores[pool_name][method][metric][tid] = value

    return task_scores, unseen_flags, app_of_task, fold_idx, latency_sum, latency_n


def _repeat_prev_scores(candidates: list[dict], last_action: str) -> list[float]:
    """Naive 'repeat the previous tool' baseline scores over candidates."""
    return [1.0 if c["tool_name"] == last_action else 0.0 for c in candidates]


def _run_span_seed(
    seed: int,
    negatives_cfg,
    dataset_cfg,
    classifier_cfg,
    features_cfg,
    tasks,
    catalog,
    app_index,
    tool_specs,
    pools_on,
    baselines_on,
    k_values,
    n_folds,
    dsr_model: str = "intfloat/e5-small-v2",
):
    """Span-level (per-decision, state-aware) CV + scoring for one seed.

    Trains two models per fold: ``model_state`` (context includes state before
    the decision; ``span_index=k``) and ``model_nostate`` (context is
    trajectory-level, ``span_index=None``) as the paper-style ablation.
    Evaluates every decision of the test tasks at its true state, plus
    baselines (random / popularity / BM25 / repeat_prev) on identical rows.
    """
    negatives_cfg.random_state = seed
    groups = [t.task_id for t in tasks]
    gkf = GroupKFold(n_splits=n_folds)

    task_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    unseen_flags = defaultdict(lambda: defaultdict(dict))
    latency_sum: dict[str, float] = {}
    latency_n: dict[str, int] = {}
    rng = np.random.default_rng(2024 + seed)
    bm25 = build_bm25_scorer(catalog) if "bm25" in baselines_on else None
    dsr = build_dsr_scorer(catalog, dsr_model) if "dsr_e5" in baselines_on else None

    feats_off = FeaturesConfig(
        include_state_features=False, include_dependency_features=False
    )

    fold_idx = 0
    for train_idx, test_idx in gkf.split(tasks, groups=groups):
        fold_idx += 1
        train_trajs = [tasks[i] for i in train_idx]
        test_trajs = [tasks[i] for i in test_idx]
        log.info(
            f"  [seed={seed}] span-fold {fold_idx}/{n_folds}: "
            f"train={len(train_trajs)} test={len(test_trajs)}"
        )

        # State-aware training model (corpus stats frozen here).
        b_state = DatasetBuilder(
            config=dataset_cfg, features_config=features_cfg,
            negatives_config=negatives_cfg, tool_catalog=catalog or None, tool_specs=tool_specs,
        )
        state_df = b_state.build_span_dataset(train_trajs, with_state=True)
        train_freq = b_state.corpus_stats.tool_frequency
        clf_state = ShortChainClassifier(classifier_cfg, features_config=features_cfg)
        clf_state.fit(state_df.drop(columns=["label"]), state_df["label"])

        # No-state ablated model: same decisions/labels, trajectory-level ctx.
        b_nostate = DatasetBuilder(
            config=dataset_cfg, features_config=feats_off,
            negatives_config=negatives_cfg, tool_catalog=catalog or None, tool_specs=tool_specs,
            corpus_stats=b_state.corpus_stats,
        )
        nostate_df = b_nostate.build_span_dataset(train_trajs, with_state=False)
        clf_nostate = ShortChainClassifier(classifier_cfg, features_config=feats_off)
        clf_nostate.fit(nostate_df.drop(columns=["label"]), nostate_df["label"])

        e_state = DatasetBuilder(
            config=dataset_cfg, features_config=features_cfg,
            negatives_config=negatives_cfg, tool_catalog=catalog or None, tool_specs=tool_specs,
            corpus_stats=b_state.corpus_stats,
        )
        e_nostate = DatasetBuilder(
            config=dataset_cfg, features_config=feats_off,
            negatives_config=negatives_cfg, tool_catalog=catalog or None, tool_specs=tool_specs,
            corpus_stats=b_state.corpus_stats,
        )

        for traj in test_trajs:
            for k, span in enumerate(traj.spans):
                action = span.tool_name
                if not action:
                    continue
                last_action = traj.spans[k - 1].tool_name if k > 0 else ""
                dec_id = f"{traj.task_id}:{k}"
                for pool_name in pools_on:
                    candidates = build_candidates_for_task(traj, pool_name, catalog, app_index)
                    rows = e_state.build_candidates(
                        traj, candidates, relevant_tools={action}, span_index=k
                    )
                    if not rows:
                        continue
                    df = pd.DataFrame(rows)
                    y = df["label"].values
                    tids = df["task_id"].values
                    tools = df["tool_name"].values

                    model_p, ms = _timed(clf_state.predict_proba, df.drop(columns=["label"]))
                    latency_sum["model_state"] = latency_sum.get("model_state", 0.0) + ms
                    latency_n["model_state"] = latency_n.get("model_state", 0) + 1
                    probas = {
                        "model_state": model_p,
                        "random": rng.random(len(df)),
                        "popularity": np.array([train_freq.get(t, 0) for t in tools], dtype=float),
                        "repeat_prev": np.array(_repeat_prev_scores(candidates, last_action)),
                    }
                    if bm25 is not None:
                        proba, ms = _timed(bm25_score_task, bm25[0], bm25[1], bm25[2], traj.intent, candidates)
                        probas["bm25"] = np.array(proba)
                        latency_sum["bm25"] = latency_sum.get("bm25", 0.0) + ms
                        latency_n["bm25"] = latency_n.get("bm25", 0) + 1
                    if dsr is not None:
                        proba, ms = _timed(dsr_score_task, dsr[0], dsr[1], dsr[2], traj.intent, candidates)
                        probas["dsr_e5"] = np.array(proba)
                        latency_sum["dsr_e5"] = latency_sum.get("dsr_e5", 0.0) + ms
                        latency_n["dsr_e5"] = latency_n.get("dsr_e5", 0) + 1
                    unseen = int(action not in train_freq)
                    unseen_flags[pool_name][dec_id] = unseen

                    for method, proba in probas.items():
                        scores = task_level_scores(y, proba, tids, k_values=k_values)
                        for tid, metric_map in scores.items():
                            for metric, value in metric_map.items():
                                task_scores[pool_name][method][metric][dec_id] = value

                    # Ablation model evaluated at its own (state-free) context.
                    rows_n = e_nostate.build_candidates(
                        traj, candidates, relevant_tools={action}, span_index=None
                    )
                    if not rows_n:
                        continue
                    df_n = pd.DataFrame(rows_n)
                    pn, ms_n = _timed(clf_nostate.predict_proba, df_n.drop(columns=["label"]))
                    latency_sum["model_nostate"] = latency_sum.get("model_nostate", 0.0) + ms_n
                    latency_n["model_nostate"] = latency_n.get("model_nostate", 0) + 1
                    scores_n = task_level_scores(y, pn, df_n["task_id"].values, k_values=k_values)
                    for tid, metric_map in scores_n.items():
                        for metric, value in metric_map.items():
                            task_scores[pool_name]["model_nostate"][metric][dec_id] = value

    return task_scores, unseen_flags, {}, fold_idx, latency_sum, latency_n


def _decision_features(clf, eval_builder, traj, candidates: list[dict]):
    """Per-task decision: full-shortlist correctness (P@R == 1).

    Confidence = the highest predicted probability inside the top-R predicted
    candidates (R = number of relevant tools); outcome = 1 iff the top-R
    shortlist is exactly the relevant set. Top-1 correctness is usually
    saturated (MRR == 1) and would leave calibration without variance.
    """
    rows = eval_builder.build_candidates(
        traj, candidates, relevant_tools=traj.tools_used, span_index=None
    )
    if not rows:
        return None
    d = pd.DataFrame(rows)
    p = clf.predict_proba(d.drop(columns=["label"]))
    r = int(d["label"].values.sum())
    if r == 0:
        return None
    order = np.lexsort((np.arange(len(p)), -p))[:r]  # stable top-R
    conf = float(p[order].max())
    pr_complete = int(int(d["label"].values[order].sum()) == r)
    return conf, pr_complete


def _llm_pr_complete(ranked: list[str], relevant: set[str]) -> int:
    ranked_clean = [t for t in ranked if t]
    top_r = set(ranked_clean[:len(relevant)])
    return int(len(relevant & top_r) == len(relevant))


def _run_calibration_analysis(
    cfg: dict,
    tasks,
    catalog,
    app_index,
    tool_specs,
    negatives_cfg,
    dataset_cfg,
    classifier_cfg,
    features_cfg,
    n_folds: int,
    llm_results_override: str | None = None,
) -> dict:
    """Cross-fold calibration + selective / LLM-fallback hybrid (task level).

    Every task receives an out-of-fold (OOF) prediction (the fold model was
    trained without it). Each fold's calibrator is fit on the OTHER folds'
    OOF (confidence, Recall@k) pairs and applied to the held-out fold — the
    calibrator never sees that fold's tasks (no leakage), and OOF predictions
    carry variance even when the model memorizes train tasks. The LLM cache
    (cost-bound baseline) supplies the deferred outcomes.
    """
    cal_cfg = cfg["calibration"]
    hy_cfg = cfg["hybrid"]
    negatives_cfg.random_state = 42  # pin the calibration OOF to a fixed seed
    method = cal_cfg.get("method", "platt")
    decision = cal_cfg.get("decision", "r_precision_complete")
    n_pts = int(cal_cfg.get("thresholds_points", 41))

    groups = [t.task_id for t in tasks]
    gkf = GroupKFold(n_splits=n_folds)
    oof_by_fold: dict[int, dict[str, tuple[float, int]]] = {}

    for fi, (train_idx, test_idx) in enumerate(gkf.split(tasks, groups=groups)):
        train_trajs = [tasks[i] for i in train_idx]
        test_trajs = [tasks[i] for i in test_idx]
        b = DatasetBuilder(
            config=dataset_cfg, features_config=features_cfg,
            negatives_config=negatives_cfg, tool_catalog=catalog or None,
            tool_specs=tool_specs,
        )
        df = b.build(train_trajs)
        clf = ShortChainClassifier(classifier_cfg, features_config=features_cfg)
        clf.fit(df.drop(columns=["label"]), df["label"])
        eb = DatasetBuilder(
            config=dataset_cfg, features_config=features_cfg,
            negatives_config=negatives_cfg, tool_catalog=catalog or None,
            corpus_stats=b.corpus_stats, tool_specs=tool_specs,
        )
        recs: dict[str, tuple[float, int]] = {}
        for traj in test_trajs:
            cands = build_candidates_for_task(traj, "catalog_wide", catalog, app_index)
            dec = _decision_features(clf, eb, traj, cands)
            if dec is not None:
                recs[traj.task_id] = (dec[0], dec[1])
        oof_by_fold[fi] = recs

    # Cross-fold calibration: each fold's calibrator fits on the other folds.
    test_rows: list[dict] = []
    for fi, recs in oof_by_fold.items():
        pool_conf: list[float] = []
        pool_out: list[int] = []
        for gj in range(n_folds):
            if gj == fi:
                continue
            for _tid, (cc, oo) in oof_by_fold[gj].items():
                pool_conf.append(cc)
                pool_out.append(oo)
        calibrator = None
        if len(set(pool_out)) >= 2:
            calibrator = create_calibrator(method).fit(
                np.asarray(pool_conf, dtype=float), np.asarray(pool_out, dtype=int)
            )
        for tid, (cc, oo) in recs.items():
            conf_cal = (
                float(calibrator.transform(np.asarray([cc]))[0]) if calibrator else float(cc)
            )
            relevant = set(next(t for t in tasks if t.task_id == tid).tools_used)
            test_rows.append({
                "task_id": tid, "conf_raw": float(cc), "conf_cal": conf_cal,
                "local_pr": int(oo), "relevant": sorted(relevant),
            })

    # Load the cost-bound LLM baseline cache.
    llm_tasks: dict = {}
    llm_path = Path(llm_results_override) if llm_results_override else Path(
        hy_cfg.get("llm_results", "models/validation/llm_results.json"))
    if llm_path.exists():
        with open(llm_path) as f:
            llm_tasks = (json.load(f) or {}).get("tasks", {})
    n_llm = len(llm_tasks)

    rows = []
    for r in test_rows:
        lr = llm_tasks.get(r["task_id"])
        if lr is None:
            continue
        rows.append({**r, "llm_pr": _llm_pr_complete(lr.get("ranked") or [], set(r["relevant"])),
                     "llm_cost": float(lr.get("cost_usd", 0.0)),
                     "llm_latency": float(lr.get("latency_ms", 0.0))})
    n_aligned = len(rows)
    conf_raw = np.array([r["conf_raw"] for r in rows], dtype=float)
    conf_cal = np.array([r["conf_cal"] for r in rows], dtype=float)
    local_pr = np.array([r["local_pr"] for r in rows], dtype=float)
    llm_pr = np.array([r["llm_pr"] for r in rows], dtype=float)

    if n_aligned == 0:
        log.warning("P4: no LLM results cached; hybrid/selective metrics unavailable (use --calibrate with --hybrid after running run_llm_baseline.py).")
        return {
            "decision": decision, "calibration_method": method,
            "n_oof_points": len(test_rows), "n_llm_cached": n_llm, "n_aligned": 0,
            "ece_raw": float("nan"), "ece_calibrated": float("nan"),
        }

    thresholds = np.asarray(
        np.quantile(conf_cal, np.linspace(0, 1, n_pts)[1:-1]).tolist() + [1.0]
    )
    local_cost = float(hy_cfg.get("local_cost_usd", 2e-7))
    local_lat = float(hy_cfg.get("local_latency_ms", 0.3))
    llm_cost = float(np.mean([r["llm_cost"] for r in rows])) if n_aligned else 0.0
    llm_lat = float(np.mean([r["llm_latency"] for r in rows])) if n_aligned else 0.0

    ece_raw = expected_calibration_error(local_pr, conf_raw, n_bins=10)
    ece_cal = expected_calibration_error(local_pr, conf_cal, n_bins=10)
    local_raw_curve = coverage_risk_curve(conf_raw, local_pr, thresholds)
    local_cal_curve = coverage_risk_curve(conf_cal, local_pr, thresholds)
    hyb = hybrid_curve(
        conf_cal, local_pr, llm_pr, thresholds,
        local_cost=1.0,
        deferred_cost=llm_cost / local_cost if local_cost else 0.0,
        local_latency=1.0,
        deferred_latency=llm_lat / local_lat if local_lat else 0.0,
    )
    llm_only_risk = 1.0 - float(llm_pr.mean()) if n_aligned else float("nan")
    coverage_at_llm, tau_at_llm = coverage_at_target_risk(hyb, llm_only_risk)

    # Aggregate LLM macro metrics (fair comparison table).
    llm_macro: dict[str, float] = {}
    if llm_tasks:
        keys = set()
        for t in llm_tasks.values():
            keys.update(t.get("metrics", {}).keys())
        for key in sorted(keys):
            vals = [t["metrics"][key] for t in llm_tasks.values() if key in t.get("metrics", {})]
            if vals:
                llm_macro[key] = float(np.mean(vals))

    report = {
        "decision": decision,
        "llm_macro_metrics": llm_macro,
        "calibration_method": method,
        "n_oof_points": len(test_rows),
        "n_llm_cached": n_llm,
        "n_aligned": n_aligned,
        "ece_raw": ece_raw,
        "ece_calibrated": ece_cal,
        "local_risk_full": 1.0 - float(local_pr.mean()),
        "llm_only_risk": llm_only_risk,
        "llm_avg_cost_usd": llm_cost,
        "llm_avg_latency_ms": llm_lat,
        "local_avg_cost_usd": local_cost,
        "local_avg_latency_ms": local_lat,
        "area_local_raw": area_under_risk_coverage(local_raw_curve),
        "area_local_calibrated": area_under_risk_coverage(local_cal_curve),
        "area_hybrid": area_under_risk_coverage(hyb, risk_key="hybrid_risk"),
        "coverage_at_llm_risk": coverage_at_llm,
        "threshold_at_llm_risk": tau_at_llm,
    }
    report["curves"] = {
        "tau": hyb["tau"].tolist(),
        "coverage": hyb["coverage"].tolist(),
        "local_risk": local_cal_curve["risk"].tolist(),
        "hybrid_risk": hyb["hybrid_risk"].tolist(),
        "norm_cost": hyb["norm_cost"].tolist(),
        "norm_latency": hyb["norm_latency"].tolist(),
    }
    log.info(
        f"P4: ECE {ece_raw:.3f} -> {ece_cal:.3f} | cover@LLM-risk {coverage_at_llm:.3f} "
        f"(LLM-only risk={llm_only_risk:.3f})"
    )
    return report


def _print_p4(r: dict) -> None:
    print("\n" + "=" * 74)
    print("  P4 — CALIBRATION & SELECTIVE / LLM-FALLBACK HYBRID (task level)")
    print("=" * 74)
    print(f"  decision                     : {r.get('decision','P@R==1')} (full-shortlist correctness)")
    print(f"  calibration                 : {r['calibration_method']} (fit on {r['n_oof_points']} OOF points)")
    print(f"  ECE (raw -> calibrated)     : {r['ece_raw']:.4f} -> {r['ece_calibrated']:.4f}")
    print(f"  local full risk             : {r['local_risk_full']:.4f}  (1 - P@R==1)")
    print(f"  LLM-only risk (from cache)  : {r['llm_only_risk']:.4f}  ({r['n_llm_cached']} cached, {r['n_aligned']} aligned)")
    print(f"  coverage at LLM-parity risk : {r['coverage_at_llm_risk']:.4f}  (tau={r['threshold_at_llm_risk']:.3f})")
    print(f"  area under risk-coverage    : local(raw)={r['area_local_raw']:.4f} "
          f"local(cal)={r['area_local_calibrated']:.4f} hybrid={r['area_hybrid']:.4f}")
    print("  LLM macro (cache)           : " + " ".join(
        f"{k}={v:.3f}" for k, v in r.get("llm_macro_metrics", {}).items()
        if k in ("r_precision", "recall_at_1", "recall_at_5", "mrr")))
    print(f"  cost/latency per decision   : local ${r['local_avg_cost_usd']:.2e}/{r['local_avg_latency_ms']:.2f}ms | "
          f"LLM ${r['llm_avg_cost_usd']:.2e}/{r['llm_avg_latency_ms']:.0f}ms")
    curve = r["curves"]
    step = max(1, len(curve["tau"]) // 8)
    for j in range(0, len(curve["tau"]), step):
        print(f"    cov={curve['coverage'][j]:.3f}  tau={curve['tau'][j]:.3f}  "
              f"local_risk={curve['local_risk'][j]:.3f}  hybrid_risk={curve['hybrid_risk'][j]:.3f}  "
              f"cost={curve['norm_cost'][j]:.2f}x  latency={curve['norm_latency'][j]:.2f}x")
    print("=" * 74 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run ShortChain P0 AppWorld validation.")
    parser.add_argument("--config", type=str, default=None, help="Path to validation.yaml.")
    parser.add_argument("--traces", type=str, default=None, help="Overrides data.traces_path.")
    parser.add_argument("--output-dir", type=str, default=None, help="Overrides output_dir.")
    parser.add_argument("--success-only", action="store_true", help="Only soft-successful traces.")
    parser.add_argument("--folds", type=int, default=None, help="Override split.n_folds.")
    parser.add_argument("--seeds", type=int, default=None, help="Override seeds (training seeds).")
    parser.add_argument("--seed", type=int, default=None, help="Override bootstrap seed.")
    parser.add_argument("--level", type=str, default=None, help="task | span")
    parser.add_argument("--calibrate", action="store_true", help="Fit cross-fold calibration + report ECE.")
    parser.add_argument("--hybrid", action="store_true", help="Report selective/hybrid metrics using cached LLM results.")
    parser.add_argument("--llm-results", type=str, default=None, help="Override hybrid.llm_results cache path.")
    args = parser.parse_args()

    cfg = _load_validation_config(args.config)
    data_cfg = cfg["data"]
    if args.traces:
        data_cfg["traces_path"] = args.traces
    if args.success_only:
        data_cfg["success_only"] = True
    n_folds = args.folds or cfg["split"]["n_folds"]
    k_values = cfg["eval"]["k_values"]
    headline = cfg["metrics"]["headline"]
    baselines_on = set(cfg["baselines"].get("enabled", ["random", "popularity", "bm25"]))
    pools_on = {k: v for k, v in cfg["eval"]["pools"].items() if v}
    bcfg = cfg["bootstrap"]
    out_dir = Path(args.output_dir or cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_seed = args.seed if args.seed is not None else bcfg["seed"]

    negatives_cfg, dataset_cfg, classifier_cfg, features_cfg = _build_model_configs(cfg)

    # 1. Load traces + catalog + (P2) AppWorld API spec
    t0 = time.time()
    traces = load_appworld_traces(data_cfg["traces_path"], success_only=data_cfg.get("success_only", False))
    if not traces:
        log.error("No traces loaded.")
        sys.exit(1)
    catalog = reconstruct_catalog(data_cfg["traces_path"])
    app_index = catalog_app_index(catalog)

    fc_dir = data_cfg.get("appworld_api_dir") or ""
    tool_specs: dict[str, Any] = {}
    spec_coverage: dict[str, Any] = {}
    if fc_dir and (Path(fc_dir).is_dir() or Path(fc_dir + "/spotify.json").exists()):
        catalog, tool_specs = build_catalog_and_schemas(fc_dir, catalog)
        spec_coverage = {
            "tools_with_specs": len(tool_specs),
            "tools_total": len(catalog),
            "coverage": round(len(tool_specs) / max(1, len(catalog)), 4),
            "missing": sorted(set(catalog) - set(tool_specs)),
        }
        log.info(f"AppWorld API spec: {spec_coverage['tools_with_specs']}/{spec_coverage['tools_total']} tools resolved")
    log.info(f"Loaded {len(traces)} tasks, catalog={len(catalog)} tools")

    # 2. Multi-seed grouped CV (per-seed task scores averaged, paper-style)
    n_seeds = int(args.seeds if args.seeds is not None else cfg.get("seeds", 1))
    level = (args.level or cfg.get("level", "task")).lower()
    if level not in ("task", "span"):
        log.error(f"Unknown level: {level!r} (expected 'task' or 'span')")
        sys.exit(1)
    tasks = list(traces)
    app_of_task = {t.task_id: t.app_name for t in tasks}

    per_seed = []
    dsr_model = (cfg.get("dsr", {}) or {}).get("encoder", "intfloat/e5-small-v2")
    for s in range(n_seeds):
        if level == "span":
            ts, uf, _app, nfolds, ls, ln = _run_span_seed(
                s, negatives_cfg, dataset_cfg, classifier_cfg, features_cfg,
                tasks, catalog, app_index, tool_specs, pools_on, baselines_on,
                k_values, n_folds, dsr_model,
            )
        else:
            ts, uf, _app, nfolds, ls, ln = _run_seed(
                s, negatives_cfg, dataset_cfg, classifier_cfg, features_cfg,
                tasks, catalog, app_index, tool_specs, pools_on, baselines_on,
                k_values, n_folds, dsr_model,
            )
        per_seed.append((ts, uf, ls, ln))
    fold_idx = nfolds
    primary = "model" if level == "task" else "model_state"

    # Average per-task scores across seeds (tasks identical across seeds).
    task_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for pool_name in pools_on:
        for method in per_seed[0][0][pool_name]:
            for metric in per_seed[0][0][pool_name][method]:
                acc: dict[str, list[float]] = defaultdict(list)
                for ts, _uf, _ls, _ln in per_seed:
                    for tid, v in ts[pool_name][method][metric].items():
                        acc[tid].append(v)
                for tid, vals in acc.items():
                    task_scores[pool_name][method][metric][tid] = float(np.mean(vals))

    unseen_flags = per_seed[0][1]  # deterministic across seeds (same fold split)

    # Aggregate per-method latency (ms / decision) across seeds.
    latency_all: dict[str, float] = {}
    for _ts, _uf, ls, ln in per_seed:
        for method, ms in ls.items():
            latency_all.setdefault(method, [0.0, 0])[0] += ms
            latency_all.setdefault(method, [0.0, 0])[1] += ln.get(method, 0)
    latency_ms = {
        method: (total / n if n else 0.0)
        for method, (total, n) in latency_all.items()
    }

    # 5. Aggregate + significance
    report: dict = {
        "protocol": "p0_appworld_halo",
        "level": level,
        "primary_method": primary,
        "n_tasks": len(tasks),
        "n_train_seeds": n_seeds,
        "n_effective_folds": fold_idx,
        "n_folds": n_folds,
        "catalog_size": len(catalog),
        "success_only": data_cfg.get("success_only", False),
        "k_values": k_values,
        "headline": headline,
        "baselines": sorted(baselines_on),
        "pools": sorted(pools_on),
        "appworld_api_spec": spec_coverage,
        "latency_ms_per_decision": latency_ms,
    }

    # Unseen-overlap: at task level this is tasks whose every relevant tool is
    # unseen; at span level it is decisions whose target tool is unseen in train.
    report["strictly_unseen_tasks" if level == "task" else "unseen_decisions"] = {
        pool: sum(1 for _, f in unseen_flags[pool].items() if f)
        for pool in pools_on
    }

    per_pool = {}
    for pool_name in pools_on:
        pool_block = {}
        methods = task_scores[pool_name]
        for method in sorted(methods):
            method_block = {}
            for metric in sorted(methods[method]):
                scores = methods[method][metric]
                mean, lo, hi = bootstrap_mean_ci(
                    scores, n_boot=bcfg["n_boot"], seed=summary_seed, ci=bcfg["ci"]
                )
                n = len(scores)
                method_block[metric] = {"n": n, "mean": mean, "lo": lo, "hi": hi}
                # per-app
                by_app = defaultdict(dict)
                for tid, v in scores.items():
                    app_key = app_of_task.get(tid.split(":")[0], "?")
                    by_app[app_key][tid] = v
                method_block.setdefault("per_app", {})[metric] = {
                    app: bootstrap_mean_ci(s_,
                                           n_boot=min(bcfg["n_boot"], 200),
                                           seed=summary_seed,
                                           ci=bcfg["ci"])[0]
                    for app, s_ in sorted(by_app.items())
                }
            pool_block[method] = method_block

        # model vs baselines significance (per metric, Holm within metric)
        if primary in methods:
            sig = {}
            for metric in headline:
                if metric not in methods[primary]:
                    continue
                rows = []
                for base in sorted(methods):
                    if base == primary or metric not in methods[base]:
                        continue
                    if metric not in methods[base]:
                        continue
                    _, lo, hi, p = paired_bootstrap_p_and_ci(
                        methods[primary][metric], methods[base][metric],
                        n_boot=min(bcfg["n_boot"], 500), seed=summary_seed, ci=bcfg["ci"],
                    )
                    rows.append((base, lo, hi, p))
                if rows:
                    rej = holm_bonferroni([r[3] for r in rows], alpha=0.05)
                    sig[metric] = [
                        {"baseline": b, "delta_lo": lo, "delta_hi": hi,
                         "p": p, "better_significant": bool(rej[j]) and lo > 0}
                        for j, (b, lo, hi, p) in enumerate(rows)
                    ]
            pool_block["model_vs_baselines"] = sig

        # strictly-unseen subset: tasks whose every relevant tool is unseen in train
        unseen_tids = [tid for tid, flag in unseen_flags[pool_name].items() if flag]
        if unseen_tids:
            unseen_block = {}
            for method in sorted(methods):
                if method == "model_vs_baselines":
                    continue
                unseen_block[method] = {}
                for metric in sorted(methods[method]):
                    restricted = {tid: v for tid, v in methods[method][metric].items()
                                  if tid in unseen_tids}
                    if not restricted:
                        continue
                    mean, lo, hi = bootstrap_mean_ci(
                        restricted, n_boot=bcfg["n_boot"], seed=summary_seed, ci=bcfg["ci"]
                    )
                    unseen_block[method][metric] = {"n": len(restricted), "mean": mean,
                                                    "lo": lo, "hi": hi}
            pool_block["strictly_unseen"] = unseen_block

        per_pool[pool_name] = pool_block

    report["results"] = per_pool

    # 6. console + persist
    results_file = out_dir / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(report, f, indent=2, default=float)
    log.info(f"Saved report to {results_file}")

    _print_report(report)

    # P4: calibration + selective / LLM-fallback hybrid (task level only).
    if (args.calibrate or args.hybrid) and level == "task":
        cal_report = _run_calibration_analysis(
            cfg, tasks, catalog, app_index, tool_specs,
            negatives_cfg, dataset_cfg, classifier_cfg, features_cfg, n_folds,
            args.llm_results,
        )
        report["p4_calibration_hybrid"] = cal_report
        with open(results_file, "w") as f:
            json.dump(report, f, indent=2, default=float)
        _print_p4(cal_report)
    elif (args.calibrate or args.hybrid) and level != "task":
        log.warning("--calibrate/--hybrid apply to task-level only; skipped for level=%s", level)

    log.info(f"Total runtime: {time.time() - t0:.1f}s")


def _print_report(report: dict) -> None:
    headline = report["headline"]
    primary = report.get("primary_method", "model")
    level = report.get("level", "task")
    print("\n" + "=" * 74)
    print(f"  SHORTCHAIN P0 — APPWORLD ({level}-level, n={report['n_tasks']} tasks, "
          f"catalog={report['catalog_size']})")
    print("=" * 74)
    for pool_name, pool_block in report["results"].items():
        print(f"\n## Pool: {pool_name}")
        methods = [m for m in sorted(pool_block) if m != "model_vs_baselines"]
        print("  " + "method       " + "  ".join(f"{m:>11}" for m in headline))
        for method in methods:
            block = pool_block[method]
            cells = []
            for m in headline:
                if m in block:
                    mean = block[m]["mean"]
                    cells.append(f"{mean:.3f}")
                else:
                    cells.append("   -  ")
            print(f"  {method:<12}" + "  ".join(f"{c:>11}" for c in cells))
        sig = pool_block.get("model_vs_baselines", {})
        if sig:
            print(f"\n  {primary} vs baseline significance (95% CI for delta; †=sig after Holm):")
            for metric, rows in sig.items():
                desc = []
                for r in rows:
                    star = "†" if r["better_significant"] else " "
                    desc.append(f"{r['baseline']}{star}[{r['delta_lo']:+.3f},{r['delta_hi']:+.3f}]")
                print(f"    {metric:<12} " + "  ".join(desc))
        us = pool_block.get("strictly_unseen")
        if us and primary in us:
            print(f"\n  {'strictly-unseen subset (all relevant tools unseen)' if level=='task' else 'unseen-target decisions'}:")
            cells = []
            for m in headline:
                mean = us[primary].get(m, {}).get("mean")
                cells.append(f"{mean:.3f}" if mean is not None else "   -  ")
            print("    " + primary + "  " + "  ".join(f"{c:>11}" for c in cells) + f"  (n={us[primary].get(list(us[primary])[0], {}).get('n', '?')})")
    latency_ms = report.get("latency_ms_per_decision", {})
    if latency_ms:
        print("\n  latency (ms / decision; per method):")
        for method in sorted(latency_ms, key=lambda m: latency_ms[m]):
            print(f"    {method:<12} {latency_ms[method]:.4f} ms")
    print("\nNote: lo>0 and † means the model is significantly BETTER than baseline (Holm).")
    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()
