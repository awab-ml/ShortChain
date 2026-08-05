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
  tool frequency), BM25 (TF-IDF cosine, fit on catalog documents only).
- Per-task metric values are pooled over test folds and macro-averaged with
  paired-bootstrap 95% CIs; model vs. baseline contrasts get Holm-Bonferroni
  control within each metric.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupKFold

from shortchain.config import ClassifierConfig, DatasetConfig, FeaturesConfig, NegativeSamplingConfig
from shortchain.dataset.builder import DatasetBuilder
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
from shortchain.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_validation_config(path: str | Path | None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parent.parent / "configs" / "validation.yaml"
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _build_model_configs(cfg: dict):
    negatives = NegativeSamplingConfig(**cfg["negatives"], random_state=42)
    dataset = DatasetConfig(**cfg["dataset"])
    classifier = ClassifierConfig(**cfg["classifier"])
    features = FeaturesConfig()
    return negatives, dataset, classifier, features


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def _normalize_doc(name: str) -> str:
    return name.lower().replace("__", " ").replace("_", " ").strip()


def build_bm25_scorer(catalog: dict[str, str]):
    """Fit a TF-IDF lexical scorer on catalog tool documents (name only; no
    labels involved, so it is a legitimate unsupervised baseline)."""
    docs = {name: _normalize_doc(name) for name in catalog}
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
    return [{"tool_name": n, "tool_description": ""} for n in names]


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

    out = {
        "model": clf.predict_proba(df.drop(columns=["label"])),
        "random": rng.random(len(df)),
        "popularity": np.array([train_freq.get(t, 0) for t in tool_names], dtype=float),
    }
    return y_true, task_ids, tool_names, out, int(strictly_unseen)


def _run_seed(
    seed: int,
    negatives_cfg,
    dataset_cfg,
    classifier_cfg,
    features_cfg,
    tasks,
    catalog,
    app_index,
    pools_on,
    baselines_on,
    k_values,
    n_folds,
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
    rng = np.random.default_rng(2024 + seed)

    bm25 = None
    if "bm25" in baselines_on:
        bm25 = build_bm25_scorer(catalog)

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
            tool_catalog=catalog or None,
        )
        train_df = train_builder.build(train_trajs)
        train_freq = train_builder.corpus_stats.tool_frequency

        clf = ShortChainClassifier(classifier_cfg, features_config=features_cfg)
        clf.fit(train_df.drop(columns=["label"]), train_df["label"])

        eval_builder = DatasetBuilder(
            config=dataset_cfg,
            features_config=features_cfg,
            negatives_config=negatives_cfg,
            tool_catalog=catalog or None,
            corpus_stats=train_builder.corpus_stats,  # frozen train stats
        )

        for traj in test_trajs:
            app_of_task[traj.task_id] = traj.app_name
            for pool_name in pools_on:
                candidates = build_candidates_for_task(traj, pool_name, catalog, app_index)
                res = _score_methods(clf, eval_builder, traj, candidates, train_freq, rng)
                if res is None:
                    continue
                y_true, tids, tools, method_probas, strictly_unseen = res
                unseen_flags[pool_name][traj.task_id] = strictly_unseen

                if "bm25" in baselines_on:
                    method_probas["bm25"] = np.array(
                        bm25_score_task(bm25[0], bm25[1], bm25[2], traj.intent, candidates)
                    )
                for method, proba in method_probas.items():
                    scores = task_level_scores(y_true, proba, tids, k_values=k_values)
                    for tid, metric_map in scores.items():
                        for metric, value in metric_map.items():
                            task_scores[pool_name][method][metric][tid] = value

    return task_scores, unseen_flags, app_of_task, fold_idx


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

    # 1. Load traces + catalog
    t0 = time.time()
    traces = load_appworld_traces(data_cfg["traces_path"], success_only=data_cfg.get("success_only", False))
    if not traces:
        log.error("No traces loaded.")
        sys.exit(1)
    catalog = reconstruct_catalog(data_cfg["traces_path"])
    app_index = catalog_app_index(catalog)
    log.info(f"Loaded {len(traces)} tasks, catalog={len(catalog)} tools")

    # 2. Multi-seed grouped CV (per-seed task scores averaged, paper-style)
    n_seeds = int(args.seeds if args.seeds is not None else cfg.get("seeds", 1))
    tasks = list(traces)
    app_of_task = {t.task_id: t.app_name for t in tasks}

    per_seed = []
    for s in range(n_seeds):
        ts, uf, _app, nfolds = _run_seed(
            s, negatives_cfg, dataset_cfg, classifier_cfg, features_cfg,
            tasks, catalog, app_index, pools_on, baselines_on, k_values, n_folds,
        )
        per_seed.append((ts, uf))
    fold_idx = nfolds

    # Average per-task scores across seeds (tasks identical across seeds).
    task_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for pool_name in pools_on:
        for method in per_seed[0][0][pool_name]:
            for metric in per_seed[0][0][pool_name][method]:
                acc: dict[str, list[float]] = defaultdict(list)
                for ts, _uf in per_seed:
                    for tid, v in ts[pool_name][method][metric].items():
                        acc[tid].append(v)
                for tid, vals in acc.items():
                    task_scores[pool_name][method][metric][tid] = float(np.mean(vals))

    unseen_flags = per_seed[0][1]  # deterministic across seeds (same fold split)

    # 5. Aggregate + significance
    report: dict = {
        "protocol": "p0_appworld_halo",
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
    }

    # All-tools-unseen-overlap occurs only when every relevant tool of a task
    # is absent from the train corpus; report it explicitly (0 is expected with
    # AppWorld's shared catalog and is the paper's stable-catalog regime).
    report["strictly_unseen_tasks"] = {
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
                    by_app[app_of_task.get(tid, "?")][tid] = v
                method_block.setdefault("per_app", {})[metric] = {
                    app: bootstrap_mean_ci(s_,
                                           n_boot=min(bcfg["n_boot"], 200),
                                           seed=summary_seed,
                                           ci=bcfg["ci"])[0]
                    for app, s_ in sorted(by_app.items())
                }
            pool_block[method] = method_block

        # model vs baselines significance (per metric, Holm within metric)
        if "model" in methods:
            sig = {}
            for metric in headline:
                if metric not in methods["model"]:
                    continue
                rows = []
                for base in sorted(methods):
                    if base == "model" or metric not in methods[base]:
                        continue
                    if metric not in methods[base]:
                        continue
                    _, lo, hi, p = paired_bootstrap_p_and_ci(
                        methods["model"][metric], methods[base][metric],
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
    log.info(f"Total runtime: {time.time() - t0:.1f}s")


def _print_report(report: dict) -> None:
    headline = report["headline"]
    print("\n" + "=" * 74)
    print(f"  SHORTCHAIN P0 — APPWORLD (n={report['n_tasks']} tasks, "
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
            print("\n  model vs baseline significance (95% CI for delta; †=sig after Holm):")
            for metric, rows in sig.items():
                desc = []
                for r in rows:
                    star = "†" if r["better_significant"] else " "
                    desc.append(f"{r['baseline']}{star}[{r['delta_lo']:+.3f},{r['delta_hi']:+.3f}]")
                print(f"    {metric:<12} " + "  ".join(desc))
        us = pool_block.get("strictly_unseen")
        if us and "model" in us:
            print("\n  strictly-unseen subset (all relevant tools unseen in train):")
            cells = []
            for m in headline:
                mean = us["model"].get(m, {}).get("mean")
                cells.append(f"{mean:.3f}" if mean is not None else "   -  ")
            print("    model  " + "  ".join(f"{c:>11}" for c in cells) + f"  (n={us['model'].get(list(us['model'])[0], {}).get('n', '?')})")
    print("\nNote: lo>0 and † means the model is significantly BETTER than baseline (Holm).")
    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()
