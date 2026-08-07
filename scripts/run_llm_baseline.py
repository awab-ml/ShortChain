#!/usr/bin/env python3
"""Cost-bound LLM tool-selection baseline for ShortChain P4 (task-level).

Runs an off-the-shelf LLM (default ``deepseek/deepseek-v4-flash-0731`` via
OpenRouter) as a zero-shot tool shortlister on the SAME inputs a deployment
would see: the task instruction (intent) + the candidate tool definitions
from the AppWorld function_calling spec. The LLM sees no labels or solutions.

Outputs are cached to JSON (ranked tools, tokens, cost estimate per task) so
the deterministic validation harness can compute hybrid metrics offline. The
API key is read from the OPENROUTER_API_KEY environment variable (or
``--api-key``); it is never written to disk.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import requests

from shortchain.integrations.appworld_api import load_appworld_api_spec
from shortchain.integrations.halo import load_appworld_traces, reconstruct_catalog
from shortchain.utils.logging import get_logger

log = get_logger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _tool_line(name: str, description: str, n_args: int) -> str:
    desc = (description or "").strip()
    if len(desc) > 120:
        desc = desc[:120].rsplit(" ", 1)[0] + "…"
    return f"- {name}: {desc} [{n_args} args]"


def build_tool_list(catalog: dict[str, str], specs: dict) -> str:
    lines = []
    for name in sorted(catalog):
        spec = specs.get(name)
        desc = spec.description if spec is not None else catalog.get(name, "")
        n_args = spec.n_params if spec is not None else 0
        lines.append(_tool_line(name, desc, n_args))
    return "\n".join(lines)


def call_llm(intent: str, tool_list_text: str, top_k: int, model: str, api_key: str):
    system = (
        "You are a tool-selection assistant for an autonomous agent. "
        f"Given a user task, select the up to {top_k} most relevant tools from the "
        "provided list that the agent should call, ranked by descending relevance. "
        "Respond with ONLY a JSON array of tool names (strings), no commentary."
    )
    user = f"User task:\n{intent}\n\nAvailable tools:\n{tool_list_text}\n\nReturn a JSON array of the top-{top_k} tool names in ranked order."
    start = time.perf_counter()
    resp = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 1024,
        },
        timeout=120,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    resp.raise_for_status()
    data = resp.json()
    message = data["choices"][0]["message"]
    content = message.get("content") or message.get("reasoning") or ""
    usage = data.get("usage", {})
    tokens_in = int(usage.get("prompt_tokens", 0))
    tokens_out = int(usage.get("completion_tokens", 0))
    cost = float(usage.get("cost", 0.0) or 0.0)
    if not cost:
        # reference estimate only (per-1k tokens); deepseek-v4-flash-class pricing
        cost = tokens_in * 2e-7 + tokens_out * 8e-7
    return {"content": content, "latency_ms": latency_ms,
            "tokens_in": tokens_in, "tokens_out": tokens_out, "cost_usd": cost}


def _as_name(item) -> str | None:
    if isinstance(item, str):
        return item if item.strip() else None
    if isinstance(item, dict):
        for key in ("name", "tool_name", "tool"):
            v = item.get(key)
            if isinstance(v, str) and v.strip():
                return v
    if isinstance(item, (int, float)):
        return str(item)
    return None


def parse_ranked(content) -> list[str]:
    """Very defensive extraction of the ranked tool-name list."""
    import re

    text = (content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()

    parsed = None
    # 1) try the whole text as JSON
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        parsed = None
    # 2) try extracting the first [...] block
    if parsed is None:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
            except (json.JSONDecodeError, ValueError):
                parsed = None
    # 3) fallback: ordered quoted strings
    names: list[str] = []
    if parsed is not None:
        if isinstance(parsed, dict):
            for key in ("tools", "tool_names", "result", "ranked", "output"):
                if key in parsed:
                    parsed = parsed[key]
                    break
        if isinstance(parsed, list):
            for item in parsed:
                name = _as_name(item)
                if name and name not in names:
                    names.append(name)
    if not names:
        for tok in re.findall(r'"([^"]+)"', text):
            if tok and tok not in names:
                names.append(tok)
    return names

def task_metrics(ranked: list[str], relevant: set[str], k_values: list[int]) -> dict:
    ranked = [t for t in ranked if t]  # names only
    order = {}
    for pos, name in enumerate(ranked):
        order.setdefault(name, pos)
    r = len(relevant)
    metrics: dict = {
        "r_precision": 0.0,
        "mrr": 0.0,
    }
    if r > 0:
        rel_orders = [order[t] for t in relevant if t in order]
        top_r = [o for o in rel_orders if o < r]
        metrics["r_precision"] = len(top_r) / r
        if rel_orders:
            metrics["mrr"] = 1.0 / (min(rel_orders) + 1)
    for k in k_values:
        hits = sum(1 for o in rel_orders if o < k) if r > 0 else 0
        metrics[f"recall_at_{k}"] = (hits / r) if r > 0 else 0.0
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM tool-selection baseline (task-level).")
    parser.add_argument("--config", type=str, default="configs/validation.yaml")
    parser.add_argument("--model", type=str, default="deepseek/deepseek-v4-flash-0731")
    parser.add_argument("--top-k", type=int, default=9)
    parser.add_argument("--sample-size", type=int, default=0, help="0 = all tasks")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="models/validation/llm_results.json")
    parser.add_argument("--api-key", type=str, default=None, help="OpenRouter key (env OPENROUTER_API_KEY preferred)")
    args = parser.parse_args()

    api_key = args.api_key or __import__("os").environ.get("OPENROUTER_API_KEY")
    if not api_key:
        log.error("No OpenRouter API key. Set OPENROUTER_API_KEY or pass --api-key.")
        sys.exit(1)

    import yaml
    with open(args.config) as f:
        vcfg = yaml.safe_load(f) or {}
    data_cfg = vcfg.get("data", {})
    traces_path = data_cfg.get("traces_path", "data/traces.jsonl")
    fc_dir = data_cfg.get("appworld_api_dir") or ""

    traces = load_appworld_traces(traces_path, success_only=data_cfg.get("success_only", False))
    tasks = list(traces)
    if args.sample_size and args.sample_size < len(tasks):
        rng = np.random.default_rng(args.sample_seed)
        tasks = [tasks[i] for i in sorted(rng.choice(len(tasks), size=args.sample_size, replace=False))]
    catalog = reconstruct_catalog(traces_path)
    specs = load_appworld_api_spec(fc_dir) if fc_dir and Path(fc_dir).is_dir() else {}
    tool_list_text = build_tool_list(catalog, specs)
    k_values = vcfg.get("eval", {}).get("k_values", [1, 3, 5, 7, 9])

    results: dict = {}
    total_cost = 0.0
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _checkpoint():
        tmp = out_path.with_suffix(".json.tmp")
        with open(tmp, "w") as fh:
            json.dump({"model": args.model, "tasks": results,
                       "total_cost_usd": total_cost}, fh, indent=2)
        tmp.replace(out_path)
    for i, traj in enumerate(tasks, 1):
        try:
            resp = call_llm(traj.intent, tool_list_text, args.top_k, args.model, api_key)
        except Exception as err:  # noqa: BLE001
            log.warning(f"[{i}/{len(tasks)}] {traj.task_id[:8]} failed: {err}")
            continue
        ranked = parse_ranked(resp.get("content"))
        metrics = task_metrics(ranked, traj.tools_used, k_values)
        total_cost += resp["cost_usd"]
        results[traj.task_id] = {
            "task_id": traj.task_id,
            "intent": traj.intent,
            "ranked": ranked,
            "metrics": metrics,
            "tokens_in": resp["tokens_in"],
            "tokens_out": resp["tokens_out"],
            "cost_usd": resp["cost_usd"],
            "latency_ms": resp["latency_ms"],
            "model": args.model,
        }
        log.info(f"[{i}/{len(tasks)}] {traj.task_id[:8]} rank0={ranked[:1]} cost=${resp['cost_usd']:.5f} r1={metrics['recall_at_1']:.2f}")
        _checkpoint()
        time.sleep(0.05)  # gentle rate limiting

    _checkpoint()
    log.info(f"Saved {len(results)} tasks to {out_path} (total cost ~${total_cost:.3f})")


if __name__ == "__main__":
    main()
