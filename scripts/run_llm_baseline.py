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

from shortchain.adapters.appworld_api import load_appworld_api_spec
from shortchain.adapters.halo import load_appworld_traces, reconstruct_catalog
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


_BATCH_URL = "https://openrouter.ai/api/beta/batches"


def _messages(intent: str, tool_list_text: str, top_k: int) -> list[dict]:
    system = (
        "You are a tool-selection assistant for an autonomous agent. "
        f"Given a user task, select the up to {top_k} most relevant tools from the "
        "provided list that the agent should call, ranked by descending relevance. "
        "Respond with ONLY a JSON array of tool names (strings), no commentary."
    )
    user = (f"User task:\n{intent}\n\nAvailable tools:\n{tool_list_text}\n\n"
            f"Return a JSON array of the top-{top_k} tool names in ranked order.")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_requests(tasks, catalog: dict[str, str], specs: dict, top_k: int) -> list[dict]:
    tool_list_text = build_tool_list(catalog, specs)
    requests = []
    for traj in tasks:
        requests.append({
            "custom_id": traj.task_id,
            "body": {
                "messages": _messages(traj.intent, tool_list_text, top_k),
                "temperature": 0.1,
                "top_p": 0.9,
                "max_tokens": 1024,
            },
        })
    return requests


def submit_batch(api_key: str, model: str, reqs: list[dict]) -> dict:
    resp = requests.post(
        _BATCH_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"endpoint": "/v1/chat/completions", "model": model, "requests": reqs},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


def poll_batch(api_key: str, batch_id: str, retries: int = 6, backoff: float = 10.0) -> dict:
    """GET batch status, retrying transient 404/5xx (batch may not be visible
    immediately after submission)."""
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(
                f"{_BATCH_URL}/{batch_id}",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                timeout=180,
            )
            if resp.status_code < 500 or attempt >= retries:
                resp.raise_for_status()
                return resp.json()
        except requests.RequestException as err:  # noqa: BLE001
            last = err
        time.sleep(backoff)
    raise RuntimeError(f"poll_batch failed after {retries} retries: {last}")


def _save_meta(out_path: Path, meta: dict) -> None:
    payload = {"model": meta["model"], "batch_meta": meta, "tasks": {},
               "total_cost_usd": 0.0}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def _read_meta(out_path: Path) -> dict:
    if out_path.exists():
        try:
            d = json.load(open(out_path))
            if d.get("batch_meta"):
                return d["batch_meta"]
        except Exception:
            pass
    return {}


def run_batch(
    api_key: str, model: str, tasks, catalog, specs, top_k: int, k_values: list[int],
    out_path: Path, batch_id: str | None = None,
    max_wait_min: int = 120, poll_s: int = 20,
) -> None:
    task_map = {t.task_id: t for t in tasks}
    meta = _read_meta(out_path) if batch_id is None else {"batch_id": batch_id}
    meta.setdefault("model", model)

    if not meta.get("batch_id"):
        reqs = build_requests(tasks, catalog, specs, top_k)
        batch = submit_batch(api_key, model, reqs)
        meta = {"batch_id": batch["id"], "status": batch["status"], "model": model,
                "submitted_requests": len(reqs)}
        _save_meta(out_path, meta)
        log.info(f"Submitted batch {batch['id']} with {len(reqs)} requests.")
    else:
        log.info(f"Resuming/existing batch {meta['batch_id']}.")

    batch_id = meta["batch_id"]
    deadline = time.time() + max_wait_min * 60
    batch = None
    while True:
        batch = poll_batch(api_key, batch_id)
        st = batch["status"]
        counts = batch.get("request_counts") or {}
        meta["status"] = st
        log.info(f"  batch {batch_id}: {st} completed={counts.get('completed')}/{counts.get('total')}")
        _save_meta(out_path, meta)
        if st == "completed":
            break
        if st in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"batch {st}: {batch.get('error')}")
        if time.time() > deadline:
            log.error(f"Batch still {st} after {max_wait_min}min. Resume later with --batch-id {batch_id}")
            sys.exit(2)
        time.sleep(poll_s)

    results = {r["custom_id"]: r for r in (batch.get("results") or [])}
    out: dict = {}
    total_cost = 0.0
    for task_id, it in results.items():
        if it.get("error"):
            log.warning(f"  {task_id[:8]} batch error: {it['error']}")
            continue
        body = (it.get("response") or {}).get("body") or {}
        message = (body.get("choices") or [{}])[0].get("message") or {}
        content = message.get("content") or message.get("reasoning") or ""
        usage = body.get("usage") or {}
        traj = task_map.get(task_id)
        if traj is None:
            continue
        ranked = parse_ranked(content)
        metrics = task_metrics(ranked, traj.tools_used, k_values)
        cost = float(usage.get("cost", 0.0) or 0.0)
        if not cost:
            cost = int(usage.get("prompt_tokens", 0)) * 1e-7 + int(usage.get("completion_tokens", 0)) * 4e-7
        total_cost += cost
        out[task_id] = {
            "task_id": task_id, "intent": traj.intent, "ranked": ranked,
            "metrics": metrics,
            "tokens_in": int(usage.get("prompt_tokens", 0)),
            "tokens_out": int(usage.get("completion_tokens", 0)),
            "cost_usd": cost, "latency_ms": 0.0, "model": model, "batch": True,
        }
    with open(out_path, "w") as f:
        json.dump({"model": model, "tasks": out, "total_cost_usd": total_cost,
                   "batch_meta": meta}, f, indent=2)
    log.info(f"Batch complete: {len(out)} tasks, total cost ~${total_cost:.3f} -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM tool-selection baseline (task-level).")
    parser.add_argument("--config", type=str, default="configs/validation.yaml")
    parser.add_argument("--model", type=str, default="deepseek/deepseek-v4-flash-0731")
    parser.add_argument("--top-k", type=int, default=9)
    parser.add_argument("--sample-size", type=int, default=0, help="0 = all tasks")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="models/validation/llm_results.json")
    parser.add_argument("--api-key", type=str, default=None, help="OpenRouter key (env OPENROUTER_API_KEY preferred)")
    parser.add_argument("--batch-id", type=str, default=None, help="Resume polling an existing OpenRouter batch.")
    parser.add_argument("--max-wait-min", type=int, default=120, help="Batch poll timeout (minutes).")
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

    # Async OpenRouter batch route (e.g. google/gemini-3.6-flash:batch).
    if ":batch" in args.model or args.batch_id:
        run_batch(
            api_key=api_key, model=args.model, tasks=tasks, catalog=catalog,
            specs=specs, top_k=args.top_k, k_values=k_values,
            out_path=Path(args.output), batch_id=args.batch_id,
            max_wait_min=args.max_wait_min,
        )
        return

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
