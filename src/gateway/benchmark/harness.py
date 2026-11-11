"""Labelled eval for routing accuracy, cache hits, latency, and savings.

The published numbers use the mock provider plus a few milliseconds of simulated
upstream delay. No network and no API key.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.catalog import label_tier
from gateway.config import Settings

_LABELS = ("easy", "medium", "hard")


def default_prompts_path() -> Path:
    cwd = Path("benchmarks/prompts.jsonl")
    if cwd.exists():
        return cwd
    return Path(__file__).resolve().parents[3] / "benchmarks" / "prompts.jsonl"


def load_prompts(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        item = json.loads(stripped)
        rows.append(
            {
                "id": str(item["id"]),
                "label": str(item["label"]),
                "prompt": str(item["prompt"]),
                "paraphrase": str(item.get("paraphrase") or ""),
            }
        )
    return rows


_AMBIENT_KEYS = (
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GATEWAY_GROQ_API_KEY",
    "GATEWAY_GEMINI_API_KEY",
    "GATEWAY_OLLAMA_BASE_URL",
)


def run_benchmark(
    prompts_path: Path | None = None,
    *,
    simulate_latency: bool = True,
    api_key: str = "gw_bench_key",
) -> dict[str, Any]:
    path = prompts_path or default_prompts_path()
    prompts = load_prompts(path)
    saved_env = {name: os.environ.pop(name) for name in _AMBIENT_KEYS if name in os.environ}
    try:
        return _run(prompts, simulate_latency=simulate_latency, api_key=api_key)
    finally:
        os.environ.update(saved_env)


def _run(prompts: list[dict[str, str]], *, simulate_latency: bool, api_key: str) -> dict[str, Any]:
    settings = Settings(
        database_url="sqlite:///:memory:",
        redis_url=None,
        otel_exporter="none",
        embeddings="hash",
        admin_api_key="gw_bench_admin",
        bootstrap_api_key=api_key,
        enable_mock=True,
        groq_api_key=None,
        gemini_api_key=None,
        ollama_base_url=None,
        mock_latency_ms=8 if simulate_latency else 0,
        retry_backoff_seconds=0,
        default_policy="cost",
    )
    app = create_app(settings)
    tiers = {model.id: model.tier for model in app.state.registry.models}
    headers = {"Authorization": f"Bearer {api_key}"}
    with TestClient(app) as client:
        first = [_ask(client, headers, row["prompt"]) for row in prompts]
        second = [_ask(client, headers, row["prompt"]) for row in prompts]
        paraphrases = [
            _ask(client, headers, row["paraphrase"]) for row in prompts if row["paraphrase"]
        ]
    return _summarize(prompts, first, second, paraphrases, tiers)


def _ask(client: TestClient, headers: dict[str, str], prompt: str) -> dict[str, Any]:
    response = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "auto",
            "policy": "cost",
            "temperature": 0,
            "max_tokens": 256,
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    response.raise_for_status()
    return response.json()


def _summarize(
    prompts: list[dict[str, str]],
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
    paraphrases: list[dict[str, Any]],
    tiers: dict[str, int],
) -> dict[str, Any]:
    correct = 0
    sufficient = 0
    escalated = 0
    latencies: list[float] = []
    cost = 0.0
    saved = 0.0
    by_label: dict[str, dict[str, float]] = {
        label: {"n": 0, "correct": 0, "cost": 0.0, "saved": 0.0} for label in _LABELS
    }
    for row, payload in zip(prompts, first, strict=True):
        gateway = payload["gateway"]
        label = row["label"]
        by_label[label]["n"] += 1
        if gateway["complexity"] == label:
            correct += 1
            by_label[label]["correct"] += 1
        model_tier = tiers.get(str(payload["model"]), 0)
        if model_tier >= label_tier(label):
            sufficient += 1
        if gateway["escalated"]:
            escalated += 1
        latencies.append(float(gateway["latency_ms"]))
        row_cost = float(gateway["cost_usd"])
        row_saved = float(gateway["cost_saved_usd"])
        cost += row_cost
        saved += row_saved
        by_label[label]["cost"] += row_cost
        by_label[label]["saved"] += row_saved
    exact_hits = sum(1 for payload in second if payload["gateway"]["cache_kind"] == "exact")
    semantic_hits = sum(
        1 for payload in paraphrases if payload["gateway"]["cache_kind"] == "semantic"
    )
    total = len(prompts)
    baseline = cost + saved
    per_label = {
        label: {
            "n": int(stats["n"]),
            "accuracy": (stats["correct"] / stats["n"]) if stats["n"] else 0.0,
            "savings_pct": (
                stats["saved"] / (stats["cost"] + stats["saved"]) * 100
                if stats["cost"] + stats["saved"]
                else 0.0
            ),
        }
        for label, stats in by_label.items()
    }
    return {
        "mode": "mock",
        "policy": "cost",
        "prompts": total,
        "routing_accuracy": correct / total if total else 0.0,
        "sufficient_model_rate": sufficient / total if total else 0.0,
        "escalation_rate": escalated / total if total else 0.0,
        "escalations": escalated,
        "exact_cache_hit_rate": exact_hits / total if total else 0.0,
        "semantic_prompts": len(paraphrases),
        "semantic_cache_hit_rate": (semantic_hits / len(paraphrases)) if paraphrases else 0.0,
        "latency_ms": {"p50": _percentile(latencies, 50), "p95": _percentile(latencies, 95)},
        "cost_usd": cost,
        "cost_saved_usd": saved,
        "cost_if_always_largest_usd": baseline,
        "savings_pct": (saved / baseline * 100) if baseline else 0.0,
        "by_label": per_label,
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def format_table(results: dict[str, Any]) -> str:
    latency = results["latency_ms"]
    lines = [
        f"prompts: {results['prompts']}",
        f"routing accuracy: {results['routing_accuracy']:.1%}",
        f"sufficient model rate: {results['sufficient_model_rate']:.1%}",
        f"escalation rate: {results['escalation_rate']:.1%}",
        f"exact cache hit rate (2nd pass): {results['exact_cache_hit_rate']:.1%}",
        f"semantic cache hit rate: {results['semantic_cache_hit_rate']:.1%} "
        f"({results['semantic_prompts']} paraphrases)",
        f"latency p50/p95 ms: {latency['p50']:.1f} / {latency['p95']:.1f}",
        f"savings vs always-largest: {results['savings_pct']:.1f}%",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m gateway.benchmark")
    parser.add_argument("--prompts", type=Path, default=default_prompts_path())
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results.json"))
    parser.add_argument("--no-latency", action="store_true")
    args = parser.parse_args(argv)
    results = run_benchmark(args.prompts, simulate_latency=not args.no_latency)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(format_table(results))
    print(f"wrote {args.output}")
