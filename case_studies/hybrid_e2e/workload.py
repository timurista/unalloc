"""One month of AI spend for a mid-size org, generated deterministically.

The org runs a LiteLLM gateway in front of OpenAI and Anthropic, a self-hosted
vLLM fleet on Kubernetes, and - like every org - some traffic that bypasses the
gateway: a research team calling providers directly from notebooks.

Because the providers see *all* traffic, their billing APIs are the ground truth
invoice. The gateway sees only gateway traffic, but with team labels. That gap
is what the scenarios in `__main__` measure.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from case_studies import common

# Illustrative per-million-token list prices (input, output). Order of magnitude
# only; the study reports shares and deltas, not absolute bills.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
PROVIDER = {
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "claude-sonnet-4-6": "anthropic",
    "claude-haiku-4-5": "anthropic",
}


@dataclass(frozen=True)
class Stream:
    """A steady daily flow of gateway traffic from one key."""

    team: str | None
    key_alias: str
    feature: str
    model: str
    daily_requests: int
    prompt_tokens: int
    completion_tokens: int


STREAMS: tuple[Stream, ...] = (
    Stream("search", "search-rerank", "rerank", "gpt-4o-mini", 42_000, 1_900, 40),
    Stream("search", "search-answer", "answer", "claude-sonnet-4-6", 9_000, 3_200, 280),
    Stream("agents", "agent-runtime", "agent-loop", "claude-sonnet-4-6", 6_500, 7_800, 900),
    Stream("agents", "agent-runtime", "tool-router", "claude-haiku-4-5", 18_000, 900, 60),
    Stream("platform", "evals", "nightly-evals", "gpt-4o", 3_000, 2_400, 500),
    Stream(None, "sandbox-shared", "playground", "gpt-4o", 900, 1_500, 700),
    Stream(None, "unknown-legacy-key", "unknown", "claude-sonnet-4-6", 1_100, 2_600, 800),
)

# Direct provider usage from notebooks: never touches the gateway.
BYPASS_DAILY_USD = {"openai": 58.0, "anthropic": 91.0}


def _cost(model: str, prompt: int, completion: int) -> float:
    price_in, price_out = PRICES[model]
    return (prompt * price_in + completion * price_out) / 1_000_000


def build(seed: int = 11, days: int = 31) -> dict[str, Any]:
    rng = random.Random(seed)
    litellm: list[dict[str, Any]] = []
    # provider -> day -> line_item -> usd
    provider_days: dict[str, list[dict[str, float]]] = {
        "openai": [{} for _ in range(days)],
        "anthropic": [{} for _ in range(days)],
    }

    for day in range(days):
        start = common.WINDOW_START + timedelta(days=day)
        weekday = start.weekday() < 5
        for index, stream in enumerate(STREAMS):
            volume = stream.daily_requests * (1.0 if weekday else 0.45) * rng.uniform(0.85, 1.15)
            prompt = int(volume * stream.prompt_tokens)
            completion = int(volume * stream.completion_tokens)
            spend = _cost(stream.model, prompt, completion)
            tags = [f"feature:{stream.feature}", "environment:prod"]
            litellm.append(
                common.litellm_record(
                    f"agg-{day:02d}-{index}",
                    model=stream.model,
                    spend=spend,
                    prompt_tokens=prompt,
                    completion_tokens=completion,
                    start=start,
                    duration_s=86_400,
                    team_id=stream.team,
                    api_key_alias=stream.key_alias,
                    tags=tags,
                    metadata={"aggregation": "daily"},
                )
            )
            provider = PROVIDER[stream.model]
            items = provider_days[provider][day]
            items[stream.model] = items.get(stream.model, 0.0) + spend

    openai_buckets = []
    anthropic_buckets = []
    for day in range(days):
        start = common.WINDOW_START + timedelta(days=day)
        end = start + timedelta(days=1)
        openai_buckets.append(
            {
                "object": "bucket",
                "start_time": int(start.timestamp()),
                "end_time": int(end.timestamp()),
                "results": [
                    *(
                        {
                            "object": "organization.costs.result",
                            "amount": {"value": round(usd, 6), "currency": "usd"},
                            "line_item": model,
                            "project_id": "proj_gateway",
                            "project_name": "gateway-prod",
                        }
                        for model, usd in sorted(provider_days["openai"][day].items())
                    ),
                    {
                        "object": "organization.costs.result",
                        "amount": {
                            "value": round(BYPASS_DAILY_USD["openai"] * rng.uniform(0.6, 1.4), 6),
                            "currency": "usd",
                        },
                        "line_item": "gpt-4o",
                        "project_id": "proj_research",
                        "project_name": "research-notebooks",
                    },
                ],
            }
        )
        anthropic_buckets.append(
            {
                "starting_at": common.iso(start),
                "ending_at": common.iso(end),
                "results": [
                    *(
                        {
                            "amount": f"{usd:.6f}",
                            "currency": "USD",
                            "description": model,
                            "workspace_id": "wrkspc_gateway",
                            "workspace_name": "gateway",
                        }
                        for model, usd in sorted(provider_days["anthropic"][day].items())
                    ),
                    {
                        "amount": f"{BYPASS_DAILY_USD['anthropic'] * rng.uniform(0.6, 1.4):.6f}",
                        "currency": "USD",
                        "description": "claude-sonnet-4-6",
                        "workspace_id": "wrkspc_research",
                        "workspace_name": "ml-research",
                    },
                ],
            }
        )

    opencost = common.opencost_payload(
        [
            common.opencost_allocation(
                "inference/vllm-llama-70b",
                namespace="inference",
                controller="vllm-llama-70b",
                controller_kind="statefulset",
                labels={"environment": "prod", "costCenter": "CC-4471"},
                gpu_cost=8 * 720 * common.GPU_HOUR_USD * 0.92,
                cpu_cost=290.0,
                ram_cost=144.2,
                gpu_hours=8 * 720,
                model="llama-3.3-70b",
            ),
            common.opencost_allocation(
                "inference/llm-d-gateway",
                namespace="inference",
                controller="llm-d-gateway",
                labels={"team": "platform", "environment": "prod"},
                cpu_cost=412.18,
                ram_cost=188.44,
            ),
            common.opencost_allocation(
                "search/vector-db",
                namespace="search",
                controller="qdrant",
                controller_kind="statefulset",
                labels={"team": "search", "feature": "answer"},
                cpu_cost=1_240.0,
                ram_cost=860.0,
            ),
            common.opencost_allocation(
                "batch/embeddings-worker",
                namespace="batch",
                controller="embeddings-worker",
                controller_kind="cronjob",
                labels={},
                cpu_cost=980.4,
                ram_cost=402.1,
                gpu_cost=3_120.0,
                gpu_hours=1_040,
            ),
            common.opencost_allocation(
                "__idle__",
                namespace="",
                controller="",
                labels={},
                cpu_cost=610.0,
                ram_cost=220.0,
                gpu_cost=5_240.0,
            ),
        ]
    )

    return {
        "opencost": opencost,
        "litellm": common.litellm_payload(litellm),
        "openai_buckets": openai_buckets,
        "anthropic_buckets": anthropic_buckets,
        "invoice": {
            "openai": sum(
                float(r["amount"]["value"]) for b in openai_buckets for r in b["results"]
            ),
            "anthropic": sum(float(r["amount"]) for b in anthropic_buckets for r in b["results"]),
        },
        "bypass_usd": {
            "openai": sum(
                float(r["amount"]["value"])
                for b in openai_buckets
                for r in b["results"]
                if r["project_id"] == "proj_research"
            ),
            "anthropic": sum(
                float(r["amount"])
                for b in anthropic_buckets
                for r in b["results"]
                if r["workspace_id"] == "wrkspc_research"
            ),
        },
    }
