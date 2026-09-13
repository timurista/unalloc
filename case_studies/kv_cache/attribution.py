"""Turn a simulated serving run into showback under competing metering rules.

Every method answers the same question - what fraction of the pod's GPU bill
belongs to each tenant - from a different meter:

* ``tokens``      raw prompt + completion tokens (what most gateways count)
* ``list_price``  tokens weighted like API list prices: input 1, cached input
                  0.1, output 4 (what a per-token internal price sheet does)
* ``compute``     measured share of engine-step time; idle time is overhead
* ``memory``      share of KV block-seconds; unheld blocks are overhead
* ``blended``     half compute, half memory

Token methods have no notion of overhead: they spread idle GPUs and empty KV
pool across tenants silently. Measured methods expose overhead as its own
number, which is then either left unlabeled (and shows up in unalloc as
unallocated) or redistributed explicitly as shared cost.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from case_studies.kv_cache.sim import RunResult, TenantSpec

METHODS = ("tokens", "list_price", "compute", "memory", "blended")
OVERHEAD = "__overhead__"


def tenant_stats(run: RunResult) -> dict[str, dict[str, Any]]:
    groups: dict[str, list] = defaultdict(list)
    for req in run.requests:
        groups[req.tenant.name].append(req)
    stats: dict[str, dict[str, Any]] = {}
    for name, reqs in groups.items():
        ttft = [r.first_token_at - r.arrival for r in reqs if r.first_token_at is not None]
        e2e = [r.finished_at - r.arrival for r in reqs if r.finished_at is not None]
        prompt = sum(r.prompt_len for r in reqs)
        cached = sum(min(r.cached_tokens, r.prompt_len) for r in reqs)
        stats[name] = {
            "team": reqs[0].tenant.team,
            "requests": len(reqs),
            "prompt_tokens": prompt,
            "cached_prompt_tokens": cached,
            "completion_tokens": sum(r.output_len for r in reqs),
            "cache_hit_rate": cached / prompt if prompt else 0.0,
            "compute_s": sum(r.compute_s for r in reqs),
            "kv_block_s": sum(r.kv_block_s for r in reqs),
            "preemptions": sum(r.preemptions for r in reqs),
            "recomputed_tokens": sum(r.recomputed_tokens for r in reqs),
            "ttft_p50_s": statistics.median(ttft) if ttft else None,
            "ttft_p95_s": _pct(ttft, 0.95),
            "e2e_p50_s": statistics.median(e2e) if e2e else None,
        }
    return dict(sorted(stats.items()))


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def shares(run: RunResult, stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    """method -> {tenant | OVERHEAD: fraction of the pod bill}. Each sums to 1."""
    pool_block_s = run.config["engine"]["num_blocks"] * run.wall_s
    out: dict[str, dict[str, float]] = {}

    def normalise(raw: dict[str, float]) -> dict[str, float]:
        total = sum(raw.values())
        return {k: v / total for k, v in raw.items()}

    out["tokens"] = normalise(
        {t: s["prompt_tokens"] + s["completion_tokens"] for t, s in stats.items()}
    ) | {OVERHEAD: 0.0}
    out["list_price"] = normalise(
        {
            t: (s["prompt_tokens"] - s["cached_prompt_tokens"])
            + 0.1 * s["cached_prompt_tokens"]
            + 4.0 * s["completion_tokens"]
            for t, s in stats.items()
        }
    ) | {OVERHEAD: 0.0}
    out["compute"] = {t: s["compute_s"] / run.wall_s for t, s in stats.items()}
    out["compute"][OVERHEAD] = 1.0 - sum(out["compute"].values())
    out["memory"] = {t: s["kv_block_s"] / pool_block_s for t, s in stats.items()}
    out["memory"][OVERHEAD] = 1.0 - sum(out["memory"].values())
    out["blended"] = {
        k: 0.5 * out["compute"][k] + 0.5 * out["memory"][k] for k in out["compute"]
    }
    return out


def redistribute(split: dict[str, float]) -> dict[str, float]:
    """Spread overhead over tenants in proportion to their direct share."""
    direct = {k: v for k, v in split.items() if k != OVERHEAD}
    total = sum(direct.values())
    return {k: v / total for k, v in direct.items()}


def divergence(split_a: dict[str, float], split_b: dict[str, float]) -> dict[str, Any]:
    """Largest per-tenant disagreement between two methods, overhead redistributed."""
    a, b = redistribute(split_a), redistribute(split_b)
    deltas = {k: a[k] - b[k] for k in a}
    worst = max(deltas, key=lambda k: abs(deltas[k]))
    return {"tenant": worst, "share_points": deltas[worst] * 100, "all": deltas}


def team_of(tenants: tuple[TenantSpec, ...], name: str) -> str | None:
    return next(t.team for t in tenants if t.name == name)
