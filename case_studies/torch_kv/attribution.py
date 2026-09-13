"""Apportion one fixed GPU bill to callers, several ways, and hand each split to unalloc.

OpenCost sees the inference deployment as one pod with one bill. Whoever wants
showback has to pick an apportionment key. This module computes the split
under each candidate key from the *measured* serving data, then emits the
split as OpenCost allocations so unalloc reports on it exactly as it would on
a real cluster.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from case_studies import common
from case_studies.torch_kv.trace import PROFILES

# --- the cost pool ----------------------------------------------------------------

GPU_COUNT = 8
HOURS_PER_MONTH = 720
"""30 x 24. The study's pool is a flat monthly reservation, not the 744 h of
common.WINDOW_START..WINDOW_END; only shares matter for the comparison."""

POOL_GPU_HOURS = GPU_COUNT * HOURS_PER_MONTH
POOL_USD = (Decimal(POOL_GPU_HOURS) * Decimal(str(common.GPU_HOUR_USD))).quantize(Decimal("0.01"))

NAMESPACE = "inference"
CONTROLLER = "torch-kv-tiny"
MODEL_NAME = "torch-kv-tiny-4l-256d"
BASE_LABELS = {"environment": "prod", "costCenter": "CC-4471"}
"""Same shape as the real vLLM pod in unalloc's fixture: a cost center, no team."""

COMPUTE_WEIGHT = 0.5
MEMORY_WEIGHT = 0.5
"""Blended split. Even weights: GPU serving is bound by both FLOPs and by the
HBM that KV caches occupy (which caps batch size), and there is no neutral
reason to prefer one."""

LIST_PRICE_OUTPUT_MULTIPLIER = 4
"""Output tokens priced at 4x input tokens, the typical ratio on provider price
sheets. A sensitivity check on the plain per-token method."""

METHODS: tuple[str, ...] = (
    "requests",
    "tokens",
    "tokens_list_price",
    "compute_seconds",
    "kv_byte_seconds",
    "blended",
    "analytic_flops",
)
MEASURED_METHODS: tuple[str, ...] = ("compute_seconds", "kv_byte_seconds", "blended")
TOKEN_METHODS: tuple[str, ...] = ("tokens", "tokens_list_price")

METHOD_NOTES = {
    "requests": "equal cost per request",
    "tokens": "prompt + completion tokens; what a flat per-token gateway price does",
    "tokens_list_price": f"prompt + {LIST_PRICE_OUTPUT_MULTIPLIER} x completion tokens",
    "compute_seconds": "measured prefill + decode seconds, shared-prefix prefill spread by hits",
    "kv_byte_seconds": "measured KV cache occupancy integrated over time (memory-time)",
    "blended": f"{COMPUTE_WEIGHT} x compute share + {MEMORY_WEIGHT} x memory share",
    "analytic_flops": "Kaplan forward FLOPs (reference, hardware independent)",
}


def caller_usage(
    requests: Sequence[Mapping[str, Any]],
    *,
    shared_prefix_prefill_s: float,
    shared_prefix_kv_byte_seconds: float,
) -> dict[str, dict[str, float]]:
    """Raw usage per caller in every method's native unit.

    The shared system-prompt prefix is computed once and reused, so its cost
    belongs to nobody in particular. It is spread over the callers that hit
    it, by hit count: the least-opinionated choice, and stated here because it
    moves `search` and `agents` slightly.
    """
    usage: dict[str, dict[str, float]] = {
        p.caller: {m: 0.0 for m in (*METHODS, "prefix_hits")} for p in PROFILES
    }
    for r in requests:
        u = usage[r["caller"]]
        u["requests"] += 1
        u["tokens"] += r["prompt_tokens"] + r["completion_tokens"]
        u["tokens_list_price"] += (
            r["prompt_tokens"] + LIST_PRICE_OUTPUT_MULTIPLIER * r["completion_tokens"]
        )
        u["compute_seconds"] += r["prefill_s"] + r["decode_s"]
        u["kv_byte_seconds"] += r["kv_byte_seconds"]
        u["analytic_flops"] += r["analytic_flops"]
        u["prefix_hits"] += 1 if r["shared_prefix"] else 0
    hits = sum(u["prefix_hits"] for u in usage.values())
    if hits:
        for u in usage.values():
            frac = u["prefix_hits"] / hits
            u["compute_seconds"] += shared_prefix_prefill_s * frac
            u["kv_byte_seconds"] += shared_prefix_kv_byte_seconds * frac
    return usage


def shares_by_method(usage: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for method in METHODS:
        if method == "blended":
            continue
        total = sum(u[method] for u in usage.values())
        out[method] = {c: (u[method] / total if total else 0.0) for c, u in usage.items()}
    out["blended"] = {
        c: COMPUTE_WEIGHT * out["compute_seconds"][c] + MEMORY_WEIGHT * out["kv_byte_seconds"][c]
        for c in usage
    }
    return {m: out[m] for m in METHODS}


def split_usd(pool: Decimal, weights: Mapping[str, float]) -> dict[str, Decimal]:
    """Split `pool` to the cent by largest remainder, so the parts sum exactly to the pool."""
    dec = {k: Decimal(repr(float(v))) for k, v in weights.items()}
    total = sum(dec.values(), Decimal("0"))
    cents = int((pool * 100).to_integral_value())
    raw = {k: v / total * cents for k, v in dec.items()}
    parts = {k: int(v) for k, v in raw.items()}
    for key in sorted(raw, key=lambda k: (raw[k] - parts[k], k), reverse=True)[
        : cents - sum(parts.values())
    ]:
        parts[key] += 1
    return {k: (Decimal(c) / 100).quantize(Decimal("0.01")) for k, c in parts.items()}


def divergence(
    shares: Mapping[str, Mapping[str, float]], pool: Decimal = POOL_USD
) -> dict[str, Any]:
    """Max absolute share gap between each token method and each measured method."""
    out: dict[str, Any] = {}
    for ref in TOKEN_METHODS:
        per: dict[str, Any] = {}
        for method in MEASURED_METHODS:
            diffs = {c: shares[method][c] - shares[ref][c] for c in shares[ref]}
            worst = max(diffs, key=lambda c: abs(diffs[c]))
            per[method] = {
                "max_abs_share_diff": abs(diffs[worst]),
                "caller": worst,
                "usd": float(pool) * abs(diffs[worst]),
                "signed_share_diff": diffs,
            }
        top = max(per, key=lambda m: per[m]["max_abs_share_diff"])
        out[ref] = {
            "by_method": per,
            "headline": {
                "method": top,
                "caller": per[top]["caller"],
                "max_abs_share_diff": per[top]["max_abs_share_diff"],
                "usd": per[top]["usd"],
            },
        }
    return out


# --- unalloc ------------------------------------------------------------------------


def _baseline_allocation() -> dict[str, Any]:
    return common.opencost_allocation(
        f"{NAMESPACE}/{CONTROLLER}",
        namespace=NAMESPACE,
        controller=CONTROLLER,
        controller_kind="statefulset",
        labels=BASE_LABELS,
        gpu_cost=float(POOL_USD),
        gpu_hours=float(POOL_GPU_HOURS),
        model=MODEL_NAME,
    )


def _split_allocations(dollars: Mapping[str, Decimal]) -> list[dict[str, Any]]:
    total = sum(dollars.values(), Decimal("0"))
    team_of = {p.caller: p.team for p in PROFILES}
    allocations = []
    for caller, amount in dollars.items():
        labels = dict(BASE_LABELS)
        if team_of[caller]:
            labels["team"] = team_of[caller]
        allocations.append(
            common.opencost_allocation(
                # Names must be unique: opencost_payload keys the window by name.
                f"{NAMESPACE}/{CONTROLLER}:{caller}",
                namespace=NAMESPACE,
                controller=CONTROLLER,
                controller_kind="statefulset",
                labels=labels,
                gpu_cost=float(amount),
                gpu_hours=POOL_GPU_HOURS * float(amount / total) if total else 0.0,
                model=MODEL_NAME,
            )
        )
    return allocations


def run_unalloc(
    outdir: Path, shares: Mapping[str, Mapping[str, float]], dimension: str = "team"
) -> dict[str, Any]:
    """Write the baseline and every per-method split, then attribute each through unalloc."""
    results: dict[str, Any] = {}
    base_dir = outdir / "baseline"
    common.write_payloads(base_dir, opencost=common.opencost_payload([_baseline_allocation()]))
    results["baseline"] = common.summarize(common.load_rows(base_dir, ["opencost"]), dimension)

    for method, weights in shares.items():
        method_dir = outdir / method
        dollars = split_usd(POOL_USD, weights)
        common.write_payloads(
            method_dir, opencost=common.opencost_payload(_split_allocations(dollars))
        )
        results[method] = common.summarize(common.load_rows(method_dir, ["opencost"]), dimension)
    for summary in results.values():
        summary["total_matches_pool"] = Decimal(str(summary["total_usd"])) == POOL_USD
    return results
