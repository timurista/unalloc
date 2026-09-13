"""Orchestration: correctness check, E1, E2, attribution, unalloc, metrics.json."""

from __future__ import annotations

import os
import platform
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import torch

from case_studies import common
from case_studies.torch_kv import attribution as attr
from case_studies.torch_kv.experiments import run_scaling, serve_trace
from case_studies.torch_kv.model import ModelConfig, TinyDecoder, equivalence_check
from case_studies.torch_kv.trace import PROFILES, build_trace

STUDY = "torch_kv"


@dataclass(frozen=True, slots=True)
class Settings:
    contexts: tuple[int, ...]
    warmup: int
    repeats_cached: int
    repeats_uncached: int
    n_requests: int
    trace_repeats: int


FULL = Settings((32, 64, 128, 256, 512, 1024), 3, 50, 15, 96, 3)
QUICK = Settings((32, 128, 512), 1, 10, 3, 20, 1)


def _round(value: Any, digits: int = 6) -> Any:
    """Trim float noise so metrics.json stays small and diffable."""
    if isinstance(value, float):
        return float(f"{value:.{digits}g}")
    if isinstance(value, dict):
        return {k: _round(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [_round(v, digits) for v in value]
    return value


def _environment(threads: int) -> dict[str, Any]:
    return {
        "torch_version": torch.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "requested_threads": threads,
        "device": "cpu",
    }


def _caller_summary(
    usage: dict[str, dict[str, float]], requests: list[dict[str, Any]]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in PROFILES:
        rows = [r for r in requests if r["caller"] == p.caller]
        n = len(rows) or 1
        out[p.caller] = {
            "team": p.team,
            "shape": p.shape,
            "shared_prefix": p.shared_prefix,
            "requests": len(rows),
            "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
            "completion_tokens": sum(r["completion_tokens"] for r in rows),
            "prefill_s": sum(r["prefill_s"] for r in rows),
            "decode_s": sum(r["decode_s"] for r in rows),
            "mean_prefill_s": sum(r["prefill_s"] for r in rows) / n,
            "mean_decode_s": sum(r["decode_s"] for r in rows) / n,
            "kv_byte_seconds": usage[p.caller]["kv_byte_seconds"],
            "compute_seconds": usage[p.caller]["compute_seconds"],
            "analytic_flops": usage[p.caller]["analytic_flops"],
        }
    return out


def _repeat_spread(serving: dict[str, Any], wall_s: float) -> dict[str, Any]:
    """Share range across independent serving repeats: how much timing noise moves a split."""
    per_run = []
    for run in serving["runs"]:
        usage = attr.caller_usage(
            run,
            shared_prefix_prefill_s=serving["shared_prefix_prefill_s"],
            shared_prefix_kv_byte_seconds=serving["shared_prefix_kv_bytes"] * wall_s,
        )
        per_run.append(attr.shares_by_method(usage))
    return {
        method: {
            c: max(s[method][c] for s in per_run) - min(s[method][c] for s in per_run)
            for c in per_run[0][method]
        }
        for method in attr.MEASURED_METHODS
    }


def run(out: Path, *, quick: bool = False, seed: int = 7, threads: int = 4) -> dict[str, Any]:
    """Run the whole study into `out` and return the metrics that were written."""
    started = time.perf_counter()
    settings = QUICK if quick else FULL
    torch.set_num_threads(threads)
    out.mkdir(parents=True, exist_ok=True)

    cfg = ModelConfig()
    model = TinyDecoder(cfg, seed=seed)
    correctness = equivalence_check(model, seed=seed)

    scaling = run_scaling(
        model,
        list(settings.contexts),
        warmup=settings.warmup,
        repeats_cached=settings.repeats_cached,
        repeats_uncached=settings.repeats_uncached,
        seed=seed,
    )

    trace = build_trace(settings.n_requests, seed)
    serving = serve_trace(model, trace, repeats=settings.trace_repeats)
    requests = serving["requests"]
    # The server holds the shared prefix cache for the whole serving run.
    prefix_kv_bs = serving["shared_prefix_kv_bytes"] * serving["serve_wall_s"]
    usage = attr.caller_usage(
        requests,
        shared_prefix_prefill_s=serving["shared_prefix_prefill_s"],
        shared_prefix_kv_byte_seconds=prefix_kv_bs,
    )
    shares = attr.shares_by_method(usage)
    dollars = {m: attr.split_usd(attr.POOL_USD, w) for m, w in shares.items()}
    tokens = {c: usage[c]["tokens"] for c in usage}
    usd_per_1k_tokens = {
        m: {c: float(d[c]) / tokens[c] * 1000 if tokens[c] else None for c in d}
        for m, d in dollars.items()
    }
    div = attr.divergence(shares)
    summaries = attr.run_unalloc(out, shares)

    runtime = time.perf_counter() - started
    serving_meta = {k: v for k, v in serving.items() if k not in ("requests", "runs")}
    metrics: dict[str, Any] = {
        "study": STUDY,
        "mode": "quick" if quick else "full",
        "seed": seed,
        "environment": _environment(threads),
        "model": {
            **cfg.as_dict(),
            "n_params": model.n_params(),
            "n_params_non_embedding": model.n_params(non_embedding=True),
            "weights": "random, N(0, 0.02), fixed seed; cost is independent of weight values",
        },
        "correctness": correctness,
        "e1_scaling": scaling,
        "e2_trace": {
            **serving_meta,
            "shared_prefix_kv_byte_seconds": prefix_kv_bs,
            "n_requests": len(requests),
            "profiles": [
                {
                    "caller": p.caller,
                    "team": p.team,
                    "weight": p.weight,
                    "prompt_range": list(p.prompt_range),
                    "completion_range": list(p.completion_range),
                    "shared_prefix": p.shared_prefix,
                    "shape": p.shape,
                }
                for p in PROFILES
            ],
            "by_caller": _caller_summary(usage, requests),
            "requests": requests,
        },
        "attribution": {
            "pool": {
                "gpu_count": attr.GPU_COUNT,
                "hours": attr.HOURS_PER_MONTH,
                "gpu_hour_usd": common.GPU_HOUR_USD,
                "gpu_hours": attr.POOL_GPU_HOURS,
                "usd": attr.POOL_USD,
            },
            "blend_weights": {"compute": attr.COMPUTE_WEIGHT, "memory": attr.MEMORY_WEIGHT},
            "methods": list(attr.METHODS),
            "method_notes": attr.METHOD_NOTES,
            "measured_methods": list(attr.MEASURED_METHODS),
            "callers": [p.caller for p in PROFILES],
            "shares": shares,
            "usd": dollars,
            "usd_per_1k_tokens": usd_per_1k_tokens,
            "measured_share_spread_across_repeats": _repeat_spread(
                serving, serving["serve_wall_s"]
            ),
            "divergence": div,
            "unallocated_pct_by_method": {
                m: Decimal(str(s["unallocated_pct"])) for m, s in summaries.items()
            },
        },
        "unalloc": summaries,
        "runtime_s": runtime,
    }
    metrics = _round(metrics)
    common.dump_json(out / "metrics.json", metrics)
    return metrics
