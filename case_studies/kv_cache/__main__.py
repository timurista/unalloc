"""Run the KV-cache attribution study.

    python -m case_studies.kv_cache            # full run, ~1-2 min
    python -m case_studies.kv_cache --quick    # smoke run, a few seconds
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path
from typing import Any

from case_studies import common
from case_studies.kv_cache import attribution as attr
from case_studies.kv_cache.sim import TENANTS, EngineConfig, RunResult, simulate

STUDY = "kv_cache"
MODEL = "llama-3.1-8b"
REPLICAS = 8
"""The simulated engine is one replica of an 8-GPU fleet behind a load balancer.
Shares are per replica; dollars scale by the fleet."""


def monthly_pool_usd() -> float:
    return REPLICAS * 720 * common.GPU_HOUR_USD


def run_summary(run: RunResult) -> dict[str, Any]:
    stats = attr.tenant_stats(run)
    prompt = sum(s["prompt_tokens"] for s in stats.values())
    cached = sum(s["cached_prompt_tokens"] for s in stats.values())
    return {
        "config": run.config,
        "wall_s": run.wall_s,
        "idle_s": run.idle_s,
        "busy_fraction": 1 - run.idle_s / run.wall_s,
        "steps": run.steps,
        "requests": len(run.requests),
        "dropped": run.dropped,
        "evictions": run.evictions,
        "preemptions": sum(s["preemptions"] for s in stats.values()),
        "cache_hit_rate": cached / prompt if prompt else 0.0,
        "output_tokens_per_s": run.tokens_generated / run.wall_s,
        "cache_reserve_fraction": run.cache_reserve_block_s
        / (run.config["engine"]["num_blocks"] * run.wall_s),
        "tenants": stats,
        "shares": attr.shares(run, stats),
    }


def emit_allocations(outdir: Path, split: dict[str, float], *, redistribute: bool) -> None:
    pool = monthly_pool_usd()
    shares = attr.redistribute(split) if redistribute else split
    direct = attr.redistribute(split)
    overhead = split.get(attr.OVERHEAD, 0.0)
    allocations = []
    for tenant, share in shares.items():
        if tenant == attr.OVERHEAD:
            if share <= 0:
                continue
            labels = {"environment": "prod", "costCenter": "CC-4471"}
            name = f"inference/vllm-{MODEL}/{attr.OVERHEAD}"
            gpu, shared = pool * share, 0.0
        else:
            team = attr.team_of(TENANTS, tenant)
            labels = {"environment": "prod", "costCenter": "CC-4471"}
            if team:
                labels["team"] = team
            name = f"inference/vllm-{MODEL}/{tenant}"
            if redistribute:
                # Direct usage on gpuCost, the redistributed overhead on sharedCost,
                # which is how OpenCost itself reports shared spend.
                gpu = pool * split[tenant]
                shared = pool * overhead * direct[tenant]
            else:
                gpu, shared = pool * share, 0.0
        allocations.append(
            common.opencost_allocation(
                name,
                namespace="inference",
                controller=f"vllm-{MODEL}",
                controller_kind="statefulset",
                labels=labels,
                gpu_cost=gpu,
                shared_cost=shared,
                model=MODEL,
            )
        )
    common.write_payloads(outdir, opencost=common.opencost_payload(allocations))


def emit_baseline(outdir: Path) -> None:
    pod = common.opencost_allocation(
        f"inference/vllm-{MODEL}",
        namespace="inference",
        controller=f"vllm-{MODEL}",
        controller_kind="statefulset",
        labels={"environment": "prod", "costCenter": "CC-4471"},
        gpu_cost=monthly_pool_usd(),
        gpu_hours=REPLICAS * 720,
        model=MODEL,
    )
    common.write_payloads(outdir, opencost=common.opencost_payload([pod]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="short smoke run")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=common.RESULTS / STUDY)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    duration = 120.0 if args.quick else 1800.0
    sweep_duration = 60.0 if args.quick else 600.0
    # 3 req/s: below the saturation knee over a long run. At 4 req/s agent
    # follow-up turns accumulate and median TTFT climbs past 40 s over 30
    # minutes, even though a 5-minute run looks healthy.
    main_rate = 3.0
    sweep_rates = (0.5, 2.0, 4.0) if args.quick else (0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0)

    run = simulate(rate=main_rate, duration_s=duration, seed=args.seed)
    headline = run_summary(run)
    split = headline["shares"]

    outdir: Path = args.out
    unalloc: dict[str, Any] = {}
    emit_baseline(outdir / "baseline")
    rows = common.load_rows(outdir / "baseline", ["opencost"])
    unalloc["baseline"] = common.summarize(rows, "team")
    for method in attr.METHODS:
        for redistribute in (False, True):
            if redistribute and split[method][attr.OVERHEAD] <= 0:
                continue
            variant = f"{method}{'_redistributed' if redistribute else ''}"
            emit_allocations(outdir / variant, split[method], redistribute=redistribute)
            rows = common.load_rows(outdir / variant, ["opencost"])
            unalloc[variant] = common.summarize(rows, "team")

    sweep = []
    for caching in (True, False):
        for rate in sweep_rates:
            result = simulate(
                rate=rate,
                duration_s=sweep_duration,
                engine=EngineConfig(prefix_caching=caching),
                seed=args.seed,
            )
            summary = run_summary(result)
            sweep.append(
                {
                    "prefix_caching": caching,
                    "rate_rps": rate,
                    "busy_fraction": summary["busy_fraction"],
                    "cache_hit_rate": summary["cache_hit_rate"],
                    "preemptions": summary["preemptions"],
                    "output_tokens_per_s": summary["output_tokens_per_s"],
                    "overhead": {m: summary["shares"][m][attr.OVERHEAD] for m in attr.METHODS},
                    "shares_redistributed": {
                        m: attr.redistribute(summary["shares"][m]) for m in attr.METHODS
                    },
                    "ttft_p50_s": {t: s["ttft_p50_s"] for t, s in summary["tenants"].items()},
                }
            )

    metrics = {
        "study": STUDY,
        "quick": args.quick,
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
        "assumptions": {
            "gpu_hour_usd": common.GPU_HOUR_USD,
            "replicas": REPLICAS,
            "monthly_pool_usd": monthly_pool_usd(),
            "main_rate_rps": main_rate,
            "tenants": [t.__dict__ for t in TENANTS],
            "list_price_weights": {"input": 1.0, "cached_input": 0.1, "output": 4.0},
        },
        "headline": headline,
        "divergence": {
            "tokens_vs_compute": attr.divergence(split["tokens"], split["compute"]),
            "tokens_vs_memory": attr.divergence(split["tokens"], split["memory"]),
            "list_price_vs_blended": attr.divergence(split["list_price"], split["blended"]),
        },
        "unalloc": {k: {**v, "buckets": v["buckets"]} for k, v in unalloc.items()},
        "sweep": sweep,
        "runtime_s": time.perf_counter() - started,
    }
    common.dump_json(outdir / "metrics.json", metrics)

    print(f"kv_cache: {len(run.requests)} requests over {run.wall_s:,.0f}s simulated, "
          f"busy {headline['busy_fraction']:.0%}, prefix-cache hits "
          f"{headline['cache_hit_rate']:.0%}, {headline['preemptions']} preemptions")
    for tenant, s in headline["tenants"].items():
        print(f"  {tenant:<9} TTFT p50 {s['ttft_p50_s']:.3f}s  p95 {s['ttft_p95_s']:.3f}s")
    print(f"{'tenant':<10}" + "".join(f"{m:>12}" for m in attr.METHODS))
    for tenant in [*headline["tenants"], attr.OVERHEAD]:
        print(f"{tenant:<10}" + "".join(f"{split[m][tenant]:>12.1%}" for m in attr.METHODS))
    for key, value in metrics["divergence"].items():
        print(f"divergence {key}: {value['tenant']} {value['share_points']:+.1f} pts")
    for variant, summary in unalloc.items():
        print(f"unalloc {variant:<26} {summary['unallocated_pct']}% unallocated")
    print(f"wrote {outdir / 'metrics.json'} in {metrics['runtime_s']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
