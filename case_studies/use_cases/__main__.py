"""Four short use cases for the joined ledger, beyond "what share is unowned".

    python -m case_studies.use_cases [--quick]

U1  Labeling Pareto      how few label fixes remove most unallocated spend
U2  Feature economics    cost per 1k requests per product feature, infra included
U3  Self-host break-even at what load a self-hosted 8B pod beats per-token API prices
U4  CI budget gate       `unalloc report --budget` as a deploy check
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

from case_studies import common
from case_studies.hybrid_e2e import workload
from case_studies.kv_cache import attribution as kv_attr
from case_studies.kv_cache.sim import simulate
from unalloc.cli import FIXTURE_DIR
from unalloc.core.attribute import attribute, unallocated_rows
from unalloc.core.models import CostRow
from unalloc.sources import REGISTRY

STUDY = "use_cases"
ZERO = Decimal("0")


def pareto(rows: list[CostRow], dimension: str) -> dict[str, Any]:
    """Unallocated share after fixing the k most expensive unlabeled rows."""
    report = attribute(rows, dimension)
    total = report.total_usd
    backlog = unallocated_rows(rows, dimension, limit=len(rows))
    curve = [float(report.unallocated_pct)]
    remaining = report.unallocated_usd
    for row in backlog:
        remaining -= row.amount_usd
        curve.append(float((remaining / total * 100).quantize(Decimal("0.1"))) if total else 0.0)
    fixes_to = {
        str(threshold): next((k for k, pct in enumerate(curve) if pct <= threshold), None)
        for threshold in (25, 10, 5)
    }
    return {
        "dimension": dimension,
        "rows": len(rows),
        "unlabeled_rows": len(backlog),
        "curve_pct": curve,
        "fixes_to_reach_pct": fixes_to,
        "top_fixes": [
            {"source": r.source, "name": r.name, "amount_usd": r.amount_usd} for r in backlog[:5]
        ],
    }


def fixture_rows() -> list[CostRow]:
    rows: list[CostRow] = []
    for name, filename in common.FIXTURE_FILES.items():
        rows.extend(REGISTRY[name]().from_fixture(FIXTURE_DIR / filename))
    return rows


def hybrid_rows(days: int) -> tuple[list[CostRow], dict[str, Any]]:
    data = workload.build(days=days)
    rows = REGISTRY["opencost"]().parse(data["opencost"])
    rows += REGISTRY["litellm"]().parse(data["litellm"])
    return rows, data


def feature_economics(rows: list[CostRow], days: int) -> list[dict[str, Any]]:
    """Joins cluster rows and gateway rows on `feature`, then divides by volume."""
    by_feature = attribute(rows, "feature")
    requests: dict[str, float] = {}
    api_usd: dict[str, Decimal] = {}
    for stream in workload.STREAMS:
        # Weekday/weekend mix used by the generator: 5/7 at 1.0, 2/7 at 0.45.
        requests[stream.feature] = requests.get(stream.feature, 0.0) + (
            stream.daily_requests * days * (5 + 2 * 0.45) / 7
        )
    for row in rows:
        if row.source == "litellm" and row.label("feature"):
            key = row.labels["feature"]
            api_usd[key] = api_usd.get(key, ZERO) + row.amount_usd
    out = []
    for bucket in by_feature.buckets:
        if bucket.is_unallocated or bucket.key not in requests:
            continue
        api = api_usd.get(bucket.key, ZERO)
        infra = bucket.amount_usd - api
        per_k = bucket.amount_usd / Decimal(str(requests[bucket.key] / 1000))
        out.append(
            {
                "feature": bucket.key,
                "total_usd": bucket.amount_usd,
                "api_usd": api,
                "infra_usd": infra,
                "infra_share": float(infra / bucket.amount_usd) if bucket.amount_usd else 0.0,
                "requests": round(requests[bucket.key]),
                "usd_per_1k_requests": per_k.quantize(Decimal("0.0001")),
                "api_only_usd_per_1k_requests": (
                    api / Decimal(str(requests[bucket.key] / 1000))
                ).quantize(Decimal("0.0001")),
            }
        )
    return out


# Illustrative per-million-token prices for two API tiers: (input, cached input, output).
API_TIERS = {
    "small_api_tier": (0.15, 0.075, 0.60),
    "mid_api_tier": (1.00, 0.10, 5.00),
}


def break_even(rates: tuple[float, ...], duration: float) -> list[dict[str, Any]]:
    """Cost of serving the same traffic on one self-hosted GPU vs per-token APIs.

    Self-hosted cost is wall-clock GPU time whether busy or not; API cost is the
    traffic's tokens at list price. Quality is not held equal - an 8B model is
    not a frontier API model - so this bounds cost, not value.
    """
    points = []
    for rate in rates:
        run = simulate(rate=rate, duration_s=duration)
        stats = kv_attr.tenant_stats(run)
        prompt = sum(s["prompt_tokens"] for s in stats.values())
        cached = sum(s["cached_prompt_tokens"] for s in stats.values())
        output = sum(s["completion_tokens"] for s in stats.values())
        self_hosted = run.wall_s / 3600 * common.GPU_HOUR_USD
        point: dict[str, Any] = {
            "rate_rps": rate,
            "busy_fraction": 1 - run.idle_s / run.wall_s,
            "output_tokens_per_s": output / run.wall_s,
            "self_hosted_usd_per_hour": self_hosted / run.wall_s * 3600,
            "self_hosted_usd_per_mtok_output": self_hosted / output * 1e6 if output else None,
        }
        for tier, (p_in, p_cached, p_out) in API_TIERS.items():
            api = ((prompt - cached) * p_in + cached * p_cached + output * p_out) / 1e6
            point[f"{tier}_usd_per_hour"] = api / run.wall_s * 3600
            point[f"{tier}_cost_ratio"] = self_hosted / api if api else None
        points.append(point)
    return points


def budget_gate() -> list[dict[str, Any]]:
    results = []
    for budget in ("50", "80"):
        proc = subprocess.run(
            [sys.executable, "-m", "unalloc.cli", "report", "--fixtures", "-D", "team",
             "--json", "--budget", budget],
            capture_output=True,
            text=True,
            check=False,
        )
        results.append({"budget_pct": budget, "exit_code": proc.returncode,
                        "stderr": proc.stderr.strip()})
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", type=Path, default=common.RESULTS / STUDY)
    args = parser.parse_args(argv)
    started = time.perf_counter()

    days = 7 if args.quick else 31
    rows, _ = hybrid_rows(days)
    rates = (0.5, 2.0, 6.0) if args.quick else (0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    metrics = {
        "study": STUDY,
        "quick": args.quick,
        "U1_pareto": {
            "bundled_fixtures": pareto(fixture_rows(), "team"),
            "hybrid_org": pareto(rows, "team"),
        },
        "U2_feature_economics": feature_economics(rows, days),
        "U3_break_even": {
            "gpu_hour_usd": common.GPU_HOUR_USD,
            "api_tiers_per_mtok": API_TIERS,
            "points": break_even(rates, 60.0 if args.quick else 300.0),
        },
        "U4_budget_gate": budget_gate(),
    }
    metrics["runtime_s"] = time.perf_counter() - started
    common.dump_json(args.out / "metrics.json", metrics)

    for name, p in metrics["U1_pareto"].items():
        print(f"U1 {name}: {p['curve_pct'][0]}% unallocated; fixes to reach "
              f"<=10%: {p['fixes_to_reach_pct']['10']} of {p['unlabeled_rows']} unlabeled rows")
    for f in metrics["U2_feature_economics"]:
        print(f"U2 {f['feature']:<14} ${f['usd_per_1k_requests']}/1k requests "
              f"(API-only ${f['api_only_usd_per_1k_requests']}, infra {f['infra_share']:.0%})")
    for p in metrics["U3_break_even"]["points"]:
        print(f"U3 rate {p['rate_rps']:>4} rps busy {p['busy_fraction']:.0%}: self-host/API "
              f"cost small {p['small_api_tier_cost_ratio']:.2f}x  "
              f"mid {p['mid_api_tier_cost_ratio']:.2f}x")
    for g in metrics["U4_budget_gate"]:
        print(f"U4 --budget {g['budget_pct']}: exit {g['exit_code']}")
    print(f"wrote {args.out / 'metrics.json'} in {metrics['runtime_s']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
