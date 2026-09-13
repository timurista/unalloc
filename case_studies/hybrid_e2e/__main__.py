"""Run unalloc's real CLI, as a subprocess, against a live mock provider stack.

    python -m case_studies.hybrid_e2e [--quick]

Scenarios:
  A  gateway + cluster          the recommended setup: opencost + litellm
  B  every source               what happens if you turn everything on
  C  provider billing only      openai + anthropic, with and without fallbacks
  R  reconciliation             ledgers vs the providers' own totals
  P  pagination counterfactual  what reading only page one would have reported
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from case_studies import common
from case_studies.hybrid_e2e import servers, workload
from unalloc.core.models import Invoice
from unalloc.core.reconcile import reconcile
from unalloc.sources import REGISTRY

STUDY = "hybrid_e2e"


def cli(stack: servers.Stack, *args: str) -> dict[str, Any]:
    env = {**os.environ, **stack.env()}
    proc = subprocess.run(
        [sys.executable, "-m", "unalloc.cli", *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"unalloc {' '.join(args)} failed:\n{proc.stdout}\n{proc.stderr}")
    return json.loads(proc.stdout)


def brief(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_usd": report["total_usd"],
        "unallocated_usd": report["unallocated_usd"],
        "unallocated_pct": report["unallocated_pct"],
        "by_source": report["by_source"],
        "buckets": [
            {"key": b["key"], "amount_usd": b["amount_usd"], "sources": list(b["by_source"])}
            for b in report["buckets"]
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="one week of data instead of a month")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", type=Path, default=common.RESULTS / STUDY)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    data = workload.build(seed=args.seed, days=7 if args.quick else 31)
    now = datetime.now(tz=UTC)
    results: dict[str, Any] = {}

    with servers.serve(data) as stack:
        a = cli(stack, "report", "--json", "-s", "opencost", "-s", "litellm", "-D", "team")
        b = cli(stack, "report", "--json", "-D", "team")
        c = cli(stack, "report", "--json", "-s", "openai", "-s", "anthropic", "-D", "team")
        c_fb = cli(
            stack, "report", "--json", "-s", "openai", "-s", "anthropic",
            "-D", "team", "-f", "project", "-f", "workspace",
        )
        feature = cli(stack, "report", "--json", "-s", "opencost", "-s", "litellm", "-D", "feature")
        results["A_gateway_and_cluster"] = brief(a)
        results["B_all_sources"] = brief(b)
        results["C_provider_billing"] = brief(c)
        results["C_provider_billing_with_fallback"] = brief(c_fb)
        results["A_by_feature"] = brief(feature)

        # Reconciliation, through the same fetch path the CLI uses.
        env = stack.env()
        fetched = {
            name: REGISTRY[name](
                base_url=env[f"UNALLOC_{name.upper()}_URL"],
                token=env.get({"litellm": "UNALLOC_LITELLM_KEY", "openai": "OPENAI_ADMIN_KEY",
                               "anthropic": "ANTHROPIC_ADMIN_KEY"}.get(name, ""), None),
            ).fetch(common.WINDOW_START, now)
            for name in ("litellm", "openai", "anthropic")
        }
        invoices = [
            Invoice(source=p, amount_usd=Decimal(f"{data['invoice'][p]:.2f}"),
                    start=common.WINDOW_START, end=now)
            for p in ("openai", "anthropic")
        ]
        provider_rec = reconcile(fetched["openai"] + fetched["anthropic"], invoices)
        gateway_invoice = Invoice(
            source="litellm",
            amount_usd=Decimal(f"{data['invoice']['openai'] + data['invoice']['anthropic']:.2f}"),
            start=common.WINDOW_START,
            end=now,
        )
        gateway_rec = reconcile(fetched["litellm"], [gateway_invoice])
        results["R_reconciliation"] = {
            "providers": [
                {"source": r.source, "ledger_usd": r.ledger_usd, "invoice_usd": r.invoice_usd,
                 "delta_pct": r.delta_pct, "status": r.status()}
                for r in provider_rec.per_source
            ],
            "gateway_vs_provider_invoices": [
                {"source": r.source, "ledger_usd": r.ledger_usd, "invoice_usd": r.invoice_usd,
                 "delta_usd": r.delta_usd, "delta_pct": r.delta_pct, "status": r.status()}
                for r in gateway_rec.per_source
            ],
            "bypass_usd": data["bypass_usd"],
        }

        # What the pre-fix adapters would have reported: first page only.
        first_page = {}
        for name, path, params in (
            ("openai", "/organization/costs", {"limit": 180}),
            ("anthropic", "/organizations/cost_report", {"limit": 180}),
        ):
            source = REGISTRY[name](base_url=env[f"UNALLOC_{name.upper()}_URL"],
                                    token=env[f"{name.upper()}_ADMIN_KEY"])
            rows = source.parse(source._get(path, params=params))
            first_page[name] = {
                "first_page_usd": sum((r.amount_usd for r in rows), Decimal("0")),
                "all_pages_usd": sum((r.amount_usd for r in fetched[name]), Decimal("0")),
            }
        results["P_pagination"] = first_page

        mocks = (stack.opencost, stack.litellm, stack.openai, stack.anthropic)
        requests_per_server = {s.name: len(s.hits) for s in mocks}
        backend = stack.litellm_backend

    gateway_spend = Decimal(a["by_source"]["litellm"])
    # Adding provider billing on top of the gateway adds two things: the
    # bypass traffic (real, previously invisible) and the gateway's own traffic
    # a second time (double counting). Only the latter is an error.
    bypass = Decimal(str(sum(data["bypass_usd"].values())))
    double_counted = Decimal(b["total_usd"]) - Decimal(a["total_usd"]) - bypass
    metrics = {
        "study": STUDY,
        "quick": args.quick,
        "litellm_backend": backend,
        "assumptions": {
            "prices_per_mtok": workload.PRICES,
            "bypass_daily_usd": workload.BYPASS_DAILY_USD,
            "streams": [s.__dict__ for s in workload.STREAMS],
            "page_size": servers.PAGE_SIZE,
        },
        "invoice": data["invoice"],
        "scenarios": results,
        "findings": {
            "double_counted_usd": double_counted,
            "double_counted_vs_gateway_spend": float(double_counted / gateway_spend),
            "gateway_coverage_of_provider_invoices": float(
                gateway_spend / Decimal(str(sum(data["invoice"].values())))
            ),
        },
        "http_requests": requests_per_server,
        "runtime_s": time.perf_counter() - started,
    }
    common.dump_json(args.out / "metrics.json", metrics)

    print(f"hybrid_e2e: litellm backend={backend}, http requests={requests_per_server}")
    for key in ("A_gateway_and_cluster", "B_all_sources", "C_provider_billing",
                "C_provider_billing_with_fallback"):
        r = results[key]
        print(f"  {key:<34} total {common.usd(r['total_usd']):>12}  "
              f"unallocated {r['unallocated_pct']}%")
    print(f"  double counted when every source is on: {common.usd(double_counted)}")
    for item in results["R_reconciliation"]["providers"]:
        print(f"  reconcile {item['source']:<10} {item['status']} ({item['delta_pct']}%)")
    gw = results["R_reconciliation"]["gateway_vs_provider_invoices"][0]
    print(f"  gateway ledger vs provider invoices: {gw['status']} {gw['delta_pct']}%")
    for name, p in first_page.items():
        print(f"  first page only, {name}: {common.usd(p['first_page_usd'])} "
              f"of {common.usd(p['all_pages_usd'])}")
    print(f"wrote {args.out / 'metrics.json'} in {metrics['runtime_s']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
