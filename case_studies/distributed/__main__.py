"""CLI: python -m case_studies.distributed [--quick] [--seed N] [--out DIR]."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from case_studies.common import results_dir, usd
from case_studies.distributed.study import run


def _mib(n: int) -> str:
    return f"{n / 2**20:.2f} MiB"


def report(metrics: dict[str, Any], out: Path) -> None:
    main = metrics["bench"]["main_case"]
    print(f"distributed study ({'quick' if metrics['quick'] else 'full'}), torch "
          f"{metrics['environment']['torch']}, results in {out}")
    print("\ncorrectness vs single-process reference")
    for cid, c in metrics["correctness"].items():
        print(f"  {cid:<4} tokens_match={c['tokens_match']}  max|dlogit|="
              f"{c['max_abs_logit_diff']:.2e}  (min top-2 margin "
              f"{c['reference_min_top2_margin']:.3f})")
    print(f"\ntiming, case {main}: tokens/s and per-rank comm fraction; memory per rank")
    for cfg in metrics["configs"]:
        t = cfg["timing"][main]
        fracs = " ".join(f"{e['comm_fraction']:.0%}" for e in t["per_rank"])
        mem = " ".join(_mib(m["param_bytes"] + m["kv_cache_bytes"]) for m in cfg["memory_per_rank"])
        print(f"  {cfg['id']:<4} {t['tokens_per_s']:>8.1f} tok/s  comm [{fracs}]  mem [{mem}]")

    attr = metrics["attribution"]
    print("\nattribution on `team` (LeaderWorkerSet month)")
    for sid, sc in attr["scenarios"].items():
        s = sc["summary"]
        acc = s["accuracy"]
        print(f"  {sid:<8} unallocated {s['unallocated_pct']}% ({usd(s['unallocated_usd'])}), "
              f"misattributed {usd(acc['misattributed_usd'])}  fallback={s['fallback']}")
    comm = attr["communication"]
    print(f"  communication GPU time: {usd(comm['communication_usd_total'])} "
          f"({usd(comm['communication_usd_on_workers'])} on worker pods); per-token showback "
          f"shifts {usd(comm['cross_tenant_shift_usd'])} between tenants")
    canon = attr["canonicalization"]
    for canon_key, raws in canon["slash_collisions"].items():
        print(f"  collision: {raws} -> '{canon_key}'")
    order = canon["order_dependence"]
    print(f"  on {order['pod']}: name={order['name_when_keys_sorted_as_go_emits']!r} "
          f"(sorted keys) vs {order['name_when_key_order_reversed']!r} (reversed)")
    print(f"\nruntime {metrics['runtime_s']['total']} s")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m case_studies.distributed")
    parser.add_argument("--quick", action="store_true", help="one bench case, fewer repeats")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="results directory")
    args = parser.parse_args(argv)
    out = args.out if args.out is not None else results_dir("distributed")
    report(run(out, quick=args.quick, seed=args.seed), out)


if __name__ == "__main__":
    main()
