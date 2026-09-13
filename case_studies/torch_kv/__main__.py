"""CLI: python -m case_studies.torch_kv [--quick] [--seed N] [--out DIR]."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from case_studies import common
from case_studies.torch_kv.study import STUDY, run


def _print_summary(m: dict[str, Any], out: Path) -> None:
    corr = m["correctness"]
    e1 = m["e1_scaling"]["series"]
    att = m["attribution"]
    callers = att["callers"]
    print(f"torch_kv ({m['mode']}) finished in {m['runtime_s']:.1f}s -> {out / 'metrics.json'}")
    print(
        f"correctness: {'PASS' if corr['passed'] else 'FAIL'} "
        f"(cached max |dlogit| {corr['cached_max_abs_logit_diff']:.2e}, "
        f"prefix reuse {corr['prefix_reuse_max_abs_logit_diff']:.2e})"
    )
    print(
        f"E1: at context {e1['context'][-1]} a cached step takes "
        f"{e1['cached_step_s'][-1] * 1e3:.2f} ms vs {e1['uncached_step_s'][-1] * 1e3:.2f} ms "
        f"uncached ({e1['speedup'][-1]:.1f}x); KV {e1['kv_bytes_analytic'][-1] / 2**20:.1f} MiB"
    )
    print(f"\nshare of {common.usd(att['pool']['usd'])}/month GPU pool by method")
    header = f"{'method':<20}" + "".join(f"{c:>10}" for c in callers) + f"{'unalloc %':>11}"
    print(header)
    for method in att["methods"]:
        cells = "".join(f"{att['shares'][method][c]:>10.1%}" for c in callers)
        print(f"{method:<20}{cells}{att['unallocated_pct_by_method'][method]:>10}%")
    print(f"{'baseline (1 pod)':<20}{'':>40}{att['unallocated_pct_by_method']['baseline']:>10}%")
    head = att["divergence"]["tokens"]["headline"]
    print(
        f"\nheadline: per-token showback vs {head['method']} mis-prices '{head['caller']}' by "
        f"{head['max_abs_share_diff']:.1%} of the pool ({common.usd(head['usd'])}/month)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m case_studies.torch_kv", description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small run for smoke tests (~20 s)")
    parser.add_argument("--seed", type=int, default=7, help="seed for weights and trace")
    parser.add_argument("--threads", type=int, default=4, help="torch.set_num_threads value")
    parser.add_argument("--out", type=Path, default=None, help="results directory")
    args = parser.parse_args(argv)

    out = args.out or common.results_dir(STUDY)
    metrics = run(out, quick=args.quick, seed=args.seed, threads=args.threads)
    _print_summary(metrics, out)
    return 0 if metrics["correctness"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
