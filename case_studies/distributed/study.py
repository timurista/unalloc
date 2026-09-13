"""Orchestration: run every parallel config, aggregate, attribute, write metrics.json.

Raw per-repeat timings stay in the workers; only medians and flat, plot-ready
series reach metrics.json so the file stays small and the figures script never
has to re-derive anything.
"""

from __future__ import annotations

import os
import platform
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from case_studies.common import dump_json
from case_studies.distributed.attribution import run_attribution
from case_studies.distributed.model import ModelConfig, Stage, init_weights
from case_studies.distributed.parallel import (
    THREAD_BUDGET,
    Case,
    RunPlan,
    launch,
    median,
    threads_per_rank,
)

CONFIGS: tuple[tuple[str, int, int | None], ...] = (
    ("tp", 1, None),
    ("tp", 1, 1),
    ("tp", 1, 2),
    ("tp", 2, None),
    ("tp", 4, None),
    ("pp", 2, None),
)
"""(mode, world_size, threads override). `tp1_t1` / `tp1_t2` are single-process
baselines at the per-rank thread counts of tp4 / tp2 and pp2. A 4-thread process is
not faster than a 1-thread one on a model this small, so without matched baselines a
distributed config can look faster for reasons unrelated to parallelism."""

LOGIT_TOLERANCE = 1e-4


@dataclass(frozen=True, slots=True)
class Budget:
    warmup: int
    repeats: int
    correctness: Case
    main: Case
    cases: tuple[Case, ...]


def budget(quick: bool) -> Budget:
    if quick:
        main = Case(batch=4, prompt_len=64, new_tokens=16)
        return Budget(1, 2, Case(2, 16, 16), main, (main,))
    main = Case(batch=4, prompt_len=64, new_tokens=32)
    sweep = (Case(4, 16, 32), Case(4, 256, 32), Case(1, 64, 32), Case(16, 64, 32))
    return Budget(2, 5, Case(2, 16, 32), main, (main, *sweep))


def _r(x: float, digits: int = 6) -> float:
    return float(f"{x:.{digits}g}") if x == x else x


def aggregate(
    cid: str, plan: RunPlan, ranks: list[dict[str, Any]], main: Case
) -> dict[str, Any]:
    """Collapse per-rank, per-repeat results into medians for one config."""
    corr = [{"rank": r["rank"], **r["correctness"]} for r in ranks]
    diffs = [c["max_abs_logit_diff"] for c in corr if c["max_abs_logit_diff"] is not None]
    max_diff = max(diffs)
    correctness = {
        "case": vars_case(plan.correctness),
        "tokens_match": all(c["tokens_match"] for c in corr if c["tokens_match"] is not None),
        "max_abs_logit_diff": max_diff,
        "tolerance": LOGIT_TOLERANCE,
        "within_tolerance": max_diff <= LOGIT_TOLERANCE,
        "reference_min_top2_margin": _r(ranks[0]["reference_top2_margin"]),
        "ranks": corr,
    }

    timing: dict[str, Any] = {}
    for case in plan.cases:
        per_rank = []
        reps_by_rank = [r["timings"][case.key]["repeats"] for r in ranks]
        for r, reps in zip(ranks, reps_by_rank, strict=True):
            entry: dict[str, Any] = {
                "rank": r["rank"],
                "total_s": _r(median([x["total_s"] for x in reps])),
                "compute_s": _r(median([x["total_s"] - x["comm_s"] for x in reps])),
                "comm_s": _r(median([x["comm_s"] for x in reps])),
                "comm_fraction": _r(median([x["comm_s"] / x["total_s"] for x in reps]), 4),
            }
            if case == main:
                entry["ops"] = {
                    op: {
                        "calls": stats["calls"],
                        "bytes": stats["bytes"],
                        "seconds": _r(median([x["ops"][op]["seconds"] for x in reps])),
                    }
                    for op, stats in reps[0]["ops"].items()
                }
            per_rank.append(entry)
        walls = [max(reps[i]["total_s"] for reps in reps_by_rank) for i in range(plan.repeats)]
        wall = median(walls)
        timing[case.key] = {
            **vars_case(case),
            "wall_s": _r(wall),
            "tokens_per_s": _r(case.batch * case.new_tokens / wall, 5),
            "first_token_s": _r(median([x["first_token_s"] for x in reps_by_rank[0]])),
            "comm_fraction_mean": _r(
                sum(e["comm_fraction"] for e in per_rank) / len(per_rank), 4
            ),
            "per_rank": per_rank,
        }

    memory = [
        {
            "rank": r["rank"],
            "param_bytes": r["param_bytes"],
            "kv_cache_bytes": r["timings"][main.key]["kv_cache_bytes"],
        }
        for r in ranks
    ]
    return {
        "id": cid,
        "mode": plan.mode,
        "world_size": plan.world_size,
        "threads_per_rank": plan.threads,
        "correctness": correctness,
        "memory_per_rank": memory,
        "timing": timing,
    }


def vars_case(case: Case) -> dict[str, int]:
    return {"batch": case.batch, "prompt_len": case.prompt_len, "new_tokens": case.new_tokens}


def reference_memory(cfg: ModelConfig, seed: int, case: Case) -> dict[str, int]:
    kv = 2 * cfg.n_layers * case.batch * cfg.n_heads * (case.prompt_len + case.new_tokens)
    return {
        "param_bytes": Stage(init_weights(cfg, seed), cfg).param_bytes,
        "kv_cache_bytes": kv * cfg.head_dim * 4,  # float32
    }


def run(
    out: Path,
    *,
    quick: bool = False,
    seed: int = 0,
    cfg: ModelConfig | None = None,
    configs: Sequence[tuple[str, int, int | None]] = CONFIGS,
) -> dict[str, Any]:
    started = time.perf_counter()
    cfg = cfg or ModelConfig()
    bud = budget(quick)
    out.mkdir(parents=True, exist_ok=True)

    results, config_runtime = [], {}
    for mode, world, threads in configs:
        cid = f"{mode}{world}" if threads is None else f"{mode}{world}_t{threads}"
        plan = RunPlan(
            mode=mode,
            world_size=world,
            cfg=cfg,
            seed=seed,
            correctness=bud.correctness,
            cases=bud.cases,
            warmup=bud.warmup,
            repeats=bud.repeats,
            threads=threads if threads is not None else threads_per_rank(world),
        )
        t0 = time.perf_counter()
        results.append(aggregate(cid, plan, launch(plan), bud.main))
        config_runtime[cid] = round(time.perf_counter() - t0, 2)

    by_id = {r["id"]: r for r in results}
    fractions = {
        mode: [e["comm_fraction"] for e in by_id[cid]["timing"][bud.main.key]["per_rank"]]
        for mode, cid in (("tp", "tp4"), ("pp", "pp2"))
        if cid in by_id
    }
    attribution = run_attribution(out, fractions)

    throughput = [
        {"config": r["id"], "mode": r["mode"], "world_size": r["world_size"],
         **{k: t[k] for k in ("batch", "prompt_len", "new_tokens", "tokens_per_s", "wall_s",
                              "first_token_s", "comm_fraction_mean")}}
        for r in results
        for t in r["timing"].values()
    ]
    breakdown = [
        {"config": r["id"], "case": key, "rank": e["rank"],
         **{k: e[k] for k in ("compute_s", "comm_s", "total_s", "comm_fraction")}}
        for r in results
        for key, t in r["timing"].items()
        for e in t["per_rank"]
    ]
    memory = [
        {"config": r["id"], **m, "total_bytes": m["param_bytes"] + m["kv_cache_bytes"]}
        for r in results
        for m in r["memory_per_rank"]
    ]

    metrics: dict[str, Any] = {
        "study": "distributed",
        "quick": quick,
        "seed": seed,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "device": "cpu",
            "backend": "gloo (TCP over 127.0.0.1)",
            "thread_budget": THREAD_BUDGET,
            "threads_per_rank": {r["id"]: r["threads_per_rank"] for r in results},
            "interop_threads_per_rank": 1,
        },
        "model": cfg.as_dict(),
        "bench": {
            "warmup": bud.warmup,
            "repeats": bud.repeats,
            "main_case": bud.main.key,
            "cases": [c.key for c in bud.cases],
        },
        "correctness": {r["id"]: r["correctness"] for r in results},
        "configs": results,
        "series": {"throughput": throughput, "timing_breakdown": breakdown, "memory": memory},
        "reference_memory": reference_memory(cfg, seed, bud.main),
        "attribution": attribution,
        "notes": NOTES,
        "runtime_s": {
            "total": round(time.perf_counter() - started, 2),
            "per_config": config_runtime,
        },
    }
    dump_json(out / "metrics.json", metrics)
    return metrics


NOTES = [
    "CPU + gloo over loopback TCP: distribution is expected to be slower than a single "
    "process here, and that is honest, not a bug. The study does not claim or seek a speedup; "
    "it measures the communication fraction and the per-rank cost structure. Compare each "
    "distributed config with the single-process baseline at the same threads per rank "
    "(tp2/pp2 vs tp1_t2, tp4 vs tp1_t1), not with tp1 (4 threads): the tiny model scales "
    "poorly, even negatively, across intra-op threads, so splitting a thread budget across "
    "processes can look faster for reasons unrelated to model parallelism. PP2 in "
    "particular runs one 2-thread stage at a time and its send/recv payloads are tiny, so "
    "it tracks tp1_t2 minus a small overhead; its comm share is almost all bubble.",
    "comm_s is wall time blocked inside all_reduce/send/recv, including waiting for a peer. "
    "For TP ranks do symmetric work so waiting is small; for PP the recv wait is the "
    "pipeline bubble (no micro-batching, so each stage idles while the other computes).",
    "compute_s = total_s - comm_s per repeat; medians are taken per field over repeats, so "
    "compute_s + comm_s can differ from total_s by rounding of independent medians.",
    "wall_s = median over repeats of the slowest rank's generate time; tokens_per_s = "
    "batch * new_tokens / wall_s and includes prefill.",
    "Thread budget: THREAD_BUDGET total intra-op threads split evenly across ranks "
    "(4 -> 4/2/1 per rank for world 1/2/4), interop threads 1, so ranks never oversubscribe "
    "the 4 performance cores this was calibrated on.",
    "TP shards QKV/up-proj column-wise and out/down-proj row-wise with an all_reduce after "
    "each row-parallel layer (2 per layer); embeddings, LayerNorms and LM head are replicated, "
    "so param_bytes per TP rank does not shrink by exactly W.",
    "Communication fractions measured on CPU/gloo are far higher than on NVLink GPUs; the "
    "attribution section uses them as measured (an upper bound) and adds a sensitivity series "
    "at GPU-plausible fractions.",
]
