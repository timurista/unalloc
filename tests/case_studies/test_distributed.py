"""Distributed case study: TP numerics and the LeaderWorkerSet attribution leak."""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

# case_studies is a repo-root package, not part of the installed wheel; plain
# `pytest` (unlike `python -m pytest`) does not put the root on sys.path.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from case_studies.distributed.attribution import run_attribution  # noqa: E402
from case_studies.distributed.model import TINY  # noqa: E402
from case_studies.distributed.parallel import Case, RunPlan, launch  # noqa: E402


def test_tensor_parallel_world2_matches_reference() -> None:
    plan = RunPlan(
        mode="tp",
        world_size=2,
        cfg=TINY,
        seed=0,
        correctness=Case(batch=2, prompt_len=8, new_tokens=8),
        cases=(Case(batch=1, prompt_len=8, new_tokens=4),),
        warmup=0,
        repeats=1,
        threads=1,
    )
    ranks = launch(plan)
    assert len(ranks) == 2
    for rank in ranks:
        assert rank["correctness"]["tokens_match"]
        assert rank["correctness"]["max_abs_logit_diff"] < 1e-4
        timing = rank["timings"]["b1_p8_n4"]["repeats"][0]
        # 2 all_reduces per layer, per forward (1 prefill + 3 decode steps)
        assert timing["ops"]["all_reduce"]["calls"] == 2 * TINY.n_layers * 4
    # sharded parameters: both ranks hold the same amount
    assert ranks[0]["param_bytes"] == ranks[1]["param_bytes"]


def test_attribution_scenarios(tmp_path: Path) -> None:
    fractions = {"tp": [0.3, 0.4, 0.4, 0.4], "pp": [0.5, 0.5]}
    result = run_attribution(tmp_path, fractions)
    sc = result["scenarios"]
    for sid in ("s1", "s2", "s3"):
        assert (tmp_path / sid / "opencost_allocation.json").exists()

    s1 = sc["s1"]["summary"]
    s3 = sc["s3"]["summary"]
    # search: 2 replicas x 3 workers; agents: 3 replicas x 1 worker -> 9 of 14 pods unowned
    assert s1["accuracy"]["misattributed_usd"] == Decimal("0")
    assert s1["unallocated_pct"] > Decimal("50")
    # S3 leaves only the cluster __idle__ allocation unallocated
    assert s3["accuracy"]["unallocated_owned_usd"] == Decimal("0")
    assert s3["unallocated_usd"] == s3["accuracy"]["unallocated_idle_usd"]
    assert s3["unallocated_pct"] < Decimal("10")

    # S2: the `name` fallback "fixes" the gap by landing worker spend in the chart name
    s2 = sc["s2"]["summary"]
    assert s2["unallocated_pct"] == s3["unallocated_pct"]
    assert "vllm" in s2["accuracy"]["non_owner_buckets"]
    canon = result["canonicalization"]
    assert "name" in canon["slash_collisions"]
    assert canon["order_dependence"]["name_when_keys_sorted_as_go_emits"] == "vllm"

    comm = result["communication"]
    assert comm["communication_usd_total"] > comm["communication_usd_on_workers"] > 0
