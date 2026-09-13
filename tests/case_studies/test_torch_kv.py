"""Smoke test for case_studies/torch_kv: the model decodes correctly and the pipeline ties out."""

from __future__ import annotations

import sys
import time
from decimal import Decimal
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

# `case_studies` lives at the repo root, outside the installed package, and
# plain `pytest` does not put the root on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_cached_decoding_matches_uncached() -> None:
    from case_studies.torch_kv.model import ModelConfig, TinyDecoder, equivalence_check

    cfg = ModelConfig(d_model=64, n_layers=2, n_heads=2, d_ff=128, max_seq_len=128)
    result = equivalence_check(TinyDecoder(cfg, seed=1), prompt_len=20, prefix_len=8, n_new=12)
    assert result["cached_tokens_equal"]
    assert result["prefix_reuse_tokens_equal"]
    assert result["cached_max_abs_logit_diff"] <= 1e-4
    assert result["prefix_reuse_max_abs_logit_diff"] <= 1e-4
    assert result["passed"]


def test_quick_pipeline(tmp_path: Path) -> None:
    from case_studies.torch_kv.study import run

    started = time.perf_counter()
    metrics = run(tmp_path, quick=True, seed=7, threads=2)
    assert time.perf_counter() - started < 30

    assert metrics["correctness"]["passed"]
    assert (tmp_path / "metrics.json").exists()
    assert metrics["unalloc"]["baseline"]["unallocated_pct"] == Decimal("100.0")

    att = metrics["attribution"]
    for method, shares in att["shares"].items():
        assert sum(shares.values()) == pytest.approx(1.0, abs=1e-4), method
        assert sum(att["usd"][method].values()) == att["pool"]["usd"]
        summary = metrics["unalloc"][method]
        assert summary["total_matches_pool"]
        # Only the unlabeled sandbox key should land in <unallocated>.
        assert summary["unallocated_usd"] == att["usd"][method]["sandbox"]

    series = metrics["e1_scaling"]["series"]
    assert series["kv_bytes_analytic"] == series["kv_bytes_measured"]
