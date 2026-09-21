"""Every number in the GPU results table must still match the measured run.

The published table once printed two time-share entries without redistributing
overhead, which no test could see: the figure code redistributes, the table was
typed by hand. This reads the numbers back out of the Typst source and compares
them against `metrics.json`, so a stale table fails the build rather than a
reviewer's arithmetic.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper" / "unalloc.typ"
METRICS = ROOT / "case_studies" / "results" / "gpu_validation" / "metrics.json"

pytestmark = pytest.mark.skipif(not METRICS.exists(), reason="GPU metrics not present")

# [2], [446 (3.7)], [773], [67%], [28 / 44], [6.3], [97%], [469], [16.5%], [4.8%],
ROW = re.compile(
    r"^\s*\[(?P<load>\d+)\], \[(?P<reqs>[\d,]+) \((?P<done>[\d.]+)\)\], \[(?P<out>[\d,]+)\], "
    r"\[(?P<cache>\d+)%\], \[(?P<ttft50>[\d.]+) / (?P<ttft95>[\d.]+)\], \[(?P<tpot>[\d.]+)\], "
    r"\[(?P<util>\d+)%\], \[(?P<power>\d+)\], \[(?P<tok>[\d.]+)%\], \[(?P<time>[\d.]+)%\],\s*$",
    re.MULTILINE,
)


def _num(text: str) -> float:
    return float(text.replace(",", ""))


def _close(printed: float, actual: float, places: int) -> bool:
    """Printed to `places` decimals is honest if it is within half a unit there."""
    return abs(printed - actual) <= 0.5 * 10**-places + 1e-9


@pytest.fixture(scope="module")
def rows():
    runs = {r["rate_rps"]: r for r in json.loads(METRICS.read_text())["runs"]}
    printed = [m.groupdict() for m in ROW.finditer(SOURCE.read_text())]
    assert len(printed) == len(runs), f"table has {len(printed)} rows, data has {len(runs)}"
    return [(p, runs[float(p["load"])]) for p in printed]


def test_table_reports_completed_requests_and_throughput(rows):
    for p, run in rows:
        assert _num(p["reqs"]) == run["requests"]
        assert _close(_num(p["done"]), run["requests"] / run["wall_s"], 1), p["load"]
        assert _close(_num(p["out"]), run["output_tokens_per_s"], 0), p["load"]
        assert _close(_num(p["cache"]), run["cache_hit_rate_usage"] * 100, 0), p["load"]


def test_table_latency_and_device_columns(rows):
    for p, run in rows:
        lat = run["latency"]["all"]
        assert _close(_num(p["ttft50"]), lat["ttft_s"]["p50"] * 1000, 0), p["load"]
        assert _close(_num(p["ttft95"]), lat["ttft_s"]["p95"] * 1000, 0), p["load"]
        assert _close(_num(p["tpot"]), lat["tpot_s"]["p50"] * 1000, 1), p["load"]
        assert _close(_num(p["util"]), run["gpu"]["gpu_util_pct_mean"], 0), p["load"]
        assert _close(_num(p["power"]), run["gpu"]["power_w_mean"], 0), p["load"]


def test_meter_shares_are_printed_with_overhead_redistributed(rows):
    """The caption promises redistribution; both meter columns must honour it."""
    for p, run in rows:
        for column, meter in (("tok", "tokens"), ("time", "time_share")):
            split = run["meters"][meter]
            redistributed = split["search"] / (1 - split["__overhead__"]) * 100
            assert _close(_num(p[column]), redistributed, 1), (
                f"load {p['load']} {meter}: paper prints {p[column]}%, "
                f"redistributed is {redistributed:.3f}% "
                f"(raw {split['search'] * 100:.3f}%)"
            )


def test_prose_ranges_match_the_table(rows):
    text = " ".join(SOURCE.read_text().split())
    tok = [_num(p["tok"]) for p, _ in rows]
    share = [_num(p["time"]) for p, _ in rows]
    assert f"search {min(tok):.1f}–{max(tok):.1f}% of the bill" in text
    assert f"time-share meter assigns {min(share):.1f}–{max(share):.1f}%" in text

    divergence = [run["divergence_search_tokens_vs_time_pts"] for _, run in rows]
    lo, hi = min(divergence), max(divergence)
    assert f"a disagreement of {lo:.1f}–{hi:.1f} points" in text


def test_totals_and_error_free_claim(rows):
    flowed = " ".join(SOURCE.read_text().split())
    total = sum(run["requests"] for _, run in rows)
    assert f"All {total:,} requests completed without error" in flowed
    assert all(run["errors"] == 0 for _, run in rows)


def test_queue_claim_is_scoped_to_the_sampling_interval(rows):
    """max_waiting is a half-second Prometheus sample, not a continuous guarantee."""
    text = " ".join(SOURCE.read_text().split())
    assert all(run["max_waiting"] == 0 for _, run in rows)
    assert "no queued request was observed in any half-second telemetry sample" in text
    assert "none waited in the queue" not in text


def test_configured_rate_is_distinguished_from_completed_traffic(rows):
    text = " ".join(SOURCE.read_text().split())
    assert "session-initial" in text
    done = [run["requests"] / run["wall_s"] for _, run in rows]
    assert f"{min(done):.1f} to {max(done):.1f} completed requests per second" in text
