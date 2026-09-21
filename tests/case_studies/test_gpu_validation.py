"""The GPU validation pipeline's pure parts: metrics parsing, traces and meters.

No GPU and no server: these guard the code that turns a paid GPU hour into
numbers, so a regression is caught before the next rental rather than during it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from case_studies.gpu_validation.analyze import meters
from case_studies.gpu_validation.bench import parse_metrics
from case_studies.gpu_validation.workload import TraceBuilder

PROMETHEUS = """\
# HELP vllm:kv_cache_usage_perc KV-cache usage.
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.25
vllm:num_requests_running{engine="0",model_name="m"} 3.0
vllm:num_requests_waiting_by_reason{engine="0",reason="capacity"} 9.0
vllm:prefix_cache_hits_total{engine="0",model_name="m"} 10
vllm:prefix_cache_hits_total{engine="1",model_name="m"} 5
vllm:time_to_first_token_seconds_bucket{le="0.1"} 7
"""


def test_parse_metrics_sums_label_sets_and_ignores_lookalikes():
    values = parse_metrics(PROMETHEUS)
    assert values["vllm:kv_cache_usage_perc"] == 0.25
    assert values["vllm:num_requests_running"] == 3.0
    assert values["vllm:prefix_cache_hits"] == 15.0
    # `_by_reason` and histogram buckets must not leak into the base metrics.
    assert "vllm:num_requests_waiting" not in values


def test_traces_share_system_prompts_and_respect_context_limit():
    builder = TraceBuilder(seed=1)
    requests = builder.arrivals(rate=5.0, duration=60.0)
    search = [r for r in requests if r.tenant == "search"]
    assert search
    # Three system-prompt groups, byte-identical across requests, so prefix caching can hit.
    assert len({tuple(r.prompt[:512]) for r in search}) <= 3

    agent = next(r for r in requests if r.session and r.session.turns_left > 0)
    assert builder.follow_up(agent, finished_at=1.0, duration=1e9, max_len=10) is None
    nxt = builder.follow_up(agent, finished_at=1.0, duration=1e9, max_len=100_000)
    assert nxt is not None
    assert nxt.prompt[: len(agent.prompt)] == agent.prompt


def _request(rid, tenant, send, first, done, prompt, completion, cached):
    return {
        "rid": rid, "tenant": tenant, "send": send, "first": first, "done": done,
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                  "prompt_tokens_details": {"cached_tokens": cached}},
    }


def test_meters_split_the_whole_bill_and_disagree_by_workload_shape():
    run = {
        "requests": [
            # A RAG call: many prompt tokens, done quickly.
            _request(1, "search", 0.0, 0.1, 0.2, prompt=1000, completion=10, cached=0),
            # An agent turn: few prompt tokens, long decode.
            _request(2, "agents", 0.0, 0.05, 1.0, prompt=100, completion=200, cached=50),
        ],
        "metrics": [{"t": 0.5, "vllm:kv_cache_usage_perc": 0.5}],
    }
    result = meters(run)
    for split in result.values():
        assert abs(sum(split.values()) - 1.0) < 1e-9
    assert result["tokens"]["search"] > result["tokens"]["agents"]
    assert result["time_share"]["agents"] > result["time_share"]["search"]
    assert result["kv_memory"]["__overhead__"] == 0.5


def _stream(rid, tenant, send, dur, prompt, completion):
    """A request that occupies [send, send+dur) with known token counts."""
    return _request(rid, tenant, send, send + dur * 0.1, send + dur, prompt, completion, 0)


def test_window_stability_drops_warmup_and_reports_spread():
    from case_studies.gpu_validation.analyze import window_stability

    # 45 s of steady traffic: one search request and one agents request per second,
    # search always prompt-heavy. Window 0 is deliberately search-only, the pattern
    # the real runs show while the pipeline fills.
    reqs = []
    rid = 0
    for second in range(45):
        rid += 1
        reqs.append(_stream(rid, "search", float(second), 0.5, prompt=1000, completion=10))
        if second >= 10:
            rid += 1
            reqs.append(_stream(rid, "agents", second + 0.5, 0.4, prompt=100, completion=200))
    run = {"requests": reqs, "metrics": []}

    stability = window_stability(run, window_s=10.0)
    assert stability["skipped_warmup_windows"] == 1
    assert [w["window"] for w in stability["windows"]] == [1, 2, 3]
    # The warm-up window would have been 100% search under both meters; excluding it,
    # the remaining windows agree with each other.
    assert stability["divergence_pts"]["max"] - stability["divergence_pts"]["min"] < 1.0
    assert stability["search_tokens"]["median"] > stability["search_time_share"]["median"]


def test_window_stability_needs_enough_windows():
    from case_studies.gpu_validation.analyze import window_stability

    short = {"requests": [_stream(1, "search", 0.0, 1.0, 100, 10)], "metrics": []}
    assert window_stability(short, window_s=10.0) == {}


def test_bootstrap_ci_brackets_the_mean_and_needs_a_sample():
    from case_studies.gpu_validation.analyze import bootstrap_ci

    assert bootstrap_ci([1.0, 2.0]) is None
    lo, hi = bootstrap_ci([10.0, 11.0, 12.0, 13.0, 14.0])
    assert lo < 12.0 < hi
    assert lo >= 10.0 and hi <= 14.0


def test_spread_reports_quartiles_for_small_samples():
    from case_studies.gpu_validation.analyze import spread

    s = spread([4.0, 1.0, 3.0, 2.0])
    assert (s["n"], s["min"], s["max"], s["median"]) == (4, 1.0, 4.0, 2.5)
    assert s["p25"] < s["median"] < s["p75"]


def test_across_repeats_summarizes_independent_runs():
    from case_studies.gpu_validation.analyze import across_repeats

    entries = [
        {
            "seed": 7 + i,
            "divergence_search_tokens_vs_time_pts": d,
            "output_tokens_per_s": 700.0 + i,
            "latency": {"all": {"ttft_s": {"p50": 0.03}}},
            "meters": {"tokens": {"search": 0.17}},
        }
        for i, d in enumerate((11.0, 13.0, 12.0))
    ]
    summary = across_repeats(entries)
    assert summary["repeats"] == 3
    assert summary["seeds"] == [7, 8, 9]
    assert summary["divergence_pts"]["median"] == 12.0
    assert (summary["divergence_pts"]["min"], summary["divergence_pts"]["max"]) == (11.0, 13.0)
