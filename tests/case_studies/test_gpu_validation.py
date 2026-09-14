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
