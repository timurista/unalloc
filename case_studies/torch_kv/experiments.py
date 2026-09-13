"""The two measurements: E1 (KV cache scaling) and E2 (serving a multi-tenant trace).

Both time real forward passes of the model on this machine. Timings are taken
with `time.perf_counter` under `torch.inference_mode`, after warm-up, with a
fixed thread count, and summarised by the median so a single scheduler hiccup
does not move a share.
"""

from __future__ import annotations

import statistics
import time
from typing import Any

import torch

from case_studies.torch_kv.model import KVCache, TinyDecoder
from case_studies.torch_kv.trace import Request, system_prefix_ids


def _median_time(fn: Any, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


# --- E1: per-step latency with and without the cache ------------------------------


@torch.inference_mode()
def run_scaling(
    model: TinyDecoder,
    contexts: list[int],
    *,
    warmup: int,
    repeats_cached: int,
    repeats_uncached: int,
    seed: int,
) -> dict[str, Any]:
    """Cost of producing one more token at context length n.

    Cached: one single-token forward that attends to n stored keys/values.
    Uncached: a full forward over all n+1 tokens, which is what decoding
    without a cache does at every step.
    """
    cfg = model.cfg
    gen = torch.Generator().manual_seed(seed)
    series: dict[str, list[Any]] = {
        "context": [],
        "cached_step_s": [],
        "uncached_step_s": [],
        "speedup": [],
        "prefill_s": [],
        "kv_bytes_analytic": [],
        "kv_bytes_measured": [],
    }
    token = torch.tensor([[1]])
    # Untimed pass at the longest context first: without it the first (shortest)
    # context absorbs allocator and kernel warm-up and plots as an outlier.
    longest = max(contexts)
    warm_cache = KVCache(cfg, capacity=longest + 1)
    for _ in range(warmup):
        warm_cache.reset()
        model(torch.randint(0, cfg.vocab_size, (1, longest), generator=gen), warm_cache)
        model(token, warm_cache)
    for n in contexts:
        prompt = torch.randint(0, cfg.vocab_size, (1, n), generator=gen)
        cache = KVCache(cfg, capacity=n + 1)
        full = torch.cat([prompt, token], dim=1)

        def prefill(cache: KVCache = cache, prompt: torch.Tensor = prompt) -> None:
            cache.reset()
            model(prompt, cache)

        def cached_step(cache: KVCache = cache, n: int = n) -> None:
            cache.reset(n)  # rewind so every repeat decodes at exactly context n
            model(token, cache)

        def uncached_step(full: torch.Tensor = full) -> None:
            model(full, None)

        for _ in range(warmup):
            prefill()
            cached_step()
            uncached_step()
        prefill_s = _median_time(prefill, max(3, repeats_uncached))
        prefill()
        cached_s = _median_time(cached_step, repeats_cached)
        uncached_s = _median_time(uncached_step, repeats_uncached)

        # The cache holds n tokens when the step runs; measure a cache sized
        # to exactly that so the analytic formula and tensor.nbytes line up.
        series["context"].append(n)
        series["cached_step_s"].append(cached_s)
        series["uncached_step_s"].append(uncached_s)
        series["speedup"].append(uncached_s / cached_s)
        series["prefill_s"].append(prefill_s)
        series["kv_bytes_analytic"].append(cfg.kv_bytes(n))
        series["kv_bytes_measured"].append(KVCache(cfg, capacity=n).nbytes)
    return {
        "description": "median seconds per decode step at context length n",
        "warmup": warmup,
        "repeats_cached": repeats_cached,
        "repeats_uncached": repeats_uncached,
        "kv_bytes_formula": "2 * n_layers * n_heads * head_dim * seq_len * bytes_per_element",
        "series": series,
    }


# --- E2: serve the trace -------------------------------------------------------------


def analytic_flops(model: TinyDecoder, start: int, n_tokens: int) -> int:
    """Forward FLOPs for `n_tokens` new tokens after `start` cached ones.

    Kaplan et al.'s 2N + 2 * n_layers * n_ctx * d_model per token, plus one LM
    head matmul per forward. This is the hardware-independent cost a token
    count *should* track, kept as a reference for the measured methods.
    """
    cfg = model.cfg
    body = 2 * model.n_params(non_embedding=True)
    ctx_sum = sum(range(start + 1, start + n_tokens + 1))
    head = 2 * cfg.d_model * cfg.vocab_size
    return body * n_tokens + 2 * cfg.n_layers * cfg.d_model * ctx_sum + head


def _prompt_ids(req: Request, prefix: list[int], vocab: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(req.prompt_seed)
    head = prefix if req.shared_prefix else []
    body = torch.randint(0, vocab, (req.prompt_tokens - len(head),), generator=gen).tolist()
    return torch.tensor([head + body])


@torch.inference_mode()
def serve_request(
    model: TinyDecoder, req: Request, prompt: torch.Tensor, prefix: KVCache | None
) -> dict[str, Any]:
    """Serve one request and integrate the KV memory it holds over its wall time.

    Occupied bytes are integrated piecewise: during prefill the request is
    charged its full prompt occupancy (the cache is being filled), and during
    each decode step the occupancy after that step's write. Reserved bytes are
    the whole preallocated cache for the whole wall time, which is what a
    static allocator would hold.
    """
    cfg = model.cfg
    cache = KVCache(cfg, capacity=req.prompt_tokens + req.completion_tokens)
    use_prefix = prefix is not None and req.shared_prefix

    t0 = time.perf_counter()
    start = 0
    if use_prefix:
        assert prefix is not None
        cache.load_prefix(prefix)
        start = prefix.length
    logits = model(prompt[:, start:], cache)
    token = int(logits[0].argmax())
    t1 = time.perf_counter()
    prefill_s = t1 - t0
    kv_byte_seconds = cache.used_bytes * prefill_s

    decode_s = 0.0
    for _ in range(req.completion_tokens - 1):
        ts = time.perf_counter()
        logits = model(torch.tensor([[token]]), cache)
        token = int(logits[0].argmax())
        dt = time.perf_counter() - ts
        decode_s += dt
        kv_byte_seconds += cache.used_bytes * dt
    wall_s = time.perf_counter() - t0

    flops = analytic_flops(model, start, req.prompt_tokens - start)
    for ctx in range(req.prompt_tokens, req.prompt_tokens + req.completion_tokens - 1):
        flops += analytic_flops(model, ctx, 1)

    return {
        "request_id": req.request_id,
        "caller": req.caller,
        "team": req.team,
        "prompt_tokens": req.prompt_tokens,
        "completion_tokens": req.completion_tokens,
        "shared_prefix": use_prefix,
        "prefilled_tokens": req.prompt_tokens - start,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "wall_s": wall_s,
        "kv_peak_bytes": cache.used_bytes,
        "kv_reserved_bytes": cache.nbytes,
        "kv_byte_seconds": kv_byte_seconds,
        "kv_reserved_byte_seconds": cache.nbytes * wall_s,
        "analytic_flops": flops,
    }


@torch.inference_mode()
def serve_trace(model: TinyDecoder, trace: list[Request], *, repeats: int) -> dict[str, Any]:
    """Serve the trace sequentially `repeats` times; report per-request medians.

    Sequential (batch size 1) serving keeps each request's resource use
    separable by direct measurement, which is the ground truth the attribution
    methods are compared against. Batching would make that measurement itself
    an apportionment problem.
    """
    cfg = model.cfg
    prefix_ids = system_prefix_ids()
    prompts = {r.request_id: _prompt_ids(r, prefix_ids, cfg.vocab_size) for r in trace}

    # Warm up allocator and kernels on a representative long and short request.
    warm = sorted(trace, key=lambda r: r.prompt_tokens)
    for req in (warm[0], warm[-1]):
        serve_request(model, req, prompts[req.request_id], None)

    runs: list[list[dict[str, Any]]] = []
    prefix_prefill: list[float] = []
    serve_wall: list[float] = []
    prefix: KVCache | None = None
    for _ in range(repeats):
        # The shared prefix is recomputed each repeat so its one-time cost is
        # measured under the same conditions as the requests that reuse it.
        prefix = KVCache(cfg, capacity=len(prefix_ids))
        t0 = time.perf_counter()
        model(torch.tensor([prefix_ids]), prefix)
        prefix_prefill.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        runs.append([serve_request(model, r, prompts[r.request_id], prefix) for r in trace])
        serve_wall.append(time.perf_counter() - t0)

    measured = ("prefill_s", "decode_s", "wall_s", "kv_byte_seconds", "kv_reserved_byte_seconds")
    table: list[dict[str, Any]] = []
    for i, row in enumerate(runs[0]):
        merged = dict(row)
        for key in measured:
            merged[key] = statistics.median(run[i][key] for run in runs)
        table.append(merged)

    assert prefix is not None
    return {
        "repeats": repeats,
        "system_prefix_tokens": len(prefix_ids),
        "shared_prefix_prefill_s": statistics.median(prefix_prefill),
        "shared_prefix_kv_bytes": prefix.nbytes,
        "serve_wall_s": statistics.median(serve_wall),
        "requests": table,
        "runs": runs,
    }
