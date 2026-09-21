"""Drive a live vLLM OpenAI-compatible server and record what a cost meter could see.

    python -m case_studies.gpu_validation.bench --base http://localhost:8000 \
        --model Qwen/Qwen2.5-7B-Instruct --rates 2 4 8 --duration 120 --out results/

Per request: send time, first-token time, completion time, and the server's
usage block (including cached prompt tokens). Twice a second: vLLM's
Prometheus gauges and counters (KV-cache usage, running/waiting requests,
prefix-cache hits). Analysis happens later, off the GPU, in `analyze.py`.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import time
from pathlib import Path
from typing import Any

import httpx

from case_studies.gpu_validation.workload import TraceBuilder, TraceRequest

GAUGES = ("vllm:kv_cache_usage_perc", "vllm:num_requests_running", "vllm:num_requests_waiting")
COUNTERS = (
    "vllm:prefix_cache_hits",
    "vllm:prefix_cache_queries",
    "vllm:prompt_tokens",
    "vllm:generation_tokens",
)


def parse_metrics(text: str) -> dict[str, float]:
    """Sum each metric across label sets; accept counters with or without `_total`."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        series, _, raw = line.rpartition(" ")
        name = series.split("{", 1)[0]
        for base in GAUGES + COUNTERS:
            if name in (base, f"{base}_total"):
                with contextlib.suppress(ValueError):
                    values[base] = values.get(base, 0.0) + float(raw)
    return values


async def sample_metrics(
    client: httpx.AsyncClient, base: str, t0: float, stop: asyncio.Event, out: list[dict[str, Any]]
) -> None:
    while not stop.is_set():
        try:
            response = await client.get(f"{base}/metrics", timeout=5)
            out.append({"t": time.perf_counter() - t0, **parse_metrics(response.text)})
        except httpx.HTTPError:
            pass
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), 0.5)


async def send(
    client: httpx.AsyncClient, base: str, model: str, req: TraceRequest, t0: float
) -> dict[str, Any]:
    body = {
        "model": model,
        "prompt": req.prompt,
        "max_tokens": req.output_len,
        # Exact output lengths, so tenant shapes match the simulator.
        "min_tokens": req.output_len,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    record: dict[str, Any] = {
        "rid": req.rid,
        "tenant": req.tenant,
        "team": req.team,
        "session": req.session.sid if req.session else None,
        "scheduled": req.at,
        "prompt_len": len(req.prompt),
        "output_len": req.output_len,
        "send": time.perf_counter() - t0,
    }
    first: float | None = None
    usage: dict[str, Any] | None = None
    chunks = 0
    error: str | None = None
    try:
        async with client.stream("POST", f"{base}/v1/completions", json=body) as response:
            if response.status_code != 200:
                error = f"HTTP {response.status_code}: {(await response.aread())[:300]!r}"
            else:
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    payload = json.loads(data)
                    if payload.get("choices"):
                        chunks += 1
                        if first is None:
                            first = time.perf_counter() - t0
                    if payload.get("usage"):
                        usage = payload["usage"]
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        error = repr(exc)
    record.update(first=first, done=time.perf_counter() - t0, chunks=chunks, usage=usage,
                  error=error)
    return record


async def run_rate(
    base: str, model: str, rate: float, duration: float, seed: int, max_len: int
) -> dict[str, Any]:
    builder = TraceBuilder(seed)
    limits = httpx.Limits(max_connections=4096, max_keepalive_connections=512)
    timeout = httpx.Timeout(None, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        started_unix = time.time()
        t0 = time.perf_counter()
        stop = asyncio.Event()
        samples: list[dict[str, Any]] = []
        sampler = asyncio.create_task(sample_metrics(client, base, t0, stop, samples))
        records: list[dict[str, Any]] = []
        pending: set[asyncio.Task[None]] = set()

        def spawn(req: TraceRequest) -> None:
            task = asyncio.create_task(launch(req))
            pending.add(task)
            task.add_done_callback(pending.discard)

        async def launch(req: TraceRequest) -> None:
            delay = req.at - (time.perf_counter() - t0)
            if delay > 0:
                await asyncio.sleep(delay)
            record = await send(client, base, model, req, t0)
            records.append(record)
            if record["error"] is None:
                nxt = builder.follow_up(req, record["done"], duration, max_len)
                if nxt is not None:
                    spawn(nxt)

        for req in builder.arrivals(rate, duration):
            spawn(req)
        while pending:
            await asyncio.wait(set(pending))
        stop.set()
        await sampler
        wall = time.perf_counter() - t0
    errors = [r for r in records if r["error"]]
    return {
        "rate_rps": rate,
        "duration_s": duration,
        "started_unix": started_unix,
        "wall_s": wall,
        "seed": seed,
        "requests": sorted(records, key=lambda r: r["rid"]),
        "metrics": samples,
        "errors": len(errors),
        "first_errors": [r["error"] for r in errors[:3]],
    }


async def main_async(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=10) as client:
        version = (await client.get(f"{args.base}/version")).json()
    meta = {"model": args.model, "base": args.base, "vllm": version, "max_len": args.max_len,
            "rates": args.rates, "duration_s": args.duration, "seed": args.seed,
            "repeats": args.repeats}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))

    print("warm-up ...", flush=True)
    warm = await run_rate(args.base, args.model, 1.0, 10.0, args.seed + 999, args.max_len)
    print(f"  warm-up: {len(warm['requests'])} requests, {warm['errors']} errors "
          f"{warm['first_errors']}", flush=True)
    if warm["errors"]:
        raise SystemExit("warm-up failed; not spending GPU time on a broken client")

    # Repeats are the outer loop so the load levels interleave in time. Running every
    # repeat of one rate back to back would confound load with anything that drifts
    # during the session — clock throttling, cache state, a noisy neighbour.
    for rep in range(args.repeats):
        for rate in args.rates:
            started = time.perf_counter()
            # A distinct seed per (rate, repeat) draws a fresh arrival process and
            # prompt mix; the same seed would replay identical traffic and understate
            # how much of the spread comes from the workload sample.
            seed = args.seed + 1000 * rep
            result = await run_rate(args.base, args.model, rate, args.duration, seed, args.max_len)
            result["repeat"] = rep
            name = f"rate_{rate:g}.json" if args.repeats == 1 else f"rate_{rate:g}_r{rep}.json"
            path = args.out / name
            path.write_text(json.dumps(result))
            done = [r for r in result["requests"] if r["usage"]]
            tokens = sum(r["usage"]["completion_tokens"] for r in done)
            print(f"rate {rate:g} rep {rep}: {len(result['requests'])} requests, "
                  f"{result['errors']} errors, {tokens / result['wall_s']:.0f} output tok/s, "
                  f"{time.perf_counter() - started:.0f}s -> {path}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--rates", type=float, nargs="+", default=[2.0, 4.0, 8.0])
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--repeats", type=int, default=1,
                        help="independent runs per load level, each with its own seed")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-len", type=int, default=8192)
    parser.add_argument("--out", type=Path, required=True)
    asyncio.run(main_async(parser.parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
