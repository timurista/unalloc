"""Turn raw GPU benchmark output into latency, metering and simulator-comparison results.

    python -m case_studies.gpu_validation.analyze --raw case_studies/results/gpu_validation/raw

Runs locally after the droplet is gone, so no GPU time is spent on analysis.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import re
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

from case_studies import common
from case_studies.kv_cache import attribution as kv_attr
from case_studies.kv_cache.sim import simulate

TICK_S = 0.05
LIST_PRICE = {"input": 1.0, "cached": 0.1, "output": 4.0}


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def summary(values: list[float]) -> dict[str, float | None]:
    return {
        "p50": statistics.median(values) if values else None,
        "p95": pct(values, 0.95),
        "p99": pct(values, 0.99),
        "mean": statistics.fmean(values) if values else None,
    }


def latency(requests: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in requests if r["usage"] and r["first"] is not None]

    def block(rows: list[dict[str, Any]]) -> dict[str, Any]:
        ttft = [r["first"] - r["send"] for r in rows]
        e2e = [r["done"] - r["send"] for r in rows]
        tpot = [
            (r["done"] - r["first"]) / (r["usage"]["completion_tokens"] - 1)
            for r in rows
            if r["usage"]["completion_tokens"] > 1
        ]
        return {
            "requests": len(rows),
            "ttft_s": summary(ttft),
            "tpot_s": summary(tpot),
            "e2e_s": summary(e2e),
        }

    by_tenant = defaultdict(list)
    for r in ok:
        by_tenant[r["tenant"]].append(r)
    return {
        "all": block(ok),
        "by_tenant": {t: block(rows) for t, rows in sorted(by_tenant.items())},
    }


def meters(run: dict[str, Any]) -> dict[str, Any]:
    """The same metering rules as the simulator, computed from real telemetry."""
    reqs = [r for r in run["requests"] if r["usage"] and r["first"] is not None]
    tenants = sorted({r["tenant"] for r in reqs})
    end = max(r["done"] for r in reqs)

    tokens = defaultdict(float)
    listp = defaultdict(float)
    for r in reqs:
        u = r["usage"]
        cached = ((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
        tokens[r["tenant"]] += u["prompt_tokens"] + u["completion_tokens"]
        listp[r["tenant"]] += (
            LIST_PRICE["input"] * (u["prompt_tokens"] - cached)
            + LIST_PRICE["cached"] * cached
            + LIST_PRICE["output"] * u["completion_tokens"]
        )

    # Time share: every tick of wall time is split equally across in-flight requests.
    time_share = defaultdict(float)
    idle_ticks = 0
    n_ticks = int(end / TICK_S) + 1
    events: list[tuple[float, int, dict[str, Any]]] = []
    for r in reqs:
        events.append((r["send"], 1, r))
        events.append((r["done"], -1, r))
    events.sort(key=lambda e: (e[0], e[1]))
    active: dict[int, dict[str, Any]] = {}
    cursor = 0
    for tick in range(n_ticks):
        t = tick * TICK_S
        while cursor < len(events) and events[cursor][0] <= t:
            _, kind, r = events[cursor]
            if kind == 1:
                active[r["rid"]] = r
            else:
                active.pop(r["rid"], None)
            cursor += 1
        if not active:
            idle_ticks += 1
            continue
        share = 1.0 / len(active)
        for r in active.values():
            time_share[r["tenant"]] += share

    # KV token-seconds: prompt held through prefill, growing linearly through decode.
    kv = defaultdict(float)
    for r in reqs:
        p, o = r["usage"]["prompt_tokens"], r["usage"]["completion_tokens"]
        kv[r["tenant"]] += p * (r["first"] - r["send"]) + (p + o / 2) * (r["done"] - r["first"])
    samples = [s for s in run["metrics"] if "vllm:kv_cache_usage_perc" in s and s["t"] <= end]
    kv_usage = statistics.fmean(s["vllm:kv_cache_usage_perc"] for s in samples) if samples else None

    def norm(raw: dict[str, float], scale: float = 1.0) -> dict[str, float]:
        total = sum(raw.values())
        return {t: scale * raw[t] / total for t in tenants}

    time_overhead = idle_ticks / n_ticks
    mem_overhead = (1 - kv_usage) if kv_usage is not None else None
    out = {
        "tokens": {**norm(tokens), "__overhead__": 0.0},
        "list_price": {**norm(listp), "__overhead__": 0.0},
        "time_share": {**norm(time_share, 1 - time_overhead), "__overhead__": time_overhead},
    }
    if mem_overhead is not None:
        out["kv_memory"] = {**norm(kv, 1 - mem_overhead), "__overhead__": mem_overhead}
    return out


def _quantile(values: list[float], q: float) -> float:
    """Linear-interpolated quantile; `statistics.quantiles` needs n > 1."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def spread(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "median": statistics.median(values) if values else float("nan"),
        "mean": statistics.fmean(values) if values else float("nan"),
        "p25": _quantile(values, 0.25),
        "p75": _quantile(values, 0.75),
        "min": min(values) if values else float("nan"),
        "max": max(values) if values else float("nan"),
    }


def bootstrap_ci(values: list[float], reps: int = 2000, seed: int = 11) -> list[float] | None:
    """Percentile bootstrap for the mean. Resamples windows, not requests: the
    time-share meter depends on which requests overlap, so a request-level
    resample would destroy the concurrency it measures."""
    if len(values) < 3:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        statistics.fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(reps)
    )
    return [means[int(0.025 * reps)], means[int(0.975 * reps)]]


def window_stability(
    run: dict[str, Any], window_s: float = 10.0, skip_windows: int = 1
) -> dict[str, Any]:
    """Recompute both meters inside consecutive windows of one run.

    One two-minute run per load level gives no error bar. This does not
    manufacture one — it cannot see run-to-run or seed-to-seed variation — but it
    does say how much the answer moves as the traffic sample changes within a run,
    which is a lower bound on the uncertainty and the only one the existing data
    supports. Repeat runs (`bench.py --repeats`) are what bound the rest.

    The first window is dropped. While the pipeline fills, the requests that have
    *completed* are disproportionately the short ones, which biases any share
    computed by completion time; measured across all four loads that window sits
    3–4× above the steady-state divergence and is an artifact of the windowing,
    not of the meters.
    """
    reqs = [r for r in run["requests"] if r["usage"] and r["first"] is not None]
    if not reqs:
        return {}
    end = max(r["done"] for r in reqs)
    n_windows = int(end // window_s)
    if n_windows - skip_windows < 2:
        return {}

    rows = []
    for w in range(skip_windows, n_windows):
        lo, hi = w * window_s, (w + 1) * window_s
        tokens: dict[str, float] = defaultdict(float)
        for r in reqs:
            if lo <= r["done"] < hi:
                u = r["usage"]
                tokens[r["tenant"]] += u["prompt_tokens"] + u["completion_tokens"]

        share_time: dict[str, float] = defaultdict(float)
        busy = 0
        for tick in range(int(window_s / TICK_S)):
            t = lo + tick * TICK_S
            active = [r for r in reqs if r["send"] <= t < r["done"]]
            if not active:
                continue
            busy += 1
            for r in active:
                share_time[r["tenant"]] += 1.0 / len(active)

        total_tokens = sum(tokens.values())
        if not total_tokens or not busy:
            continue
        # Both shares are normalized within the window, i.e. overhead redistributed.
        tok = tokens.get("search", 0.0) / total_tokens
        tim = share_time.get("search", 0.0) / busy
        rows.append({
            "window": w,
            "requests_completed": sum(1 for r in reqs if lo <= r["done"] < hi),
            "search_tokens": tok,
            "search_time_share": tim,
            "divergence_pts": 100 * (tok - tim),
        })

    if len(rows) < 2:
        return {}
    divergence = [r["divergence_pts"] for r in rows]
    return {
        "window_s": window_s,
        "skipped_warmup_windows": skip_windows,
        "windows": rows,
        "search_tokens": spread([r["search_tokens"] * 100 for r in rows]),
        "search_time_share": spread([r["search_time_share"] * 100 for r in rows]),
        "divergence_pts": spread(divergence),
        "divergence_mean_ci95": bootstrap_ci(divergence),
    }


def across_repeats(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Spread of the headline numbers over independent runs at one load level."""
    def pull(path: tuple[str, ...]) -> list[float]:
        out = []
        for e in entries:
            node: Any = e
            for key in path:
                node = node[key]
            if node is not None:
                out.append(float(node))
        return out

    divergence = pull(("divergence_search_tokens_vs_time_pts",))
    return {
        "repeats": len(entries),
        "seeds": [e.get("seed") for e in entries],
        "divergence_pts": spread(divergence),
        "divergence_mean_ci95": bootstrap_ci(divergence),
        "output_tokens_per_s": spread(pull(("output_tokens_per_s",))),
        "ttft_p50_ms": spread([v * 1000 for v in pull(("latency", "all", "ttft_s", "p50"))]),
        "search_tokens_pct": spread([100 * v for v in pull(("meters", "tokens", "search"))]),
    }


def read_raw(path: Path) -> str | None:
    """Raw files are committed gzipped; accept either form."""
    if path.exists():
        return path.read_text(errors="replace")
    packed = path.with_name(path.name + ".gz")
    if packed.exists():
        return gzip.decompress(packed.read_bytes()).decode("utf-8", "replace")
    return None


def gpu_stats(nvsmi: Path, start: float, end: float) -> dict[str, Any]:
    text = read_raw(nvsmi)
    if text is None:
        return {}
    util, power, mem = [], [], []
    for row in csv.reader(text.splitlines()):
        if len(row) < 6:
            continue
        try:
            # The droplet logs UTC; parse it as UTC, not as this machine's local time.
            stamp = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
            ts = stamp.replace(tzinfo=UTC).timestamp()
        except ValueError:
            continue
        if start <= ts <= end:
            util.append(float(row[2]))
            mem.append(float(row[3]))
            power.append(float(row[5]))
    return {
        "samples": len(util),
        "gpu_util_pct_mean": statistics.fmean(util) if util else None,
        "power_w_mean": statistics.fmean(power) if power else None,
        "power_w_max": max(power) if power else None,
        "memory_used_mib_max": max(mem) if mem else None,
    }


def server_facts(log: Path) -> dict[str, Any]:
    text = read_raw(log)
    if text is None:
        return {}
    facts: dict[str, Any] = {}
    for key, pattern in {
        "kv_cache_tokens": r"GPU KV cache size:\s*([\d,]+)\s*tokens",
        "max_concurrency": r"Maximum concurrency for [\d,]+ tokens per request:\s*([\d.]+)x",
        "model_load_gib": r"Model loading took\s*([\d.]+)\s*GiB",
        "vllm_version": r"vLLM API server version\s*([^\s]+)",
    }.items():
        match = re.search(pattern, text)
        if match:
            facts[key] = match.group(1).replace(",", "")
    return facts


def simulator_twin(rate: float, duration: float) -> dict[str, Any]:
    run = simulate(rate=rate, duration_s=duration, seed=7)
    stats = kv_attr.tenant_stats(run)
    prompt = sum(s["prompt_tokens"] for s in stats.values())
    cached = sum(s["cached_prompt_tokens"] for s in stats.values())
    split = kv_attr.shares(run, stats)
    return {
        "requests": len(run.requests),
        "output_tokens_per_s": run.tokens_generated / run.wall_s,
        "cache_hit_rate": cached / prompt if prompt else 0.0,
        "ttft_p50_s": {t: s["ttft_p50_s"] for t, s in stats.items()},
        "shares": {m: split[m] for m in ("tokens", "list_price", "compute", "memory")},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=common.RESULTS / "gpu_validation" / "raw")
    parser.add_argument("--out", type=Path, default=common.RESULTS / "gpu_validation")
    args = parser.parse_args(argv)

    meta = json.loads((args.raw / "meta.json").read_text())
    facts = server_facts(args.raw / "vllm.log")

    # rate -> [(repeat index, path)]; `rate_8.json` and `rate_8_r3.json` both belong here.
    by_rate: dict[float, list[tuple[int, Path]]] = defaultdict(list)
    for path in args.raw.glob("rate_*.json*"):
        match = re.fullmatch(r"rate_([\d.]+)(?:_r(\d+))?\.json(?:\.gz)?", path.name)
        if match:
            base = path if path.suffix == ".json" else path.with_suffix("")
            by_rate[float(match.group(1))].append((int(match.group(2) or 0), base))

    runs = []
    for rate in sorted(by_rate):
        entries = [
            analyze_run(json.loads(read_raw(path) or "{}"), args.raw, rep)
            for rep, path in sorted(by_rate[rate])
        ]
        entry = dict(entries[0])
        if len(entries) > 1:
            entry["repeat_count"] = len(entries)
            entry["representative_repeat"] = entries[0].get("repeat", 0)
            entry["across_repeats"] = across_repeats(entries)
            entry["repeats"] = entries
        # `headline` is what the paper quotes: identical to the single run when there
        # is one, the median across independent runs when there are several.
        entry["headline"] = {
            "divergence_pts": statistics.median(
                e["divergence_search_tokens_vs_time_pts"] for e in entries
            ),
            "search_tokens_pct": statistics.median(
                100 * kv_attr.redistribute(e["meters"]["tokens"])["search"] for e in entries
            ),
            "search_time_share_pct": statistics.median(
                100 * kv_attr.redistribute(e["meters"]["time_share"])["search"] for e in entries
            ),
            "runs": len(entries),
        }
        runs.append(entry)

    result = {"study": "gpu_validation", "meta": meta, "server": facts, "runs": runs}
    common.dump_json(args.out / "metrics.json", result)

    print(f"model {meta['model']} · vLLM {facts.get('vllm_version', meta['vllm'])} · "
          f"KV cache {facts.get('kv_cache_tokens', '?')} tokens")
    for e in runs:
        lat = e["latency"]["all"]
        ttft, tpot = lat["ttft_s"], lat["tpot_s"]
        print(f"rate {e['rate_rps']:>4g} rps: {e['requests']} req, {e['errors']} err, "
              f"{e['output_tokens_per_s']:.0f} out tok/s, "
              f"cache hit {e['cache_hit_rate_usage']:.0%}, "
              f"TTFT p50 {ttft['p50'] * 1e3:.0f} ms p95 {ttft['p95'] * 1e3:.0f} ms, "
              f"TPOT p50 {tpot['p50'] * 1e3:.1f} ms, e2e p50 {lat['e2e_s']['p50']:.2f} s, "
              f"GPU util {e['gpu'].get('gpu_util_pct_mean') or 0:.0f}%, "
              f"search tokens-vs-time {e['divergence_search_tokens_vs_time_pts']:+.1f} pts")
        within = e.get("within_run") or {}
        if within:
            d = within["divergence_pts"]
            ci = within.get("divergence_mean_ci95")
            band = f", mean 95% CI {ci[0]:+.1f}..{ci[1]:+.1f}" if ci else ""
            print(f"{'':>13}within run: {d['n']} × {within['window_s']:g}s windows, "
                  f"divergence median {d['median']:+.1f}, IQR {d['p25']:+.1f}..{d['p75']:+.1f}, "
                  f"range {d['min']:+.1f}..{d['max']:+.1f} pts{band}")
        if e.get("across_repeats"):
            a = e["across_repeats"]["divergence_pts"]
            print(f"{'':>13}across {e['repeat_count']} runs: divergence median {a['median']:+.1f}, "
                  f"range {a['min']:+.1f}..{a['max']:+.1f} pts")
    print(f"wrote {args.out / 'metrics.json'}")
    return 0


def analyze_run(run: dict[str, Any], raw: Path, repeat: int = 0) -> dict[str, Any]:
    """One (rate, repeat) benchmark file turned into the published numbers."""
    ok = [r for r in run["requests"] if r["usage"]]
    prompt = sum(r["usage"]["prompt_tokens"] for r in ok)
    cached = sum(((r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
                 for r in ok)
    samples = run["metrics"]
    busy = [s for s in samples if s.get("vllm:num_requests_running", 0) > 0]
    start = run.get("started_unix", 0.0)
    entry = {
        "rate_rps": run["rate_rps"],
        "repeat": run.get("repeat", repeat),
        "seed": run.get("seed"),
        "duration_s": run["duration_s"],
        "wall_s": run["wall_s"],
        "requests": len(run["requests"]),
        "completed_per_s": len(run["requests"]) / run["wall_s"],
        "errors": run["errors"],
        "output_tokens_per_s": sum(r["usage"]["completion_tokens"] for r in ok) / run["wall_s"],
        "prompt_tokens_per_s": prompt / run["wall_s"],
        "cache_hit_rate_usage": cached / prompt if prompt else None,
        "busy_fraction_engine": len(busy) / len(samples) if samples else None,
        "max_waiting": max((s.get("vllm:num_requests_waiting", 0) for s in samples), default=0),
        "metric_sample_interval_s": _sample_interval(samples),
        "latency": latency(run["requests"]),
        "meters": meters(run),
        "within_run": window_stability(run),
        "gpu": gpu_stats(raw / "nvidia_smi.csv", start, start + run["wall_s"]),
        "simulator": simulator_twin(run["rate_rps"], run["duration_s"]),
    }
    entry["divergence_search_tokens_vs_time_pts"] = 100 * (
        kv_attr.redistribute(entry["meters"]["tokens"])["search"]
        - kv_attr.redistribute(entry["meters"]["time_share"])["search"]
    )
    return entry


def _sample_interval(samples: list[dict[str, Any]]) -> float | None:
    """Median gap between telemetry samples: the resolution any "never queued"
    claim is actually entitled to."""
    stamps = sorted(s["t"] for s in samples if "t" in s)
    gaps = [b - a for a, b in pairwise(stamps)]
    return statistics.median(gaps) if gaps else None


if __name__ == "__main__":
    raise SystemExit(main())
