"""Render every figure from case_studies/results/*/metrics.json, in two themes.

    python -m paper.make_figures

* paper  (paper/figures/*.svg, *.png)       light, for the PDF, README and Medium
* ledger (paper/figures/ledger/*.png)       dark, on the timurista.ai "Ledger" system,
                                            for the project site and timurista.ai

SVG text is converted to paths, so Typst needs no fonts. A tenant or source
keeps its hue across figures within a theme. In the ledger theme rose is
reserved for spend with no owner and is always hatched as well as colored,
and the categorical order was checked for color-vision separation on the
dark surface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "case_studies" / "results"

THEMES: dict[str, dict[str, Any]] = {
    "paper": {
        "out": ROOT / "paper" / "figures",
        "formats": ("svg", "png"),
        "ink": "#17201b", "ink_2": "#4f5b54", "grid": "#e4e7e3", "axis": "#b9c2bb",
        "face": "#ffffff", "edge": "#ffffff",
        "cat": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
        "neutral": "#c9cfca",
        "good": "#0ca30c", "warn": "#fab219", "crit": "#d03b3b",
        "tenant_order": ["search", "agents", "platform", "sandbox"],
        "tenant": {"search": 0, "agents": 1, "platform": 2, "sandbox": 3},
        "source": {"opencost": 0, "litellm": 1, "openai": 2, "anthropic": 3},
        "dark_label": {2, 3},
        "hatch": set(),
    },
    "ledger": {
        "out": ROOT / "paper" / "figures" / "ledger",
        "formats": ("png",),
        "ink": "#E8EDF3", "ink_2": "#9AA6B5", "grid": "#1F2833", "axis": "#2A3441",
        "face": "#0F141B", "edge": "#0F141B",
        # teal, blue, violet, rose. Blue and violet are never stacked next to each other.
        "cat": ["#1FA88C", "#3F7FE0", "#9A79E6", "#D9634F"],
        "neutral": "#3A4553",
        "good": "#1FA88C", "warn": "#9A79E6", "crit": "#D9634F",
        "tenant_order": ["agents", "search", "platform", "sandbox"],
        "tenant": {"search": 0, "agents": 1, "platform": 2, "sandbox": 3},
        # Stacked in this order (violet, teal, blue, magenta); checked for CVD separation.
        "source": {"openai": 2, "opencost": 0, "litellm": 1, "anthropic": None},
        "source_extra": "#C75A9E",
        "dark_label": {0, 3},
        # Categories that mean "nobody owns this": hatched as well as colored.
        "hatch": {"sandbox", "unowned", "__overhead__", "idle"},
    },
}

# Set by use_theme(); the render functions below read these module globals.
INK = INK_2 = GRID = AXIS = EDGE = NEUTRAL = GOOD = WARN = CRIT = ""
CAT: list[str] = []
TENANT: dict[str, str] = {}
TENANT_ORDER: list[str] = []
SOURCE: dict[str, str] = {}
SOURCE_ORDER: list[str] = []
ON_FILL: dict[str, str] = {}
HATCH: set[str] = set()
OUT = ROOT / "paper" / "figures"
FORMATS: tuple[str, ...] = ("svg", "png")


def use_theme(name: str) -> None:
    global INK, INK_2, GRID, AXIS, EDGE, NEUTRAL, GOOD, WARN, CRIT, CAT, TENANT, TENANT_ORDER
    global SOURCE, SOURCE_ORDER, ON_FILL, HATCH, OUT, FORMATS
    t = THEMES[name]
    INK, INK_2, GRID, AXIS, EDGE = t["ink"], t["ink_2"], t["grid"], t["axis"], t["edge"]
    NEUTRAL, GOOD, WARN, CRIT = t["neutral"], t["good"], t["warn"], t["crit"]
    CAT = list(t["cat"])
    TENANT = {k: CAT[i] for k, i in t["tenant"].items()}
    TENANT_ORDER = list(t["tenant_order"])
    SOURCE = {k: (CAT[i] if i is not None else t["source_extra"]) for k, i in t["source"].items()}
    SOURCE_ORDER = list(t["source"])
    dark = "#07090D" if name == "ledger" else INK
    ON_FILL = {c: (dark if i in t["dark_label"] else "white") for i, c in enumerate(CAT)}
    ON_FILL.update({NEUTRAL: INK, GOOD: dark, WARN: "white",
                    CRIT: dark if name == "ledger" else "white"})
    HATCH = set(t["hatch"])
    OUT, FORMATS = t["out"], t["formats"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.facecolor": t["face"],
            "figure.facecolor": t["face"],
            "savefig.facecolor": t["face"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "hatch.color": "#07090D" if name == "ledger" else "#ffffff",
            "hatch.linewidth": 0.8,
            "svg.fonttype": "path",
            "svg.hashsalt": "unalloc",
        }
    )


use_theme("paper")


def load(study: str) -> dict[str, Any]:
    return json.loads((RESULTS / study / "metrics.json").read_text())


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for fmt in FORMATS:
        # No date in the metadata: with a fixed hashsalt (see use_theme) an unchanged
        # figure then re-renders byte-identically, so `make paper` leaves a clean tree.
        fig.savefig(OUT / f"{name}.{fmt}", dpi=220, bbox_inches="tight",
                    metadata={"Date": None} if fmt == "svg" else None)
    plt.close(fig)
    print(f"  {name}")


def stacked_share_bars(
    ax: plt.Axes,
    rows: list[tuple[str, dict[str, float]]],
    order: list[str],
    colors: dict[str, str],
    names: dict[str, str] | None = None,
    min_label: float = 0.07,
) -> list[Patch]:
    """One 100% bar per row; segments in a fixed order, labelled when wide enough.

    Returns legend handles for every category, including ones absent from the
    first bar - otherwise a category that only appears lower down has no key.
    """
    for i, (_, parts) in enumerate(rows):
        left = 0.0
        for key in order:
            value = parts.get(key, 0.0)
            if value <= 0:
                continue
            ax.barh(i, value, left=left, height=0.62, color=colors[key], edgecolor=EDGE,
                    linewidth=1.2, hatch="////" if key in HATCH else None)
            if value >= min_label:
                ax.text(left + value / 2, i, f"{value:.0%}", ha="center", va="center",
                        fontsize=7, color=ON_FILL.get(colors[key], INK),
                        bbox={"boxstyle": "round,pad=0.15", "fc": colors[key], "ec": "none"}
                        if key in HATCH else None)
            left += value
    ax.set_yticks(range(len(rows)), [label for label, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    return [
        Patch(facecolor=colors[k], edgecolor=EDGE, hatch="////" if k in HATCH else None,
              label=(names or {}).get(k, k))
        for k in order
    ]


def legend_above(ax: plt.Axes, ncol: int, handles: list[Patch] | None = None) -> None:
    ax.legend(handles=handles, ncol=ncol, loc="lower left", bbox_to_anchor=(0, 1.0),
              handlelength=1.0, columnspacing=1.2, fontsize=7.5)


# --- kv_cache ------------------------------------------------------------------------


def kv_cache() -> None:
    m = load("kv_cache")
    split = m["headline"]["shares"]
    names = {"tokens": "raw tokens", "list_price": "list-price tokens", "compute": "step time",
             "memory": "KV block-seconds", "blended": "blended"}
    order = [*TENANT_ORDER, "__overhead__"]
    colors = {**TENANT, "__overhead__": NEUTRAL}
    fig, ax = plt.subplots(figsize=(6.4, 2.3))
    handles = stacked_share_bars(ax, [(names[k], split[k]) for k in names], order, colors,
                                 names={"__overhead__": "overhead (no tenant)"})
    ax.set_title("Share of one shared vLLM pod's bill under five metering rules", pad=18)
    legend_above(ax, 5, handles)
    save(fig, "kv_shares")

    sweep = [p for p in m["sweep"] if p["prefix_caching"]]
    rates = [p["rate_rps"] for p in sweep]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.3))
    meters = (("compute", "step time", CAT[0]), ("memory", "KV block-seconds", CAT[1]))
    for key, label, color in meters:
        ys = [p["overhead"][key] for p in sweep]
        a.plot(rates, ys, color=color, marker="o", markersize=3.5, label=label)
    a.set_title("Bill no request meter can see")
    a.set_xlabel("offered load (requests/s)")
    a.yaxis.set_major_formatter(PercentFormatter(1.0))
    a.set_ylim(0, 1.02)
    a.legend(fontsize=7)
    for key, label, color in (("tokens", "raw tokens", CAT[0]), ("compute", "step time", CAT[1]),
                              ("memory", "KV block-seconds", CAT[2])):
        ys = [p["shares_redistributed"][key]["search"] for p in sweep]
        b.plot(rates, ys, color=color, marker="o", markersize=3.5, label=label)
    b.set_title("search's share, overhead redistributed")
    b.set_xlabel("offered load (requests/s)")
    b.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    b.set_ylim(0, None)
    b.legend(fontsize=7)
    fig.tight_layout(w_pad=2.5)
    save(fig, "kv_sweep")


# --- torch_kv --------------------------------------------------------------------------


def torch_kv() -> None:
    m = load("torch_kv")
    s = m["e1_scaling"]["series"]
    ctx = s["context"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.3))
    a.plot(ctx, [v * 1e3 for v in s["uncached_step_s"]], color=CAT[1], marker="o", markersize=3.5,
           label="recompute prefix")
    a.plot(ctx, [v * 1e3 for v in s["cached_step_s"]], color=CAT[0], marker="o", markersize=3.5,
           label="KV cache")
    a.set_xscale("log", base=2)
    a.set_yscale("log")
    a.set_xticks(ctx, [str(c) for c in ctx])
    a.set_xlabel("context length (tokens)")
    a.set_ylabel("ms per decode step")
    a.set_title("Decode step latency, measured")
    a.legend(fontsize=7)
    b.plot(ctx, s["speedup"], color=CAT[0], marker="o", markersize=3.5)
    b.annotate(f"{s['speedup'][-1]:.0f}× at {ctx[-1]}", (ctx[-1], s["speedup"][-1]),
               textcoords="offset points", xytext=(-8, 0), ha="right", va="center",
               fontsize=8, color=INK)
    b.set_xscale("log", base=2)
    b.set_xticks(ctx, [str(c) for c in ctx])
    b.set_xlabel("context length (tokens)")
    b.set_title("Speedup from the KV cache")
    b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}×"))
    fig.tight_layout(w_pad=2.5)
    save(fig, "torch_e1")

    att = m["attribution"]
    names = {
        "requests": "requests",
        "tokens": "raw tokens",
        "tokens_list_price": "list-price tokens",
        "analytic_flops": "analytic FLOPs",
        "compute_seconds": "measured compute",
        "kv_byte_seconds": "measured KV memory",
        "blended": "blended (measured)",
    }
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    handles = stacked_share_bars(ax, [(names[k], att["shares"][k]) for k in names],
                                 TENANT_ORDER, TENANT)
    ax.axhline(3.5, color=AXIS, linewidth=0.8)
    ax.set_title("A real transformer's serving bill, split seven ways", pad=18)
    legend_above(ax, 4, handles)
    save(fig, "torch_shares")


# --- distributed -----------------------------------------------------------------------


def distributed() -> None:
    m = load("distributed")
    case = m["bench"]["main_case"]
    configs = [c for c in m["configs"] if c["id"] in ("tp1", "tp2", "tp4", "pp2")]
    labels, compute, comm, tps = [], [], [], []
    for c in configs:
        t = c["timing"][case]
        ranks = t["per_rank"]
        plural = "s" if c["world_size"] > 1 else ""
        labels.append(f"{c['id'].upper()} · {c['world_size']} rank{plural}")
        compute.append(sum(r["compute_s"] for r in ranks) / len(ranks) * 1e3)
        comm.append(sum(r["comm_s"] for r in ranks) / len(ranks) * 1e3)
        tps.append(t["tokens_per_s"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.3), gridspec_kw={"width_ratios": [3, 2]})
    ys = range(len(labels))
    a.barh(ys, compute, height=0.6, color=CAT[0], edgecolor=EDGE, linewidth=1.2, label="compute")
    a.barh(ys, comm, left=compute, height=0.6, color=CAT[1], edgecolor=EDGE, linewidth=1.2,
           label="in collectives / waiting on a peer")
    for y, (c1, c2) in enumerate(zip(compute, comm, strict=True)):
        share = c2 / (c1 + c2) if c1 + c2 else 0
        a.text(c1 + c2 + 2, y, f"{share:.0%} comm", va="center", fontsize=7, color=INK_2)
    a.set_yticks(list(ys), labels)
    a.invert_yaxis()
    a.grid(axis="y", visible=False)
    a.tick_params(axis="y", length=0)
    a.set_xlabel("ms per rank, one generate call")
    a.set_title("Where each rank's time goes", pad=18)
    a.set_xlim(0, max(x + y for x, y in zip(compute, comm, strict=True)) * 1.3)
    legend_above(a, 2)
    b.barh(ys, tps, height=0.6, color=CAT[0])
    for y, v in enumerate(tps):
        b.text(v, y, f" {v:,.0f}", va="center", fontsize=7, color=INK_2)
    b.set_yticks(list(ys), [""] * len(labels))
    b.invert_yaxis()
    b.grid(axis="y", visible=False)
    b.tick_params(axis="y", length=0)
    b.set_xlim(0, max(tps) * 1.35)
    b.set_xlabel("tokens/s (CPU, gloo)")
    b.set_title("Throughput", pad=18)
    fig.tight_layout(w_pad=1.5)
    save(fig, "dist_timing")

    scenarios = m["attribution"]["scenarios"]
    names = {"s1": "S1  team on leaders only", "s2": "S2  + fallback to name",
             "s3": "S3  team on worker template"}
    rows = []
    for key, label in names.items():
        acc = scenarios[key]["summary"]["accuracy"]
        total = sum(float(acc[k]) for k in ("correct_usd", "misattributed_usd",
                                            "unallocated_owned_usd", "unallocated_idle_usd"))
        rows.append((label, {
            "correct": float(acc["correct_usd"]) / total,
            "misattributed": float(acc["misattributed_usd"]) / total,
            "unowned": float(acc["unallocated_owned_usd"]) / total,
            "idle": float(acc["unallocated_idle_usd"]) / total,
        }))
    fig, ax = plt.subplots(figsize=(6.4, 1.8))
    handles = stacked_share_bars(
        ax, rows, ["correct", "misattributed", "unowned", "idle"],
        {"correct": GOOD, "misattributed": WARN, "unowned": CRIT, "idle": NEUTRAL},
        names={"correct": "right team", "misattributed": "'allocated' to a non-team value",
               "unowned": "unallocated worker pods", "idle": "cluster idle"},
    )
    ax.set_title("A LeaderWorkerSet month ($38.4K) under three labeling states", pad=18)
    legend_above(ax, 4, handles)
    save(fig, "dist_scenarios")


# --- hybrid e2e ------------------------------------------------------------------------


def hybrid() -> None:
    m = load("hybrid_e2e")
    sc = m["scenarios"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.1), gridspec_kw={"width_ratios": [3, 2]})
    rows = [("gateway + cluster", sc["A_gateway_and_cluster"]["by_source"]),
            ("every source on", sc["B_all_sources"]["by_source"])]
    for y, (_, by_source) in enumerate(rows):
        left = 0.0
        for src in SOURCE_ORDER:
            v = float(by_source.get(src, 0)) / 1000
            if v <= 0:
                continue
            a.barh(y, v, left=left, height=0.6, color=SOURCE[src], edgecolor=EDGE, linewidth=1.2,
                   label=src if y == 1 else None)
            left += v
        a.text(left + 0.6, y, f"${left:,.1f}K", va="center", fontsize=7, color=INK_2)
    a.set_yticks([0, 1], [r[0] for r in rows])
    a.invert_yaxis()
    a.grid(axis="y", visible=False)
    a.tick_params(axis="y", length=0)
    a.set_xlim(0, float(sc["B_all_sources"]["total_usd"]) / 1000 * 1.2)
    a.set_xlabel("monthly spend in the ledger ($K)")
    a.set_title("Turning every source on double counts", pad=18)
    legend_above(a, 4)
    pages = m["scenarios"]["P_pagination"]
    providers = list(pages)
    shares = [float(pages[p]["first_page_usd"]) / float(pages[p]["all_pages_usd"])
              for p in providers]
    b.bar(providers, shares, width=0.5, color=CAT[0])
    b.axhline(1.0, color=INK_2, linewidth=0.8, linestyle=(0, (2, 2)))
    b.text(0.5, 1.03, "true spend", va="bottom", ha="center", fontsize=7, color=INK_2)
    for x, v in enumerate(shares):
        b.text(x, v, f"{v:.0%}", ha="center", va="bottom", fontsize=7, color=INK)
    b.set_ylim(0, 1.15)
    b.yaxis.set_major_formatter(PercentFormatter(1.0))
    b.grid(axis="x", visible=False)
    b.set_title("Reported if only page 1 is read", pad=18)
    fig.tight_layout(w_pad=1.5)
    save(fig, "hybrid")


# --- use cases -------------------------------------------------------------------------


def use_cases() -> None:
    m = load("use_cases")
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.3))
    for (key, label), color in zip((("bundled_fixtures", "README fixtures"),
                                    ("hybrid_org", "hybrid org")), CAT, strict=False):
        curve = m["U1_pareto"][key]["curve_pct"][:21]
        a.step(range(len(curve)), curve, where="post", color=color, label=label)
    a.axhline(10, color=INK_2, linewidth=0.8, linestyle=(0, (2, 2)))
    a.text(20, 10.5, "10%", ha="right", va="bottom", fontsize=7, color=INK_2)
    a.set_xlabel("label fixes, most expensive row first")
    a.set_ylabel("unallocated")
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    a.set_ylim(0, 100)
    a.set_title("The labeling backlog is steep")
    a.legend(fontsize=7)
    points = m["U3_break_even"]["points"]
    rates = [p["rate_rps"] for p in points]
    for tier, label, color in (("small_api_tier", "vs small API tier", CAT[0]),
                               ("mid_api_tier", "vs mid API tier", CAT[1])):
        b.plot(rates, [p[f"{tier}_cost_ratio"] for p in points], color=color, marker="o",
               markersize=3.5, label=label)
    b.axhline(1.0, color=INK_2, linewidth=0.8, linestyle=(0, (2, 2)))
    b.text(rates[-1], 1.08, "parity", ha="right", va="bottom", fontsize=7, color=INK_2)
    b.set_yscale("log")
    b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    b.set_xlabel("offered load on one GPU (requests/s)")
    b.set_title("Self-hosted cost ÷ API cost, same traffic")
    b.legend(fontsize=7)
    fig.tight_layout(w_pad=2.5)
    save(fig, "use_cases")


def gpu_validation() -> None:
    """Real vLLM on an H100: latency by load, and search's share under two meters."""
    if not (RESULTS / "gpu_validation" / "metrics.json").exists():
        return
    runs = load("gpu_validation")["runs"]
    rates = [r["rate_rps"] for r in runs]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(6.4, 2.25))
    for key, color in (("p50", CAT[0]), ("p95", CAT[1])):
        a.plot(rates, [r["latency"]["all"]["ttft_s"][key] * 1e3 for r in runs], color=color,
               marker="o", markersize=3.5, label=key)
        b.plot(rates, [r["latency"]["all"]["tpot_s"][key] * 1e3 for r in runs], color=color,
               marker="o", markersize=3.5, label=key)
    a.set_title("Time to first token")
    a.set_ylabel("ms")
    b.set_title("Time per output token")
    a.legend(fontsize=7)

    def search(split: dict[str, float]) -> float:
        return split["search"] / (1 - split.get("__overhead__", 0.0))

    # The whole-run share and the spread of the ten-second windows are two different
    # quantities, so they are drawn as two different things. An error bar hung off
    # the whole-run point would have to be clipped whenever that point sits outside
    # the window quartiles — which happens, because the whole-run share is weighted
    # by each window's traffic and a weighted aggregate need not lie inside the
    # quartiles of its parts. The bar is offset in x so both stay visible.
    for key, meter, color, label in (
        ("search_tokens", "tokens", CAT[0], "tokens (H100)"),
        ("search_time_share", "time_share", CAT[1], "time share (H100)"),
    ):
        c.plot(rates, [search(r["meters"][meter]) for r in runs], color=color,
               marker="o", markersize=3.5, label=label)
        windows = [(r["rate_rps"] * 1.07, (r.get("within_run") or {}).get(key)) for r in runs]
        spread = [(x, w["p25"] / 100, w["p75"] / 100) for x, w in windows if w]
        if spread:
            xs, p25, p75 = zip(*spread, strict=False)
            c.vlines(xs, p25, p75, color=color, alpha=0.5, linewidth=2.2)
    if any((r.get("within_run") or {}) for r in runs):
        # INK_2, not NEUTRAL: the print theme's neutral is a near-white grey that
        # disappears as a legend swatch.
        c.vlines([], [], [], color=INK_2, alpha=0.6, linewidth=2.2,
                 label="10 s window IQR")
    # Simulator points only where it is not saturated, so the comparison is fair.
    healthy = [r for r in runs if max(r["simulator"]["ttft_p50_s"].values()) < 1.0]
    c.plot([r["rate_rps"] for r in healthy],
           [search(r["simulator"]["shares"]["tokens"]) for r in healthy],
           color=CAT[0], linestyle=(0, (2, 2)), marker="s", markersize=3, label="tokens (sim)")
    c.plot([r["rate_rps"] for r in healthy],
           [search(r["simulator"]["shares"]["compute"]) for r in healthy],
           color=CAT[1], linestyle=(0, (2, 2)), marker="s", markersize=3, label="step time (sim)")
    c.set_title("search's share of the bill")
    c.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    for ax in (a, b, c):
        ax.set_xscale("log", base=2)
        ax.set_xticks(rates, [f"{r:g}" for r in rates])
        ax.set_xlabel("requests/s")
        ax.set_ylim(0, None)
    # Headroom above the lines so the legend never sits on the data.
    c.set_ylim(0, 0.36)
    c.legend(fontsize=5.5, loc="upper center", ncol=2, handlelength=1.6, columnspacing=0.8)
    fig.tight_layout(w_pad=1.8)
    save(fig, "gpu_validation")


def main() -> int:
    for theme in THEMES:
        use_theme(theme)
        print(f"rendering {theme} figures into {OUT}")
        for render in (kv_cache, torch_kv, distributed, hybrid, use_cases, gpu_validation):
            render()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
