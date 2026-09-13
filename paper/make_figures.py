"""Render every paper figure from case_studies/results/*/metrics.json.

    python -m paper.make_figures

Figures are written as SVG (text converted to paths, so Typst needs no fonts)
plus PNG for the README. Colors follow one fixed categorical order so a tenant
or source keeps its hue across figures; status colors appear only where the
mark means good / warning / critical.
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
OUT = ROOT / "paper" / "figures"

INK, INK_2, GRID, AXIS = "#17201b", "#4f5b54", "#e4e7e3", "#b9c2bb"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
NEUTRAL = "#c9cfca"
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"
TENANT = {"search": CAT[0], "agents": CAT[1], "platform": CAT[2], "sandbox": CAT[3]}
SOURCE = {"opencost": CAT[0], "litellm": CAT[1], "openai": CAT[2], "anthropic": CAT[3]}
# Label ink per fill, chosen by fill luminance.
ON_FILL = {CAT[0]: "white", CAT[1]: "white", CAT[2]: INK, CAT[3]: INK, NEUTRAL: INK,
           GOOD: "white", WARN: INK, CRIT: "white"}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
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
        "svg.fonttype": "path",
    }
)


def load(study: str) -> dict[str, Any]:
    return json.loads((RESULTS / study / "metrics.json").read_text())


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
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
            ax.barh(i, value, left=left, height=0.62, color=colors[key], edgecolor="white",
                    linewidth=1.2)
            if value >= min_label:
                ax.text(left + value / 2, i, f"{value:.0%}", ha="center", va="center",
                        fontsize=7, color=ON_FILL.get(colors[key], INK))
            left += value
    ax.set_yticks(range(len(rows)), [label for label, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    return [Patch(color=colors[k], label=(names or {}).get(k, k)) for k in order]


def legend_above(ax: plt.Axes, ncol: int, handles: list[Patch] | None = None) -> None:
    ax.legend(handles=handles, ncol=ncol, loc="lower left", bbox_to_anchor=(0, 1.0),
              handlelength=1.0, columnspacing=1.2, fontsize=7.5)


# --- kv_cache ------------------------------------------------------------------------


def kv_cache() -> None:
    m = load("kv_cache")
    split = m["headline"]["shares"]
    names = {"tokens": "raw tokens", "list_price": "list-price tokens", "compute": "step time",
             "memory": "KV block-seconds", "blended": "blended"}
    order = ["search", "agents", "platform", "sandbox", "__overhead__"]
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
                                 ["search", "agents", "platform", "sandbox"], TENANT)
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
    a.barh(ys, compute, height=0.6, color=CAT[0], edgecolor="white", linewidth=1.2, label="compute")
    a.barh(ys, comm, left=compute, height=0.6, color=CAT[1], edgecolor="white", linewidth=1.2,
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
        for src in ("opencost", "litellm", "openai", "anthropic"):
            v = float(by_source.get(src, 0)) / 1000
            if v <= 0:
                continue
            a.barh(y, v, left=left, height=0.6, color=SOURCE[src], edgecolor="white", linewidth=1.2,
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


def main() -> int:
    print(f"rendering figures into {OUT}")
    for render in (kv_cache, torch_kv, distributed, hybrid, use_cases):
        render()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
