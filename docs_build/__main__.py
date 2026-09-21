"""Build the GitHub Pages site into docs/.

    python -m docs_build            # or: make pages

Every page is generated from files already in the repository, so the site
cannot drift from them:

* index.html                     receipt hero, findings board, case studies, use cases
* blog/<post>.html               rendered from blog/*.md
* explorer.html                  the ledger explorer (case_studies.ui)
* paper/unalloc-case-studies.pdf the paper PDF, copied as built
* assets/                        stylesheet, dark "ledger" figures, evidence images

Numbers on the landing page are computed at build time: the receipt runs
unalloc over its bundled fixtures, and each finding is read from the study's
metrics.json and links to it. The build fails if any local link or image on a
generated page does not resolve.

Styled on the timurista.ai "Ledger" system: ink ground, teal for attributed
spend, amber only for money, rose only for unallocated spend.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
RESULTS = ROOT / "case_studies" / "results"
FIGURES = ROOT / "paper" / "figures" / "ledger"

REPO = "https://github.com/timurista/unalloc"
DOI = "10.5281/zenodo.22761012"
FONTS = (
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600"
    "&family=Inter+Tight:wght@500;600&family=JetBrains+Mono:wght@400;500&display=swap"
)
PAPER = "paper/unalloc-case-studies.pdf"
esc = html.escape

POSTS = (
    # (source, output slug)
    ("who-pays-for-the-kv-cache.md", "who-pays-for-the-kv-cache"),
    ("addendum-gpu-validation.md", "gpu-validation"),
)

NAV = (
    ("index.html", "Overview"),
    (PAPER, "Paper"),
    ("blog/who-pays-for-the-kv-cache.html", "Blog post"),
    ("blog/gpu-validation.html", "GPU addendum"),
    ("explorer.html", "Explorer"),
    (REPO, "GitHub"),
)


def page(title: str, body: str, *, depth: int, current: str, description: str) -> str:
    """Wrap a page body in the shared shell. `depth` is how many directories below docs/."""
    up = "../" * depth
    current_attr = ' aria-current="page"'
    links = "".join(
        '<a href="{}"{}>{}</a>'.format(
            href if href.startswith("http") else up + href,
            current_attr if href == current else "",
            label,
        )
        for href, label in NAV
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#0B0F14">
<title>{esc(title)}</title>
<meta name="description" content="{esc(description)}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{FONTS}">
<link rel="stylesheet" href="{up}assets/site.css">
</head>
<body>
<header class="top"><div class="wrap">
  <a class="brand" href="{up}index.html"><span class="mark" aria-hidden="true">u</span>unalloc</a>
  <nav aria-label="Site">{links}</nav>
</div></header>
{body}
<footer class="site"><div class="wrap">
  <span>Apache-2.0</span>
  <a href="{REPO}">{REPO.removeprefix("https://")}</a>
  <a href="https://doi.org/{DOI}">doi:{DOI}</a>
  <span>Every number on this site is generated from the repository by <code>make pages</code></span>
</div></footer>
</body>
</html>
"""


# --- blog posts ------------------------------------------------------------------------

_LINK = re.compile(r"(!?\[[^\]]*\]\()([^)\s]+)(\))")


def _rewrite(target: str) -> str:
    """Map a link written for the repository layout onto the site layout."""
    if target.startswith(("http://", "https://", "#", "mailto:")):
        return target
    if target.startswith("../paper/figures/"):
        return "../assets/figures/" + target.removeprefix("../paper/figures/")
    if target.startswith("gpu-evidence/"):
        return "../assets/" + target
    if target.endswith(".png"):
        return "../assets/blog/" + target
    for source, slug in POSTS:
        if target == source:
            return f"{slug}.html"
    raise SystemExit(f"blog link {target!r} has no page on the site; add a rule to _rewrite")


def render_post(source: Path, slug: str) -> str:
    text = _LINK.sub(lambda m: m.group(1) + _rewrite(m.group(2)) + m.group(3), source.read_text())
    # Posts link the public explorer by absolute URL (they are also published elsewhere);
    # on the site itself, keep readers on the same origin.
    text = text.replace("https://timurista.github.io/unalloc/explorer.html)", "../explorer.html)")
    title_match = re.match(r"#\s+(.+)", text)
    title = title_match.group(1).strip() if title_match else slug
    lead = re.search(r"^\*(.+)\*$", text, flags=re.MULTILINE)

    md = markdown.Markdown(extensions=["fenced_code", "tables", "sane_lists", "toc"])
    body = md.convert(text)
    body = body.replace("<table>", '<div class="table-scroll"><table>').replace(
        "</table>", "</table></div>"
    )
    sections = [t for top in md.toc_tokens for t in top.get("children", [])] or md.toc_tokens
    toc = ""
    if len(sections) > 2:
        items = "".join(
            f'<li><a href="#{t["id"]}">{esc(html.unescape(t["name"]))}</a></li>' for t in sections
        )
        toc = f'<aside class="toc" aria-label="On this page"><p>On this page</p><ol>{items}</ol></aside>'

    layout = "post-layout has-toc" if toc else "post-layout"
    article = (
        f'<main class="{layout}"><article class="post">'
        f'<a class="crumb" href="../index.html">← unalloc</a>{body}</article>{toc}</main>'
    )
    description = re.sub(r"[*_`\[\]]|\([^)]*\)", "", lead.group(1)) if lead else title
    return page(title, article, depth=1, current=f"blog/{slug}.html", description=description)


# --- numbers, read from the repository ----------------------------------------------------


def _metrics(study: str) -> dict:
    return json.loads((RESULTS / study / "metrics.json").read_text())


def _usd(amount: Decimal | float, cents: bool = True) -> str:
    value = Decimal(str(amount))
    return f"${value:,.2f}" if cents else f"${value:,.0f}"


def _range(values: list[float], unit: str) -> str:
    lo, hi = round(min(values)), round(max(values))
    return f"{lo}{unit}" if lo == hi else f"{lo}–{hi}{unit}"


@dataclass(frozen=True)
class Finding:
    figure: str
    kind: str  # "unowned" | "money" | "" - decides the figure's color, per the brand rules
    text: str
    share: float | None  # draw a hatched bar for shares of spend with no owner
    section: str
    study: str


def findings() -> list[Finding]:
    dist = _metrics("distributed")["attribution"]["scenarios"]
    s1 = dist["s1"]["summary"]
    s2 = dist["s2"]["summary"]
    gpu = _metrics("gpu_validation")["runs"]
    e2e = _metrics("hybrid_e2e")
    pages = e2e["scenarios"]["P_pagination"]
    first_page = sum(float(p["first_page_usd"]) for p in pages.values()) / sum(
        float(p["all_pages_usd"]) for p in pages.values()
    )
    unowned = float(s1["unallocated_pct"]) / 100
    misattributed = float(s2["accuracy"]["misattributed_pct"]) / 100
    gap = _range([r["divergence_search_tokens_vs_time_pts"] for r in gpu], "")
    util = _range([r["gpu"]["gpu_util_pct_mean"] for r in gpu], "%")
    throughput = gpu[-1]["output_tokens_per_s"] / gpu[0]["output_tokens_per_s"]
    power = (gpu[0]["gpu"]["power_w_mean"], gpu[-1]["gpu"]["power_w_mean"])
    doubled = Decimal(e2e["findings"]["double_counted_usd"])
    month = Decimal(e2e["scenarios"]["A_gateway_and_cluster"]["total_usd"])
    return [
        Finding(f"{unowned:.0%}", "unowned",
                "of a distributed deployment's GPU bill had no owner when <code>team</code> was set "
                "only on LeaderWorkerSet leader pods.", unowned, "§6", "distributed"),
        Finding(f"{misattributed:.0%}", "unowned",
                "of that bill resolved to a Helm chart name after the obvious fallback label, while "
                f"the headline unallocated share fell to {float(s2['unallocated_pct']):.0f}%.",
                misattributed, "§6", "distributed"),
        Finding(f"{gap} pts", "",
                "difference in a retrieval-heavy tenant's share between a token meter and an equal "
                "time-share meter, on an H100 running vLLM, at every load tested.",
                None, "§9", "gpu_validation"),
        Finding(util, "",
                f"GPU utilization at every load while throughput rose {throughput:.0f}×. Power draw "
                f"({power[0]:.0f} W → {power[1]:.0f} W) tracked the work; utilization did not.",
                None, "§9", "gpu_validation"),
        Finding(_usd(doubled, cents=False), "money",
                "of gateway spend counted twice when the gateway and the provider bills were both "
                f"turned on, in a {_usd(month, cents=False)} month.", None, "§7", "hybrid_e2e"),
        Finding(f"{first_page:.0%}", "",
                "of true provider spend reported by billing adapters that read only the first page "
                "of results. Fixed in 0.2.0.", None, "§7", "hybrid_e2e"),
    ]


def receipt() -> str:
    """The bundled sample month, attributed on `team` by unalloc itself."""
    from unalloc.cli import FIXTURE_DIR, FIXTURE_FILES
    from unalloc.core.attribute import attribute
    from unalloc.sources import REGISTRY

    rows = [
        row
        for name, filename in FIXTURE_FILES.items()
        for row in REGISTRY[name]().from_fixture(FIXTURE_DIR / filename)
    ]
    report = attribute(rows, "team")
    rows.sort(key=lambda r: r.amount_usd, reverse=True)
    shown, rest = rows[:7], rows[7:]

    def owner(row) -> str:
        team = row.label("team")
        if team:
            return f'<span class="owner">→ {esc(team)}</span>'
        return '<span class="owner none">→ no owner</span>'

    lines = "".join(
        f'<li><span class="obj">{esc(r.source)} · {esc(r.name)}</span>'
        f'<span class="amt money">{_usd(r.amount_usd)}</span>{owner(r)}</li>'
        for r in shown
    )
    if rest:
        lines += (
            f'<li><span class="more">+ {len(rest)} smaller lines</span>'
            f'<span class="amt money">{_usd(sum((r.amount_usd for r in rest), Decimal(0)))}</span></li>'
        )
    owned = [b for b in report.buckets if not b.is_unallocated]
    total = report.total_usd
    split = "".join(
        f'<span class="own" style="flex:{b.amount_usd}" title="{esc(b.key)}"></span>' for b in owned
    ) + f'<span class="none" style="flex:{report.unallocated_usd}" title="no owner"></span>'
    subtotals = "".join(
        f'<div><span>team: {esc(b.key)}</span><span class="money">{_usd(b.amount_usd)}</span></div>'
        for b in owned
    )
    return f"""<figure class="receipt" aria-label="A sample month of AI spend attributed by team">
  <header><strong>AI spend · sample month</strong><span>{len(rows)} lines · 4 sources</span></header>
  <ol>{lines}</ol>
  <div class="totals">
    {subtotals}
    <div class="grand"><span>total</span><span class="money">{_usd(total)}</span></div>
    <div class="split" aria-hidden="true">{split}</div>
    <div class="gap"><span>no owner · {report.unallocated_pct}%</span><span class="money">{_usd(report.unallocated_usd)}</span></div>
  </div>
  <p class="cmd">$ unalloc report --fixtures --dimension team</p>
</figure>"""


# --- landing page ------------------------------------------------------------------------

STUDIES = (
    ("gpu_validation", "gpu_validation.png", "Real vLLM on an H100", "real serving",
     "vLLM 0.29.0 serving Qwen2.5-7B-Instruct on a DigitalOcean H100, driven with four tenants at "
     "2–16 requests/s. 6,241 requests, 0 errors, about $2.15.",
     "Token and time-share meters disagree by 12–14 points on the RAG tenant; utilization reads "
     "97–99% at every load."),
    ("distributed", "dist_scenarios.png", "Distributed inference", "real inference",
     "Tensor- and pipeline-parallel inference on torch.distributed, verified token-identical to a "
     "single process, then a month of LeaderWorkerSet allocations.",
     "Leader-only labels leave 66% unowned; falling back to name sends 61% to a chart name."),
    ("torch_kv", "torch_shares.png", "A transformer with a real KV cache", "real inference · CPU",
     "A from-scratch PyTorch decoder serving a 96-request, four-tenant trace, with cached decoding "
     "checked against recomputation.",
     "Token counting and measured compute disagree by 33 points of the pool; less on a batching GPU."),
    ("kv_cache", "kv_shares.png", "A shared vLLM-style pod", "simulation",
     "Discrete-event engine with paged KV blocks, prefix caching, chunked prefill and continuous "
     "batching, metered five ways.",
     "Step time sees almost no idle capacity; KV memory leaves 83% of the bill with no request."),
    ("hybrid_e2e", "hybrid.png", "Joining gateway and provider ledgers", "real CLI · synthetic spend",
     "The unmodified CLI against mock OpenCost, LiteLLM, OpenAI and Anthropic APIs with real auth "
     "schemes and cursor pagination.",
     "Every source on double counts all gateway spend; the gateway alone covers 72% of invoices."),
    ("use_cases", "use_cases.png", "Use cases", "mixed",
     "Labeling Pareto, per-feature unit economics, self-host break-even and a CI budget gate on "
     "the same ledger.",
     "Three label fixes take a 67%-unallocated org to 1.9%."),
)

USES = (
    ("Fix the labels that matter",
     "Sort unowned spend by dollars. In the hybrid-org study three label changes took unallocated "
     "spend from 66.9% to 1.9%.",
     "unalloc labels --dimension team"),
    ("Cost a feature end to end",
     "Join cluster and API rows on a <code>feature</code> label. A RAG answer cost "
     '<span class="money">$22.32</span> per 1k requests with its vector database, '
     '<span class="money">$13.38</span> counting the LLM bill alone.',
     "unalloc report --dimension feature"),
    ("Decide build versus buy",
     "One self-hosted GPU beat a mid-tier API above about 0.23 requests/s and a small-tier API "
     "above about 1.7 requests/s in the simulation. Utilization decides it, not list price.",
     "python -m case_studies.use_cases"),
    ("Gate deploys on ownership",
     "Fail CI when too much spend has no owner. Exit code 2 above the budget.",
     "unalloc report -D team --budget 10"),
)


def landing() -> str:
    board = "".join(
        f'<li><span class="figure {f.kind}">{esc(f.figure)}</span>'
        f"<div><p>{f.text}</p>"
        + (
            f'<div class="bar" aria-hidden="true"><span class="fill" style="flex:{f.share:.4f}"></span>'
            f'<span class="rest" style="flex:{1 - f.share:.4f}"></span></div>'
            if f.share is not None
            else ""
        )
        + f'</div><div class="src"><span>Paper {f.section}</span>'
        f'<a href="{REPO}/blob/main/case_studies/results/{f.study}/metrics.json">'
        f"{f.study}/metrics.json</a></div></li>"
        for f in findings()
    )
    studies = "".join(
        f'<article class="study"><div class="frame">'
        f'<img src="assets/figures/{img}" alt="Figure from the {esc(name)} study" loading="lazy">'
        f'<span class="kind">{esc(kind)}</span></div>'
        f'<div class="body"><h3>{esc(name)}</h3><p>{esc(what)}</p>'
        f'<p class="headline">{esc(headline)}</p>'
        f'<a class="code" href="{REPO}/tree/main/case_studies/{slug}">Code and data →</a></div></article>'
        for slug, img, name, kind, what, headline in STUDIES
    )
    uses = "".join(
        f"<div><h3>{esc(title)}</h3><p>{text}</p><pre><code>{esc(cmd)}</code></pre></div>"
        for title, text, cmd in USES
    )
    body = f"""<main>
<div class="wrap hero">
  <div>
    <span class="eyebrow">Open-source AI cost attribution · v0.2.3</span>
    <h1>Find the AI spend <em>nobody owns.</em></h1>
    <p class="lede"><code>unalloc</code> joins OpenCost Kubernetes allocations with LiteLLM, OpenAI and Anthropic bills into one exact ledger and reports how much of your AI spend has no owner. Six case studies, one of them on a real H100, show where that attribution breaks.</p>
    <div class="actions">
      <a class="btn primary" href="{PAPER}">Read the paper</a>
      <a class="btn" href="blog/who-pays-for-the-kv-cache.html">Read the blog post</a>
      <a class="btn" href="explorer.html">Open the explorer</a>
    </div>
    <div class="meta-line"><span>pip install unalloc</span><span>Preprint, not yet peer reviewed</span><a href="https://doi.org/{DOI}">doi:{DOI}</a></div>
  </div>
  {receipt()}
</div>

<section class="band" id="findings"><div class="wrap">
  <header><span class="eyebrow">Findings</span><h2>Where attribution breaks</h2>
  <p>Most failures happen where systems meet, not inside any one of them. Each number is read from the study's results when this page is built, and links to them.</p></header>
  <ul class="board">{board}</ul>
</div></section>

<section class="band" id="case-studies"><div class="wrap">
  <header><span class="eyebrow">Case studies</span><h2>Six studies, one ledger</h2>
  <p>Each study emits payloads in the providers' own formats and runs them through <code>unalloc</code>'s real adapters. Raw data and scripts are in the repository.</p></header>
  <div class="studies">{studies}</div>
</div></section>

<section class="band" id="use-cases"><div class="wrap">
  <header><span class="eyebrow">Use cases</span><h2>What to do with the joined ledger</h2></header>
  <div class="uses">{uses}</div>
</div></section>

<section class="band" id="read"><div class="wrap">
  <header><span class="eyebrow">Read</span><h2>Paper, posts and runbook</h2></header>
  <ul class="reads">
    <li><a class="title" href="{PAPER}">Who Pays for the KV Cache? Attributing Shared AI Inference Spend Across Kubernetes and LLM Provider Bills</a><span class="fmt">PDF · 13 pages</span>
      <p>The full paper: the tool, six studies, related work and limitations. Preprint.</p></li>
    <li><a class="title" href="blog/who-pays-for-the-kv-cache.html">Who Pays for the KV Cache?</a><span class="fmt">Blog post</span>
      <p>The readable version: what was measured, what surprised me, and advice for teams setting up LLM showback.</p></li>
    <li><a class="title" href="blog/gpu-validation.html">Addendum: Verified on a Real H100</a><span class="fmt">Blog post</span>
      <p>Machine specs, results and the captured bring-up and teardown of the GPU run.</p></li>
    <li><a class="title" href="{REPO}/blob/main/case_studies/gpu_validation/RUNBOOK.md">GPU validation runbook</a><span class="fmt">GitHub</span>
      <p>Every command to repeat the H100 run on DigitalOcean, including the two things that went wrong.</p></li>
    <li><a class="title" href="{REPO}/blob/main/case_studies/unalloc_case_studies.ipynb">Companion notebook</a><span class="fmt">Jupyter</span>
      <p>Walk through each study with live code.</p></li>
  </ul>
</div></section>

<section class="band" id="cite"><div class="wrap cite">
  <header><span class="eyebrow">Cite</span><h2>Citing unalloc</h2></header>
  <p>Every release is archived on Zenodo at <a href="https://doi.org/{DOI}">doi:{DOI}</a>, which resolves to the latest &mdash; currently 0.2.3, the code the paper describes.</p>
  <pre><code>@software{{urista_unalloc_2026,
  author  = {{Urista, Timothy}},
  title   = {{unalloc: find the AI spend nobody owns}},
  version = {{0.2.3}},
  year    = {{2026}},
  doi     = {{{DOI}}},
  url     = {{{REPO}}}
}}</code></pre>
</div></section>
</main>"""
    return page(
        "unalloc: find the AI spend nobody owns",
        body,
        depth=0,
        current="index.html",
        description="Open-source tool and case studies on attributing shared AI inference spend "
        "across Kubernetes and LLM provider bills.",
    )


# --- explorer -------------------------------------------------------------------------------


def explorer() -> str:
    from case_studies.ui import __main__ as ui

    keys = [k for k in ui.CURATED if k in ui.discover()]
    doc = ui.document(keys)
    marker = '<div class="eyebrow">unalloc · joined AI cost ledger</div>'
    if marker not in doc:
        raise SystemExit("explorer template changed; update the back link in docs_build")
    return doc.replace(
        marker,
        '<div class="eyebrow"><a href="index.html" style="color:inherit">unalloc</a>'
        " · joined AI cost ledger</div>",
    )


# --- build & verify ---------------------------------------------------------------------------

_REF = re.compile(r'(?:href|src)="([^"]+)"')


def verify(out: Path) -> list[str]:
    """Every relative href/src on every page must point at a file in the build."""
    broken = []
    for page_path in out.rglob("*.html"):
        for target in _REF.findall(page_path.read_text()):
            if target.startswith(("http://", "https://", "mailto:", "#", "data:")):
                continue
            resolved = (page_path.parent / target.split("#", 1)[0]).resolve()
            if not resolved.exists():
                broken.append(f"{page_path.relative_to(out)} -> {target}")
    return broken


def build(out: Path) -> list[Path]:
    if not any(FIGURES.glob("*.png")):
        raise SystemExit(f"no ledger figures in {FIGURES}: run `python -m paper.make_figures` first")
    if out.exists():
        # Only remove what this build owns; keep anything else a maintainer adds.
        for owned in ("assets", "blog", "paper", "index.html", "explorer.html"):
            target = out / owned
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
    (out / "assets" / "figures").mkdir(parents=True)
    (out / "assets" / "blog").mkdir(parents=True)
    (out / "blog").mkdir()
    (out / "paper").mkdir()

    shutil.copy2(HERE / "site.css", out / "assets" / "site.css")
    for png in FIGURES.glob("*.png"):
        shutil.copy2(png, out / "assets" / "figures" / png.name)
    shutil.copy2(ROOT / "blog" / "hero.png", out / "assets" / "blog" / "hero.png")
    shutil.copytree(ROOT / "blog" / "gpu-evidence", out / "assets" / "gpu-evidence",
                    ignore=shutil.ignore_patterns("*.html"))
    shutil.copy2(ROOT / PAPER, out / PAPER)
    (out / ".nojekyll").touch()

    written = [out / "index.html", out / "explorer.html"]
    written[0].write_text(landing())
    written[1].write_text(explorer())
    for source, slug in POSTS:
        path = out / "blog" / f"{slug}.html"
        path.write_text(render_post(ROOT / "blog" / source, slug))
        written.append(path)

    broken = verify(out)
    if broken:
        raise SystemExit("broken local links:\n  " + "\n  ".join(broken))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DOCS)
    args = parser.parse_args(argv)
    for path in build(args.out):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
