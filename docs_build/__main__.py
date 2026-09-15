"""Build the GitHub Pages site into docs/.

    python -m docs_build            # or: make pages

Every page is generated from files already in the repository, so the site
cannot drift from them:

* index.html                     landing page: findings, case studies, use cases
* blog/<post>.html               rendered from blog/*.md
* explorer.html                  the ledger explorer (case_studies.ui)
* paper/unalloc-case-studies.pdf the paper PDF, copied as built
* assets/                        stylesheet, figures and evidence images

The build fails if any local link or image on a generated page does not
resolve, so a renamed figure or post cannot ship a broken site.
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DOCS = ROOT / "docs"

REPO = "https://github.com/timurista/unalloc"
DOI = "10.5281/zenodo.22761013"
FONTS = (
    "https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@87.5,600;87.5,700"
    "&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap"
)
PAPER = "paper/unalloc-case-studies.pdf"

POSTS = (
    # (source, output slug, nav label)
    ("who-pays-for-the-kv-cache.md", "who-pays-for-the-kv-cache", "Blog post"),
    ("addendum-gpu-validation.md", "gpu-validation", "GPU addendum"),
)

NAV = (
    ("index.html", "Overview"),
    (PAPER, "Paper (PDF)"),
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
<title>{html.escape(title)}</title>
<meta name="description" content="{html.escape(description)}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{FONTS}">
<link rel="stylesheet" href="{up}assets/site.css">
</head>
<body>
<header class="top"><div class="wrap">
  <a class="brand" href="{up}index.html">unalloc</a>
  <nav aria-label="Site">{links}</nav>
</div></header>
{body}
<footer class="site"><div class="wrap">
  <span>Apache-2.0</span>
  <a href="{REPO}">{REPO.removeprefix("https://")}</a>
  <a href="https://doi.org/{DOI}">doi:{DOI}</a>
  <span>Generated from the repository by <code>make pages</code></span>
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
    for source, slug, _ in POSTS:
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
    body = markdown.markdown(text, extensions=["fenced_code", "tables", "sane_lists"])
    body = re.sub(r"<table>", '<div class="table-scroll"><table>', body)
    body = body.replace("</table>", "</table></div>")
    article = (
        f'<main><article class="post">'
        f'<a class="crumb" href="../index.html">← unalloc overview</a>{body}</article></main>'
    )
    description = re.sub(r"[*_`\[\]]|\([^)]*\)", "", lead.group(1)) if lead else title
    return page(title, article, depth=1, current=f"blog/{slug}.html", description=description)


# --- landing page ------------------------------------------------------------------------

FINDINGS = (
    ("66%", "of a distributed deployment's GPU bill had no owner when <code>team</code> was set only "
     "on LeaderWorkerSet leader pods.", "Distributed inference, paper §6"),
    ("61%", "of that bill landed on a Helm chart name after the obvious fallback label, while the "
     "headline unallocated share fell to 4%.", "Distributed inference, paper §6"),
    ("12–14 pts", "difference in a retrieval-heavy tenant's share between a token meter and an equal "
     "time-share meter, on an H100 running vLLM, at every load tested.", "GPU validation, paper §9"),
    ("97–99%", "GPU utilization from 2 to 16 requests/s while throughput rose 7×. Power draw "
     "(469 W → 660 W) tracked load; utilization did not.", "GPU validation, paper §9"),
    ("$11,815", "of gateway spend counted twice when the gateway and the provider bills were both "
     "enabled, out of a $41K month.", "End-to-end run, paper §7"),
    ("25%", "of true provider spend reported by billing adapters that read only the first page of "
     "results. Fixed in 0.2.0.", "End-to-end run, paper §7"),
)

STUDIES = (
    ("gpu_validation", "gpu_validation.png", "Real vLLM on an H100",
     "vLLM 0.29.0 serving Qwen2.5-7B-Instruct on a DigitalOcean H100, driven with four tenants at "
     "2–16 requests/s. 6,241 requests, 0 errors, about $2.15.",
     "Token and time-share meters disagree by 12–14 points on the RAG tenant; utilization reads "
     "97–99% at every load.", "real serving"),
    ("distributed", "dist_scenarios.png", "Distributed inference",
     "Tensor- and pipeline-parallel inference on torch.distributed, verified token-identical to a "
     "single process, then a month of LeaderWorkerSet allocations.",
     "Leader-only labels leave 66% unowned; falling back to name sends 61% to a chart name.",
     "real inference"),
    ("torch_kv", "torch_shares.png", "A transformer with a real KV cache",
     "A from-scratch PyTorch decoder serving a 96-request, four-tenant trace on CPU, with cached "
     "decoding checked against recomputation.",
     "Token counting and measured compute disagree by 33 points of the pool; smaller on a "
     "batching GPU.", "real inference, CPU"),
    ("kv_cache", "kv_shares.png", "A shared vLLM-style pod",
     "Discrete-event engine with paged KV blocks, prefix caching, chunked prefill and continuous "
     "batching, metered five ways.",
     "Step time sees almost no idle capacity; KV memory leaves 83% of the bill with no request.",
     "simulation"),
    ("hybrid_e2e", "hybrid.png", "Joining gateway and provider ledgers",
     "The unmodified CLI against mock OpenCost, LiteLLM, OpenAI and Anthropic APIs with real auth "
     "schemes and cursor pagination.",
     "Every source on double counts all gateway spend; the gateway alone covers 72% of invoices.",
     "real CLI, synthetic spend"),
    ("use_cases", "use_cases.png", "Use cases",
     "Labeling Pareto, per-feature unit economics, self-host break-even and a CI budget gate on "
     "the same ledger.",
     "Three label fixes take a 67%-unallocated org to 1.9%.", "mixed"),
)

USES = (
    ("Fix the labels that matter",
     "Sort unlabeled spend by dollars. In the hybrid-org study three label changes took unallocated "
     "spend from 66.9% to 1.9%.",
     "unalloc labels --dimension team"),
    ("Cost a feature end to end",
     "Join cluster and API rows on a <code>feature</code> label. A RAG answer cost $22.32 per 1k "
     "requests with its vector database, $13.38 counting the LLM bill alone.",
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
    findings = "".join(
        f'<li><span class="figure num">{html.escape(fig)}</span>'
        f"<p>{text}<span class=\"where\">{html.escape(where)}</span></p></li>"
        for fig, text, where in FINDINGS
    )
    esc = html.escape
    studies = "".join(
        f'<article class="study">'
        f'<img src="assets/figures/{img}" alt="Figure from the {esc(name)} study" loading="lazy">'
        f'<div class="body"><span class="kind">{esc(kind)}</span><h3>{esc(name)}</h3>'
        f'<p>{esc(what)}</p><p class="headline">{esc(headline)}</p>'
        f'<a class="code" href="{REPO}/tree/main/case_studies/{slug}">'
        f"Code and data for {esc(name.lower())}</a>"
        f"</div></article>"
        for slug, img, name, what, headline, kind in STUDIES
    )
    uses = "".join(
        f"<div><h3>{esc(title)}</h3><p>{text}</p><pre><code>{esc(cmd)}</code></pre></div>"
        for title, text, cmd in USES
    )
    body = f"""<main>
<div class="wrap hero">
  <span class="eyebrow">Open-source AI cost attribution · v0.2.1</span>
  <h1>Find the AI spend nobody owns.</h1>
  <p class="lede"><code>unalloc</code> joins OpenCost Kubernetes allocations with LiteLLM, OpenAI and Anthropic bills into one exact ledger and reports how much of your AI spend has no owner. Five case studies and a run on a real H100 show where that attribution breaks.</p>
  <div class="actions">
    <a class="btn primary" href="{PAPER}">Read the paper (PDF)</a>
    <a class="btn" href="blog/who-pays-for-the-kv-cache.html">Read the blog post</a>
    <a class="btn" href="explorer.html">Open the ledger explorer</a>
  </div>
  <div class="meta-line"><code>pip install unalloc</code><span>Paper is a preprint, not yet peer reviewed</span><a href="https://doi.org/{DOI}">doi:{DOI}</a></div>
</div>

<section class="band" id="findings"><div class="wrap">
  <header><span class="eyebrow">Findings</span><h2>Where attribution breaks</h2>
  <p>Most failures happen where systems meet, not inside any one of them. Every number links back to a study in the paper.</p></header>
  <ul class="findings">{findings}</ul>
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
  <p>Release 0.2.1 is archived on Zenodo at <a href="https://doi.org/{DOI}">doi:{DOI}</a>, the code the paper describes.</p>
  <pre><code>@software{{urista_unalloc_2026,
  author  = {{Urista, Timothy}},
  title   = {{unalloc: find the AI spend nobody owns}},
  version = {{0.2.1}},
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
    for png in (ROOT / "paper" / "figures").glob("*.png"):
        shutil.copy2(png, out / "assets" / "figures" / png.name)
    shutil.copy2(ROOT / "blog" / "hero.png", out / "assets" / "blog" / "hero.png")
    shutil.copytree(ROOT / "blog" / "gpu-evidence", out / "assets" / "gpu-evidence",
                    ignore=shutil.ignore_patterns("*.html"))
    shutil.copy2(ROOT / PAPER, out / PAPER)
    (out / ".nojekyll").touch()

    written = [out / "index.html", out / "explorer.html"]
    written[0].write_text(landing())
    written[1].write_text(explorer())
    for source, slug, _ in POSTS:
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
