"""Build and execute the companion notebook.

    python -m case_studies.notebook_exec

The notebook's cells live here as plain strings so they are reviewed and
linted like code; this script assembles them with nbformat, executes them with
nbclient, and writes case_studies/unalloc_case_studies.ipynb with outputs.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
OUT = HERE / "unalloc_case_studies.ipynb"

MD, CODE = "markdown", "code"

CELLS: list[tuple[str, str]] = [
    (MD, """# unalloc case studies

A hands-on companion to `paper/unalloc-case-studies.pdf`. Every number here comes from
unalloc's real adapters and attribution code. The heavier PyTorch studies are loaded
from `case_studies/results/`; the cheap ones run live.

Run everything from scratch with `make case-studies`, then `make notebook`."""),
    (CODE, """import json
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402 - after the sys.path setup above
from matplotlib.ticker import PercentFormatter  # noqa: E402

from case_studies import common  # noqa: E402

TENANT_COLORS = {"search": "#2a78d6", "agents": "#eb6834", "platform": "#1baf7a",
                 "sandbox": "#eda100", "__overhead__": "#c9cfca"}


def load(study):
    return json.loads((common.RESULTS / study / "metrics.json").read_text())"""),
    (MD, """## 1. The joined ledger, and what a fallback really does

The bundled fixtures mix a Kubernetes cluster with three LLM bills. Attributing on `team`
leaves most spend unowned; falling back to `namespace` rescues a lot of it - and unalloc now
reports how much of the "allocated" spend only got there through the fallback."""),
    (CODE, """from unalloc.cli import FIXTURE_DIR
from unalloc.core.attribute import attribute
from unalloc.sources import REGISTRY

rows = [row for name, filename in common.FIXTURE_FILES.items()
        for row in REGISTRY[name]().from_fixture(FIXTURE_DIR / filename)]

for fallback in ((), ("namespace",)):
    report = attribute(rows, "team", fallback_dimensions=fallback)
    print(f"fallback={list(fallback)}: {report.unallocated_pct}% unallocated, "
          f"${report.fallback_usd:,.2f} via fallback")
    for bucket in report.buckets:
        sources = ", ".join(bucket.by_source)
        print(f"    {bucket.key:<16} ${bucket.amount_usd:>11,.2f}  {sources}")"""),
    (MD, """## 2. A shared vLLM pod: the metering rule decides who pays

A vLLM-style engine (paged KV cache, prefix caching, continuous batching) serves four
callers. The same run is split five ways. Token meters spread idle capacity silently;
measured meters expose it as overhead."""),
    (CODE, """from case_studies.kv_cache import attribution as kv
from case_studies.kv_cache.sim import simulate

run = simulate(rate=3.0, duration_s=300)
stats = kv.tenant_stats(run)
split = kv.shares(run, stats)
tenants = [*stats, kv.OVERHEAD]

print(f"{len(run.requests)} requests over {run.wall_s:.0f}s simulated")
print(f"{'':<14}" + "".join(f"{m:>12}" for m in kv.METHODS))
for tenant in tenants:
    print(f"{tenant:<14}" + "".join(f"{split[m][tenant]:>12.1%}" for m in kv.METHODS))

fig, ax = plt.subplots(figsize=(7, 2.6))
for i, method in enumerate(kv.METHODS):
    left = 0.0
    for tenant in tenants:
        ax.barh(i, split[method][tenant], left=left, color=TENANT_COLORS[tenant],
                edgecolor="white", linewidth=1, label=tenant if i == 0 else None)
        left += split[method][tenant]
ax.set_yticks(range(len(kv.METHODS)), kv.METHODS)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(PercentFormatter(1.0))
ax.legend(ncol=5, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False)
plt.show()"""),
    (MD, """## 3. Real inference with a KV cache (PyTorch)

A from-scratch 3.3M-parameter decoder served a multi-tenant trace on CPU. Cached decoding is
verified token-identical to recomputation before anything is timed."""),
    (CODE, """tkv = load("torch_kv")
print("correctness passed:", tkv["correctness"]["passed"])
series = tkv["e1_scaling"]["series"]
steps = zip(series["context"], series["cached_step_s"], series["uncached_step_s"], strict=True)
for n, cached, uncached in steps:
    print(f"context {n:>5}: cached {cached*1e3:6.2f} ms  uncached {uncached*1e3:7.2f} ms  "
          f"({uncached/cached:4.1f}x)")

callers = tkv["attribution"]["callers"]
print(f"\\n{'method':<20}" + "".join(f"{c:>10}" for c in callers))
for method, split in tkv["attribution"]["shares"].items():
    print(f"{method:<20}" + "".join(f"{split[c]:>10.1%}" for c in callers))"""),
    (MD, """## 4. Distributed inference: tensor and pipeline parallel on `torch.distributed`

Correctness first (every parallel config must match the single-process model), then the
cost question: what happens to a LeaderWorkerSet's bill when only leader pods carry `team`."""),
    (CODE, """dist = load("distributed")
for config, result in dist["correctness"].items():
    print(f"{config:<7} tokens match: {result['tokens_match']}  "
          f"max |dlogit| {result['max_abs_logit_diff']:.1e}")
print()
for key in ("s1", "s2", "s3"):
    summary = dist["attribution"]["scenarios"][key]["summary"]
    acc = summary["accuracy"]
    print(f"{key}: {summary['unallocated_pct']:>5}% unallocated, "
          f"{acc['misattributed_pct']:.1f}% 'allocated' to a non-team value  "
          f"({dist['attribution']['scenarios'][key]['title']})")"""),
    (MD, """## 5. End to end: the real CLI against live mock provider APIs

Mock OpenCost, LiteLLM, OpenAI and Anthropic servers enforce each provider's auth scheme and
page their responses. The CLI runs as a subprocess, exactly as in production."""),
    (CODE, """from case_studies.hybrid_e2e import servers, workload
from case_studies.hybrid_e2e.__main__ import cli

data = workload.build()
with servers.serve(data) as stack:
    gateway = cli(stack, "report", "--json", "-s", "opencost", "-s", "litellm", "-D", "team")
    everything = cli(stack, "report", "--json", "-D", "team")

for label, report in (("gateway + cluster", gateway), ("every source on", everything)):
    by_source = {k: round(float(v)) for k, v in report["by_source"].items()}
    print(f"{label:<18} ${float(report['total_usd']):>10,.0f}  "
          f"{report['unallocated_pct']:>5}% unallocated  {by_source}")"""),
    (MD, """## 6. Use cases: labeling Pareto, break-even, and a CI gate"""),
    (CODE, """uc = load("use_cases")
fig, ax = plt.subplots(figsize=(6, 2.4))
for key, color in (("bundled_fixtures", "#2a78d6"), ("hybrid_org", "#eb6834")):
    curve = uc["U1_pareto"][key]["curve_pct"][:21]
    ax.step(range(len(curve)), curve, where="post", color=color, label=key)
ax.axhline(10, color="#4f5b54", linewidth=0.8, linestyle=":")
ax.set_xlabel("label fixes, most expensive first")
ax.set_ylabel("% unallocated")
ax.legend(frameon=False)
plt.show()

for budget in ("50", "80"):
    proc = subprocess.run([sys.executable, "-m", "unalloc.cli", "report", "--fixtures",
                           "--json", "--budget", budget], capture_output=True, text=True)
    print(f"unalloc report --budget {budget} -> exit {proc.returncode}")"""),
    (MD, """## 7. Explore it yourself

```bash
python -m case_studies.ui          # ledger explorer on http://127.0.0.1:8765
```"""),
]


def build() -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    for kind, source in CELLS:
        cell = nbformat.v4.new_markdown_cell if kind == MD else nbformat.v4.new_code_cell
        nb.cells.append(cell(source))
    return nb


def main() -> int:
    nb = build()
    NotebookClient(nb, timeout=900, kernel_name="python3",
                   resources={"metadata": {"path": str(HERE)}}).execute()
    nbformat.write(nb, OUT)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
