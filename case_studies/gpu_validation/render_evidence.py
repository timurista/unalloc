"""Render captured terminal evidence into shareable images for the blog addendum.

    python -m case_studies.gpu_validation.render_evidence

Reads the text captured during the run from results/gpu_validation/evidence/ and
writes terminal-style PNG cards to blog/gpu-evidence/ via headless Chrome. The
cards contain only verbatim excerpts of captured output, never retyped text.
"""

from __future__ import annotations

import html
import re
import shutil
import subprocess
from pathlib import Path

from case_studies import common

EVIDENCE = common.RESULTS / "gpu_validation" / "evidence"
OUT = common.ROOT.parent / "blog" / "gpu-evidence"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
MAX_COLS = 132

STYLE = """
<style>
  body { margin: 0; background: #0e1210; }
  .card { width: 1200px; box-sizing: border-box; padding: 26px 30px 30px; background: #151a17;
          border: 1px solid #2b332e; border-radius: 10px; margin: 20px;
          font: 14px/1.45 "IBM Plex Mono", ui-monospace, Menlo, monospace; color: #dfe6e0; }
  .bar { display: flex; gap: 8px; margin-bottom: 16px; align-items: center; }
  .dot { width: 11px; height: 11px; border-radius: 50%; background: #3a433d; }
  .title { margin-left: 10px; color: #8d9a92; font-size: 13px; }
  pre { margin: 0; white-space: pre; overflow: hidden; }
  .cmd { color: #7cc4a4; }
  .note { color: #8d9a92; }
</style>
"""


def excerpt(name: str, keep: list[str], limit: int = 40) -> str:
    """Lines from an evidence file matching any pattern, in original order."""
    lines = (EVIDENCE / name).read_text(errors="replace").splitlines()
    patterns = [re.compile(p) for p in keep]
    chosen = [
        line
        for line in lines
        # Drop `set -x` trace lines; keep nvidia-smi's "+----" table borders.
        if any(p.search(line) for p in patterns) and not line.startswith(("+ ", "++ "))
    ]
    return "\n".join(chosen[:limit])


def colorize(text: str) -> str:
    out = []
    for raw in text.splitlines():
        # Shorten visibly rather than letting the card clip the right edge.
        line = raw if len(raw) <= MAX_COLS else raw[: MAX_COLS - 1] + "…"
        escaped = html.escape(line)
        if line.startswith("$"):
            out.append(f'<span class="cmd">{escaped}</span>')
        elif line.startswith("#"):
            out.append(f'<span class="note">{escaped}</span>')
        else:
            out.append(escaped)
    return "\n".join(out)


def card(title: str, body: str) -> str:
    return (
        f'<div class="card"><div class="bar"><span class="dot"></span><span class="dot"></span>'
        f'<span class="dot"></span><span class="title">{html.escape(title)}</span></div>'
        f"<pre>{colorize(body)}</pre></div>"
    )


def render(name: str, title: str, body: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    page = OUT / f"{name}.html"
    page.write_text(f"<!doctype html><meta charset='utf-8'>{STYLE}{card(title, body)}")
    png = OUT / f"{name}.png"
    rows = body.count("\n") + 1
    height = 110 + rows * 21
    subprocess.run(
        [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
         "--force-device-scale-factor=2", f"--window-size=1242,{height}",
         f"--screenshot={png}", page.as_uri()],
        check=True, capture_output=True, timeout=60,
    )
    page.unlink()
    return png


def main() -> int:
    if not Path(CHROME).exists() and not shutil.which("chromium"):
        raise SystemExit("needs Google Chrome for headless screenshots")
    cards = {
        "01-droplet": (
            "doctl · the droplet that ran the benchmark",
            excerpt("01_create.txt", [r"^\$", r"^ID", r"^\d+ ", r"^# (started|finished)"])
            + "\n\n"
            + excerpt("02_machine.txt", [r"^\$ doctl", r"^ID", r"^\d{6,} "]),
        ),
        "02-gpu": (
            "nvidia-smi on the droplet",
            excerpt(
                "02_machine.txt",
                [r"NVIDIA-SMI", r"^\|.*(GPU|H100|Fan|MiB)", r"^\+-", r"^name,",
                 r"^NVIDIA H100", r"Model name", r"^\s+total", r"^Mem:", r"^Ubuntu 22"],
                limit=30,
            ),
        ),
        "03-vllm": (
            "vLLM serving Qwen2.5-7B-Instruct on the H100",
            excerpt(
                "04_serve.txt",
                [r"^# (Step 4b|Fix|healthy)", r"^\$", r'"version"', r"Model loading took",
                 r"KV cache size", r"Starting vLLM", r"^name,", r"^NVIDIA H100"],
            ),
        ),
        "04-bench": (
            "benchmark sweep (client on the droplet, next to the server)",
            excerpt("05_bench.txt", [r"^#", r"^\$", r"warm-up", r"^rate "]),
        ),
    }
    if (EVIDENCE / "07_teardown.txt").exists():
        cards["05-teardown"] = (
            "teardown · the droplet no longer exists",
            excerpt("07_teardown.txt", [r".*"], limit=30),
        )
    for name, (title, body) in cards.items():
        print(render(name, title, body))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
