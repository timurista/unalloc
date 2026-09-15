"""Compile the case-study paper with Typst.

    python -m paper.build        # after `python -m paper.make_figures`

Uses the `typst` Python package, so no TeX installation is needed. Before
compiling, the source is checked for Typst constructs that render *silently*
wrong - the compiler accepts them, and the mistake only shows up in the PDF.
"""

from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "unalloc.typ"
OUTPUT = HERE / "unalloc-case-studies.pdf"

FIGURES = ("kv_shares", "torch_e1", "dist_scenarios", "hybrid", "use_cases", "gpu_validation")

_RAW_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_RAW_INLINE = re.compile(r"`[^`\n]*`")
_COMMENT = re.compile(r"//[^\n]*")

# (pattern, message). Each is legal Typst that does not mean what it looks like.
_CHECKS = (
    (re.compile(r"~"), "`~` renders as a non-breaking space; write ≈ or 'about'"),
    (
        re.compile(r"#c\([^)]*\);"),
        "`;` right after `#c(...)` ends the call and is dropped from the text",
    ),
    # A money amount like $5,703 opens math mode. Real math that starts with a
    # number ($1.07 times 10^(-6)$) is followed by an operator, so it is exempt.
    (
        re.compile(r"(?<![\\\w])\$\d[\d,]*+(?:\.\d++)?+(?!\s*(?:times|dot|[<>=^_+*/-]))"),
        "a dollar amount opens math mode; escape it as `\\$`",
    ),
)


def _mask(text: str) -> str:
    """Blank out raw code and comments, keeping line numbers intact."""

    def blank(match: re.Match[str]) -> str:
        return re.sub(r"[^\n]", " ", match.group(0))

    for pattern in (_RAW_BLOCK, _RAW_INLINE, _COMMENT):
        text = pattern.sub(blank, text)
    return text


def lint_source(text: str) -> list[str]:
    """Problems as `line N: message` strings; empty when the source is clean."""
    problems = []
    for number, line in enumerate(_mask(text).splitlines(), start=1):
        for pattern, message in _CHECKS:
            if pattern.search(line):
                problems.append(f"line {number}: {message}")
    return problems


def main() -> int:
    problems = lint_source(SOURCE.read_text())
    if problems:
        raise SystemExit("paper source has silent rendering problems:\n  " + "\n  ".join(problems))
    missing = [name for name in FIGURES if not (HERE / "figures" / f"{name}.svg").exists()]
    if missing:
        raise SystemExit(f"missing figures {missing}: run `python -m paper.make_figures` first")

    import typst  # research extra only; the lint above must run without it

    typst.compile(str(SOURCE), output=str(OUTPUT), root=str(HERE))
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
