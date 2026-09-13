"""Compile the case-study paper with Typst.

    python -m paper.build        # after `python -m paper.make_figures`

Uses the `typst` Python package, so no TeX installation is needed.
"""

from __future__ import annotations

from pathlib import Path

import typst

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "unalloc.typ"
OUTPUT = HERE / "unalloc-case-studies.pdf"


def main() -> int:
    missing = [p for p in ("kv_shares", "torch_e1", "dist_scenarios", "hybrid", "use_cases")
               if not (HERE / "figures" / f"{p}.svg").exists()]
    if missing:
        raise SystemExit(f"missing figures {missing}: run `python -m paper.make_figures` first")
    typst.compile(str(SOURCE), output=str(OUTPUT), root=str(HERE))
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
