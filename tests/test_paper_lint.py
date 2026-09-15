"""The paper build refuses Typst that compiles but renders wrong."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper.build import SOURCE, lint_source  # noqa: E402


def test_published_paper_source_is_clean():
    assert lint_source(SOURCE.read_text()) == []


def test_tilde_is_flagged_in_prose_but_not_in_code():
    assert lint_source("reads only ~25% of spend") == [
        "line 1: `~` renders as a non-breaking space; write ≈ or 'about'"
    ]
    assert lint_source("run `ls ~/.ssh` first") == []
    assert lint_source("```\ncd ~/unalloc\n```") == []


def test_semicolon_after_citation_and_bare_dollar_are_flagged():
    assert lint_source('prefills #c("sarathi"); step latency')
    assert lint_source("costs $5,703 per month")
    assert lint_source("costs \\$5,703 per month") == []
    assert lint_source("where $U > P$ holds") == []
