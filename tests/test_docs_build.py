"""The Pages site builds from the repository with every local link resolving."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("markdown")

from docs_build.__main__ import _rewrite, build


def test_site_builds_with_no_broken_links(tmp_path):
    written = build(tmp_path / "docs")
    names = {p.relative_to(tmp_path / "docs").as_posix() for p in written}
    assert {"index.html", "explorer.html", "blog/who-pays-for-the-kv-cache.html",
            "blog/gpu-validation.html"} <= names
    assert (tmp_path / "docs" / "paper" / "unalloc-case-studies.pdf").stat().st_size > 100_000
    landing = (tmp_path / "docs" / "index.html").read_text()
    assert "paper/unalloc-case-studies.pdf" in landing
    post = (tmp_path / "docs" / "blog" / "who-pays-for-the-kv-cache.html").read_text()
    assert 'href="../explorer.html"' in post
    assert ".md\"" not in post


def test_unknown_blog_link_fails_the_build():
    assert _rewrite("../paper/figures/kv_shares.png") == "../assets/figures/kv_shares.png"
    with pytest.raises(SystemExit):
        _rewrite("notes/draft.md")
