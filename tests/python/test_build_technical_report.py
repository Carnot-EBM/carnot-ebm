"""Regression test for scripts/build_technical_report.py regenerate_html.

Origin: 2026-05-29. The <article> replacement regex was `(<article>)` (bare),
but the live template uses `<article class="markdown-body">`. The bare pattern
matched 0 blocks, so the script raised "Expected exactly one <article> block,
found 0" and refused to regenerate — leaving the .html to drift from the .md
and forcing hand-edits. Fix: match `<article\b[^>]*>` and preserve the opening
tag verbatim. This test pins that an attributed <article> tag is handled.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load():
    root = Path(__file__).resolve().parents[2]
    p = root / "scripts" / "build_technical_report.py"
    spec = importlib.util.spec_from_file_location("build_technical_report", p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules["build_technical_report"] = m
    spec.loader.exec_module(m)
    return m


_M = _load()


# A complete minimal template the script accepts: <title>, <meta description>,
# and one <article> block. The script also strictly requires the title + meta
# tags (each via its own count=1 re.subn), so fixtures must include them.
def _template(article_open: str) -> str:
    return (
        "<!doctype html><html><head>"
        "<title>Technical Report - stale</title>"
        '<meta name="description" content="stale">'
        "</head><body>\n"
        f"{article_open}\nOLD STALE BODY\n</article>\n"
        "</body></html>"
    )


def test_regenerate_handles_attributed_article_tag(tmp_path):
    """An <article class="..."> opening tag must be matched and preserved
    (the bug: the bare `(<article>)` regex matched 0 attributed tags)."""
    md = tmp_path / "report.md"
    md.write_text("# Title\n\n## Subtitle Heading\n\nFresh body paragraph.\n")
    html = tmp_path / "report.html"
    html.write_text(_template('<article class="markdown-body">'))
    out = _M.regenerate_html(md_path=md, html_path=html)
    assert '<article class="markdown-body">' in out  # opening tag preserved
    assert "OLD STALE BODY" not in out               # old body replaced
    assert "Fresh body paragraph" in out             # fresh body rendered
    assert out.count("<article") == 1 and out.count("</article>") == 1


def test_bare_article_tag_still_works(tmp_path):
    """A bare <article> (no attributes) must still match — back-compat."""
    md = tmp_path / "report.md"
    md.write_text("## Sub\n\nbody\n")
    html = tmp_path / "report.html"
    html.write_text(_template("<article>"))
    out = _M.regenerate_html(md_path=md, html_path=html)
    assert "OLD STALE BODY" not in out and "body" in out
    assert out.count("<article") == 1


def test_missing_article_still_raises(tmp_path):
    """If there is genuinely no <article> block, the build must still refuse
    (the safety behavior is correct; only the attribute-blindness was the bug)."""
    import pytest

    md = tmp_path / "report.md"
    md.write_text("## Sub\n\nbody\n")
    html = tmp_path / "report.html"
    html.write_text(
        "<!doctype html><html><head><title>T</title>"
        '<meta name="description" content="x"></head>'
        "<body><div>no article here</div></body></html>"
    )
    with pytest.raises(RuntimeError, match="article"):
        _M.regenerate_html(md_path=md, html_path=html)
