#!/usr/bin/env python3
"""Regenerate ``docs/technical-report.html`` from ``docs/technical-report.md``.

**Why this exists:** the technical report is the primary public-facing
research document. It is authored in Markdown because the conductor's
doc-reconciliation agent knows how to update Markdown and because Markdown
diffs are readable in git. Github Pages serves the HTML rendering alongside
the Markdown source, and those two files drift apart in practice whenever
only one is edited -- which is almost always, because humans edit Markdown
and only occasionally hand-patch the HTML. At the time this script was
added, the HTML was stuck at "473 experiments across 22 milestones" while
the Markdown read "500+ experiments across 25 milestones" -- three full
milestones of drift.

**What this script does:** it uses the existing HTML as a self-template --
only the content between ``<article>`` and ``</article>`` is replaced with
a fresh Markdown render. The ``<head>`` (fonts, CSS, meta tags), the
``<nav>`` bar, and the ``<footer>`` are all preserved verbatim. The
``<h2>`` subtitle inside ``<article>`` and the ``<meta name="description">``
in the head are rewritten so the user-visible page metadata stays in sync
with the Markdown's subtitle line.

**Why self-template (not Jinja, not pandoc):** this is one file. Jinja
would force a template extraction pass plus a new build dependency. Pandoc
would require a system package not present in the venv. Both are overkill
for a single HTML that changes layout maybe once a year. Self-templating
keeps the script under 200 lines and keeps all styling decisions in the
single HTML file that humans already maintain.

**Usage:**

    # regenerate in place
    python scripts/build_technical_report.py

    # CI / pre-commit: fail if HTML is out of sync (non-zero exit)
    python scripts/build_technical_report.py --check

    # dry-run: print the generated HTML to stdout without writing
    python scripts/build_technical_report.py --dry-run

**Extensions used by the Markdown renderer:** ``tables`` (the headline
results tables render as GFM-style tables), ``fenced_code`` (code blocks
use triple-backticks), ``toc`` (section anchors), ``sane_lists``,
``attr_list``. Add more extensions here if the Markdown starts using a
feature python-markdown does not handle by default.
"""

from __future__ import annotations

import argparse
import html as html_stdlib
import re
import sys
from pathlib import Path

import markdown

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MD_PATH = PROJECT_ROOT / "docs" / "technical-report.md"
HTML_PATH = PROJECT_ROOT / "docs" / "technical-report.html"

# The Markdown extensions we want enabled. Ordered so output is stable.
MARKDOWN_EXTENSIONS = (
    "tables",
    "fenced_code",
    "toc",
    "sane_lists",
    "attr_list",
    "md_in_html",
)


def render_markdown_to_body(md_text: str) -> str:
    """Render the Markdown body to HTML string.

    The output is *just* the article body -- no ``<html>``, ``<head>``, or
    ``<body>`` wrapper. That is handled by the self-template assembly step.
    """
    md = markdown.Markdown(extensions=list(MARKDOWN_EXTENSIONS), output_format="html5")
    return md.convert(md_text)


def extract_subtitle(md_text: str) -> str:
    """Pull the first ``## ...`` line from the Markdown as the page subtitle.

    The technical report's very first ``##`` after ``# Carnot:`` is used as
    the HTML ``<h2>`` subtitle and the ``<meta name="description">`` stem.
    If the Markdown layout ever changes, adjust this function -- it is the
    single source of truth for the page subtitle in the generated HTML.
    """
    for line in md_text.splitlines():
        if line.startswith("## "):
            return line[3:].strip()
    return "Technical Report"


def regenerate_html(md_path: Path = MD_PATH,
                    html_path: Path = HTML_PATH) -> str:
    """Build the new HTML content and return it as a string.

    Does not write to disk -- callers decide whether to write or compare.
    """
    md_text = md_path.read_text()
    current_html = html_path.read_text()

    body_html = render_markdown_to_body(md_text)
    subtitle = extract_subtitle(md_text)

    # 1. Replace the content between <article>...</article> with the freshly
    #    rendered body.  Use a non-greedy regex so nested <article>-like
    #    substrings (unlikely but cheap insurance) do not cause over-match.
    new_html, n_replacements = re.subn(
        r"(<article>)(.*?)(</article>)",
        lambda m: m.group(1) + "\n" + body_html + "\n" + m.group(3),
        current_html,
        count=1,
        flags=re.DOTALL,
    )
    if n_replacements != 1:
        raise RuntimeError(
            f"Expected exactly one <article>...</article> block in {html_path}, "
            f"found {n_replacements}. Regenerate the template or adjust the "
            "script -- refusing to silently produce malformed HTML."
        )

    # 2. Resync the <meta name="description"> stem so search engines and
    #    social embeds show the current headline, not stale milestone counts.
    #    The description text is rebuilt to include the subtitle verbatim --
    #    we intentionally do not try to summarise the whole report here;
    #    that is what the body is for.
    escaped_subtitle = html_stdlib.escape(subtitle, quote=True)
    meta_description = (
        f'<meta name="description" content="Carnot technical report: '
        f'{escaped_subtitle}. Live GPU benchmarks with statistical confidence '
        f'intervals, honest negative results, and a public experiment history '
        f'with provenance labels for every headline claim.">'
    )
    new_html, n_meta = re.subn(
        r'<meta name="description" content="[^"]*">',
        # re.subn feeds match to a lambda so that backslash/sub tokens in
        # meta_description do not get interpreted by re.sub's replacement
        # syntax.
        lambda _m: meta_description,
        new_html,
        count=1,
    )
    if n_meta != 1:
        raise RuntimeError(
            f"Expected exactly one <meta name=\"description\"> tag in "
            f"{html_path}, found {n_meta}. Template is malformed."
        )

    # 3. Rewrite the <title> tag so browser tabs and bookmarks also reflect
    #    the current subtitle instead of a stale one.
    new_title = f"<title>Technical Report - {escaped_subtitle}</title>"
    new_html, n_title = re.subn(
        r"<title>[^<]*</title>", lambda _m: new_title, new_html, count=1,
    )
    if n_title != 1:
        raise RuntimeError(
            f"Expected exactly one <title> tag in {html_path}, found {n_title}."
        )

    return new_html


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Exit non-zero if the HTML on disk does not match "
                             "what regenerate_html() would produce. Useful for "
                             "pre-commit and CI.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the regenerated HTML to stdout instead of "
                             "writing docs/technical-report.html.")
    args = parser.parse_args()

    new_html = regenerate_html()

    if args.dry_run:
        sys.stdout.write(new_html)
        return 0

    if args.check:
        current = HTML_PATH.read_text()
        if current != new_html:
            sys.stderr.write(
                "docs/technical-report.html is out of date relative to "
                "docs/technical-report.md. Run scripts/build_technical_report.py "
                "to regenerate.\n"
            )
            return 1
        return 0

    HTML_PATH.write_text(new_html)
    print(f"Wrote {HTML_PATH} ({len(new_html):,} bytes) from {MD_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
