#!/usr/bin/env python3
"""Mechanical lint for docs/index.html — catches fever-dream patterns
that accumulate per-milestone.

Per CLAUDE.md "Adversarial Landing-Page Discipline" (2026-05-21):
the landing page is operator-curated copy. This linter catches
mechanical signs of bloat / drift / per-milestone narrative
splicing at commit-time, before they ship to carnot-ebm.org.

Rules (each emits a HARD-FAIL on violation):

1. PER-CARD WORD COUNT
   - bento-card body  > 120 words
   - r-card body      >  60 words
   Rationale: a stranger landing on the page should be able to skim
   each card in ~5 seconds. Anything over the limit is a sign that
   per-milestone updates have been appended without removing the
   earlier ones.

2. FORBIDDEN CONTENT IN ANY CARD BODY
   - Raw experiment IDs in prose (\\bExp\\s+\\d{3,4}\\b)
   - Milestone numbers in prose (\\bMilestone\\s+\\.\\d{2,3}\\b or
     2026.MM.NNN)
   - Flag syntax (foo=True, foo_bar=False, etc.)
   - Internal acronyms appearing 3+ times without a gloss (NupProbe,
     ORCA-NEXUS, NEXUS, FR-11, HardNet++, GRPO v\\d, Tier 0[a-z]).
     Two or fewer is treated as "load-bearing terminology"; three or
     more in a single card is bloat.

3. SECTION CARD COUNTS
   - results-grid (<div class="results-grid">): max 20 r-cards
   - bento-grid (<div class="bento-grid">): max 12 bento-cards
   - Rationale: landing-page sections that grow without removal cost
     stranger-readers a wall of low-value detail; supporting evidence
     lives in docs/technical-report.html, not the front door.

4. FOOTER PARAGRAPHS
   - Any <footer> paragraph > 100 words → bloat

Exit codes: 0 clean, 1 violations found.

Pairs with `scripts/canonical_url_lint.py` and the per-milestone
adversarial AI audit (`scripts/pages_adversarial_audit.py`).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_HTML = PROJECT_ROOT / "docs" / "index.html"

# Word-count thresholds (post-strip of HTML tags)
MAX_BENTO_BODY_WORDS = 120
MAX_RCARD_BODY_WORDS = 60
MAX_FOOTER_PARA_WORDS = 100

# Section-level card-count thresholds
MAX_BENTO_CARDS = 12
MAX_RCARDS_IN_RESULTS_GRID = 20

# Forbidden content patterns (per-card scope)
EXP_ID_RE = re.compile(r"\bExp\s+\d{3,4}\b")
# Milestone-narrative patterns. Bare `.NNN` is too ambiguous with
# decimals (0.525, 0.9857, etc.) — only match explicit "Milestone .NNN"
# form or the full CalVer 2026.MM.NNN form.
MILESTONE_NARRATIVE_RE = re.compile(
    r"\b(?:Milestone\s+\.\d{2,3}|2026\.\d{2}\.\d{2,3})\b"
)
FLAG_SYNTAX_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]+=(True|False|true|false)\b")
INTERNAL_ACRONYMS_RE = re.compile(
    r"\b(NupProbe|ORCA-NEXUS|NEXUS|FR-11 v\d+|HardNet\+\+|GRPO v\d+|"
    r"Tier 0[a-z]|SnareNet|FSNet|EBM-CoT|MARCH|SemEnergy|"
    r"SOS-KAN|EqM Gradient Sampler|DCCD|GBNF|BEAVER-lite|"
    r"NRGPT|SECL|FoVer v\d+|PRM v\d+|ThinkPRM|CRANE|"
    r"DiffuTruth|p[Bb]it|Soft-Gibbs|cikan|CIKAN|HILED|ROCE|"
    r"V-?JEPA|GPT-OSS|Boltzmann-GPT)\b"
)


def strip_tags(html: str) -> str:
    """Return the readable text from an HTML fragment."""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_card_bodies(html: str, card_class: str, body_tag: str) -> list[tuple[str, str, str, int]]:
    """Return [(title, body_text, all_text, line_no), ...] for each card.

    body_text: text inside <p class="{body_tag}"> (used for word-count cap)
    all_text:  ALL text inside the card div (used for forbidden-pattern scan
               — catches Exp NNNN that lives in <div class="r-stats"> etc.)

    card_class: 'bento-card' or 'r-card'
    body_tag:   'bento-text' or 'r-desc'
    """
    # Locate each card div + extract title + body
    out: list[tuple[str, str, str, int]] = []
    pat = re.compile(
        r'<div class="' + card_class + r'[^"]*"[^>]*>(.*?)</div>\s*</div>',
        re.DOTALL,
    )
    # Pre-compute line numbers
    line_offsets = [0]
    for ch in html:
        line_offsets.append(line_offsets[-1] + (1 if ch == "\n" else 0))

    for m in pat.finditer(html):
        body_html = m.group(1)
        title_m = re.search(
            r'<h3 class="(?:bento-title|r-title)">([^<]+)</h3>', body_html
        )
        body_m = re.search(
            r'<p class="' + body_tag + r'">(.*?)</p>', body_html, re.DOTALL
        )
        title = title_m.group(1).strip() if title_m else "(no title)"
        body_text = strip_tags(body_m.group(1)) if body_m else ""
        all_text = strip_tags(body_html)
        line_no = line_offsets[m.start()] + 1
        out.append((title, body_text, all_text, line_no))
    return out


def scan_card_body(body_text: str) -> list[str]:
    """Return list of violation strings found in a card body."""
    violations: list[str] = []
    exp_hits = EXP_ID_RE.findall(body_text)
    if exp_hits:
        violations.append(
            f"raw experiment IDs in prose ({len(exp_hits)}): "
            f"{', '.join(sorted(set(exp_hits))[:5])}"
        )
    ms_hits = MILESTONE_NARRATIVE_RE.findall(body_text)
    # The numeric form `.NNN` also catches version-y decimals that aren't
    # actually milestones. Require at least 2 hits OR explicit "Milestone"
    # to flag, to avoid false positives on e.g. "0.9857" or "v6".
    explicit_milestone = any("Milestone" in h or "2026." in h for h in ms_hits)
    if explicit_milestone or len(ms_hits) >= 3:
        sample = sorted(set(ms_hits))[:5]
        violations.append(
            f"per-milestone narrative ({len(ms_hits)} refs): {', '.join(sample)}"
        )
    flag_hits = FLAG_SYNTAX_RE.findall(body_text)
    if flag_hits:
        violations.append(f"flag syntax in prose ({len(flag_hits)})")
    acr_hits = INTERNAL_ACRONYMS_RE.findall(body_text)
    if len(acr_hits) >= 3:
        counts = Counter(acr_hits)
        top = counts.most_common(5)
        violations.append(
            f"internal acronyms ({len(acr_hits)} hits): "
            f"{', '.join(f'{a}×{n}' for a, n in top)}"
        )
    return violations


def count_section_cards(html: str, section_class: str, card_class: str) -> int:
    """Count cards within a specific section."""
    sec_pat = re.compile(
        r'<div class="' + section_class + r'[^"]*"[^>]*>(.*?)</section>',
        re.DOTALL,
    )
    m = sec_pat.search(html)
    if not m:
        return 0
    return len(re.findall(r'<div class="' + card_class, m.group(1)))


def main() -> int:
    if not INDEX_HTML.exists():
        print(f"docs/index.html not found at {INDEX_HTML}; skipping")
        return 0
    html = INDEX_HTML.read_text()

    violations: list[str] = []

    # Rule 1 & 2: per-card body limits + forbidden content
    for card_class, body_tag, max_words in [
        ("bento-card", "bento-text", MAX_BENTO_BODY_WORDS),
        ("r-card", "r-desc", MAX_RCARD_BODY_WORDS),
    ]:
        cards = find_card_bodies(html, card_class, body_tag)
        for title, body_text, all_text, line_no in cards:
            word_count = len(body_text.split())
            if word_count > max_words:
                violations.append(
                    f"L{line_no} {card_class} '{title}' body={word_count}w > {max_words}w cap"
                )
            # Forbidden-content scan runs against ALL text in the card
            # (including r-stats, r-tag, etc.) — that's where Exp NNNN
            # citations live, not in r-desc.
            forbidden = scan_card_body(all_text)
            for f in forbidden:
                violations.append(f"L{line_no} {card_class} '{title}' — {f}")

    # Rule 3: section card counts
    bento_count = count_section_cards(html, "bento-grid", "bento-card")
    if bento_count > MAX_BENTO_CARDS:
        violations.append(
            f"bento-grid has {bento_count} cards > {MAX_BENTO_CARDS} cap"
        )

    # results-grid card count (only counting r-cards directly inside the
    # results-grid div, not r-cards elsewhere on the page)
    rg_match = re.search(
        r'<div class="results-grid">(.*?)</section>', html, re.DOTALL
    )
    if rg_match:
        rcard_count = len(re.findall(r'<div class="r-card', rg_match.group(1)))
        if rcard_count > MAX_RCARDS_IN_RESULTS_GRID:
            violations.append(
                f"results-grid has {rcard_count} r-cards > "
                f"{MAX_RCARDS_IN_RESULTS_GRID} cap"
            )

    # Rule 4: footer paragraph length
    foot_match = re.search(r"<footer[^>]*>(.*?)</footer>", html, re.DOTALL)
    if foot_match:
        for p_match in re.finditer(r"<p[^>]*>(.*?)</p>", foot_match.group(1), re.DOTALL):
            txt = strip_tags(p_match.group(1))
            wc = len(txt.split())
            if wc > MAX_FOOTER_PARA_WORDS:
                violations.append(
                    f"footer paragraph > {MAX_FOOTER_PARA_WORDS}w ({wc}w): "
                    f"{txt[:80]}..."
                )

    if not violations:
        print(f"pages_fever_dream_lint: clean ({INDEX_HTML.relative_to(PROJECT_ROOT)})")
        return 0

    print(
        f"pages_fever_dream_lint: {len(violations)} violation(s) in "
        f"{INDEX_HTML.relative_to(PROJECT_ROOT)}\n"
        f"Per CLAUDE.md 'Adversarial Landing-Page Discipline':\n"
        f"  - bento-card body cap: {MAX_BENTO_BODY_WORDS}w\n"
        f"  - r-card body cap:     {MAX_RCARD_BODY_WORDS}w\n"
        f"  - bento-grid cap:      {MAX_BENTO_CARDS} cards\n"
        f"  - results-grid cap:    {MAX_RCARDS_IN_RESULTS_GRID} r-cards\n"
        f"  - footer paragraph:    {MAX_FOOTER_PARA_WORDS}w\n"
        f"  - NO raw expNNNN / .NNN / flags= / internal acronyms in card prose\n"
    )
    for v in violations[:30]:
        print(f"  {v}")
    if len(violations) > 30:
        print(f"  ... and {len(violations) - 30} more")
    return 1


if __name__ == "__main__":
    sys.exit(main())
