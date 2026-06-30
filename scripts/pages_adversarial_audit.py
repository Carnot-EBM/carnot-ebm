#!/usr/bin/env python3
"""Periodic adversarial AI audit of docs/index.html.

Layer 2 of the "Adversarial Landing-Page Discipline" defense
(complements `scripts/pages_fever_dream_lint.py`). Where the lint
catches mechanical bloat patterns deterministically at commit-time,
this audit runs an LLM as a HOSTILE STRANGER-REVIEWER who asks:

> "I have 30 seconds to skim carnot-ebm.org. If you'd bounce off it,
>  say why."

The audit produces a structured markdown report at
`ops/docs_audit_report.md`. It does NOT edit the landing page —
operator decides what to act on per CLAUDE.md "Public Documentation
Discipline" (the landing page is operator-curated).

Usage:
    python scripts/pages_adversarial_audit.py [--model MODEL]

The audit invokes the gemini-cli by default (the conductor's current
default per CLAUDE.md "Gemini-Default for Experiments"). Override via
--model claude-sonnet-4-6 / codex / opencode etc., but the choice is
incidental — the prompt is the adversarial layer, not the model.

Designed to be called from the conductor's milestone-close path so
the operator gets a fresh review every milestone without manual
trigger. Independent of the conductor's `_update_docs_before_planning`
docs-mutating sub-agent — see CLAUDE.md "Adversarial Landing-Page
Discipline" for why independence matters.
"""

from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_HTML = PROJECT_ROOT / "docs" / "index.html"
REPORT_PATH = PROJECT_ROOT / "ops" / "docs_audit_report.md"

# The adversarial prompt. Designed to be hostile and stranger-focused.
# Independence-of-context is critical — the auditing agent must NOT
# know about the project's internal jargon, milestone-numbering, or
# the conductor's debug logs. The prompt limits scope to docs/index.html
# only.
ADVERSARIAL_PROMPT = """\
You are a hostile adversarial reviewer of the Carnot project's landing
page at carnot-ebm.org. Your audience: a stranger who heard about
Carnot at a conference and has 30 seconds to decide whether to read
further or close the tab.

READ ONLY this file: docs/index.html. Do NOT read CLAUDE.md, ops/, or
any other project context. Your job is to be the stranger, not to
defend the project.

Find PROBLEMS. Be hostile. If you'd bounce off the page, say why.

Check for (each is a hard problem):

1. BLOAT — any section that is too long for its role on a landing page.
   Bento card bodies > 120 words, result cards > 60 words, footer
   paragraphs > 100 words. Even short cards add up: if a section has
   more than ~20 cards a stranger will skim none of them.

2. INTERNAL JARGON — text that only makes sense to project insiders:
   - Raw experiment IDs like "Exp 1688", "exp2435"
   - Milestone numbers like ".148", "2026.05.247", "Milestone .94"
   - Flag syntax like "foo=True", "auroc_lift=−0.008"
   - Acronyms used without explanation (NupProbe, ORCA-NEXUS, FR-11,
     Tier 0X, GRPO, ThinkPRM, etc.)
   - "X/Y criteria met" closeout-style cards

3. PER-MILESTONE NARRATIVE — text that reads like a copy-pasted commit
   message or retrospective: "Milestone .120 archived sampler
   security falsifications, candidate warm-start validation, ..." is
   internal status reporting, not landing-page copy.

4. INCONSISTENCY — claims that contradict each other (different AUROC
   numbers in different cards, acronyms defined inconsistently, ship
   status claims that don't match).

5. MISSING ESSENTIALS for a stranger:
   - What does Carnot DO in one sentence?
   - Why should I trust the numbers?
   - How do I install it?
   - What's the license?
   - Who maintains it?

6. FABRICATION SIGNALS — does any number look suspiciously perfect
   (1.0 AUROC, 0.0 FPR on small samples) without a credibility anchor?

Output format — exactly this structure:

```
# docs_audit_report — {date}

## TL;DR (stranger's 30-second take)
<1-2 sentences: would a stranger keep reading or close the tab? Why?>

## TOP 3 PROBLEMS
1. <one-line problem + section/card title>
2. <...>
3. <...>

## DETAILED FINDINGS
### Bloat
- <card/section> — <word count or card count> — <suggested cap>

### Internal jargon
- <card/section> — <specific tokens> — <why a stranger doesn't know
  what this means>

### Per-milestone narrative
- <card/section> — <specific milestone refs>

### Inconsistencies
- <claim A> vs <claim B>

### Missing essentials
- <what's missing>

### Fabrication signals
- <suspect number + where>

## WHAT'S WORKING
- <one or two things the page does well>

## RECOMMENDED OPERATOR ACTIONS
- <ordered list of specific edits the operator should consider>
```

Do NOT edit docs/index.html. Do NOT propose autonomous changes — the
operator owns this page per CLAUDE.md "Public Documentation
Discipline". Your output is advisory.
"""


def call_gemini(prompt: str, model: str = "gemini-3.1-pro-preview") -> tuple[bool, str]:
    """Invoke gemini-cli with the audit prompt. Returns (ok, output)."""
    try:
        full = (
            f"{prompt}\n\n"
            f"---\n"
            f"docs/index.html CONTENT:\n\n"
            f"{INDEX_HTML.read_text()}\n"
        )
        proc = subprocess.run(
            ["gemini", "--model", model, "--yolo", "-p", full],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
            cwd=PROJECT_ROOT,
        )
        if proc.returncode != 0:
            return False, f"gemini exit {proc.returncode}: {proc.stderr[:200]}"
        return True, proc.stdout
    except Exception as exc:
        return False, str(exc)


def call_claude(prompt: str, model: str = "claude-opus-4-8") -> tuple[bool, str]:
    """Invoke claude CLI with the audit prompt (Opus 4.8 + max effort per 2026-06-08 directive)."""
    try:
        full = (
            f"{prompt}\n\n"
            f"---\n"
            f"docs/index.html CONTENT:\n\n"
            f"{INDEX_HTML.read_text()}\n"
        )
        proc = subprocess.run(
            ["claude", "--model", model, "--effort", "max", "--print", full],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
            cwd=PROJECT_ROOT,
        )
        if proc.returncode != 0:
            return False, f"claude exit {proc.returncode}: {proc.stderr[:200]}"
        return True, proc.stdout
    except Exception as exc:
        return False, str(exc)


def call_codex(prompt: str, model: str = "gpt-5.5") -> tuple[bool, str]:
    """Codex (gpt-5.5) hostile-stranger reviewer — quota-conserve path (mirrors the conductor's
    codex exec pattern; prompt on stdin via `-`). Added 2026-06-30 for the Claude-quota window."""
    try:
        full = (
            f"{prompt}\n\n"
            f"---\n"
            f"docs/index.html CONTENT:\n\n"
            f"{INDEX_HTML.read_text()}\n"
        )
        proc = subprocess.run(
            ["codex", "exec", "--dangerously-bypass-approvals-and-sandbox", "--color", "never",
             "--model", model, "--cd", str(PROJECT_ROOT), "--ephemeral", "-"],
            input=full,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
            cwd=PROJECT_ROOT,
        )
        if proc.returncode != 0:
            return False, f"codex exit {proc.returncode}: {proc.stderr[:200]}"
        return True, proc.stdout
    except Exception as exc:
        return False, str(exc)


def write_report(content: str) -> None:
    """Write the audit output to ops/docs_audit_report.md."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    header = (
        f"<!-- generated by scripts/pages_adversarial_audit.py — {today} -->\n"
        f"<!-- per CLAUDE.md 'Adversarial Landing-Page Discipline' — "
        f"this report is advisory; operator decides what to act on -->\n\n"
    )
    REPORT_PATH.write_text(header + content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="claude",  # 2026-06-10 operator directive: gemini is NEVER the default (global-stall incident); claude=Opus is the audit agent per the 2026-06-08 directive. codex added 2026-06-30 for the Claude-quota-conserve window.
        choices=["gemini", "claude", "codex"],
        help="Which CLI backend to use (default: claude=Opus; codex for the quota-conserve window)",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Specific model name (e.g. gemini-3.1-pro-preview, sonnet)",
    )
    args = parser.parse_args()

    if not INDEX_HTML.exists():
        print(f"docs/index.html not found at {INDEX_HTML}")
        return 1

    if args.model == "gemini":
        model_name = args.model_name or "gemini-3.1-pro-preview"
        ok, output = call_gemini(ADVERSARIAL_PROMPT, model=model_name)
    elif args.model == "codex":
        model_name = args.model_name or "gpt-5.5"
        ok, output = call_codex(ADVERSARIAL_PROMPT, model=model_name)
    else:
        model_name = args.model_name or "sonnet"
        ok, output = call_claude(ADVERSARIAL_PROMPT, model=model_name)

    if not ok:
        print(f"adversarial audit failed: {output}", file=sys.stderr)
        return 1

    # Strip any code-fence wrapping the agent might have emitted
    report = output.strip()
    # If the agent wrapped its report in ```markdown ... ``` fences, unwrap
    fence_m = re.match(r"^```(?:markdown|md)?\s*\n(.*?)\n```\s*$", report, re.DOTALL)
    if fence_m:
        report = fence_m.group(1)

    write_report(report)
    print(f"adversarial audit complete — report at {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  model: {args.model} ({model_name})")
    print(f"  report length: {len(report):,} chars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
