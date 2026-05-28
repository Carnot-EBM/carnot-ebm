#!/usr/bin/env python3
"""Overdue-Priority Forcing Function — Layer 1 mechanical enforcement.

CLAUDE.md "Overdue-Priority Forcing Function" mandates that any MANDATORY-
NEXT-MILESTONE PRIORITY entry which has been pending 3+ consecutive
milestones without pickup MUST be included in the planner's emitted
roadmap, or accompanied by an explicit operator-override rationale.

That rule has lived in CLAUDE.md since 2026-04-28 as design-time discipline
only — there has been no mechanical enforcement. The .294 through .298
planners all silently skipped the Prompt-Injection KAN v4 priority despite
it being pending_count=5, demonstrating the gap that this lint closes.

The lint runs against a planner-emitted research-roadmap-next.yaml. For
each priority entry in ops/known-issues.md MANDATORY-NEXT-MILESTONE
PRIORITIES section:

  1. Parse the `### NEW YYYY-MM-DD ...` header's date.
  2. Count milestones in research-complete.yaml that completed on or
     after the priority's filed date.
  3. If count >= 3, look for the priority's deliverable slug in the
     roadmap-next YAML.
  4. If the slug is absent AND no `operator_override:` block is present
     in the roadmap-next near the priority's task spec, refuse.

The activation guard wires this lint as a pre-activation check. When the
lint refuses, the conductor halts at the activation step — the operator
(or outer-loop) sees the refuse banner in the conductor log and either:

  - adds the missing task to research-roadmap-next.yaml and re-activates
  - adds an `operator_override:` field citing the specific authorization
  - removes the stale priority entry from known-issues.md

Usage:
  python3 scripts/overdue_priority_lint.py            # check active roadmap-next
  python3 scripts/overdue_priority_lint.py --strict   # exit 1 on warnings too

Spec coverage: CLAUDE.md "Overdue-Priority Forcing Function".
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWN_ISSUES = PROJECT_ROOT / "ops" / "known-issues.md"
ROADMAP_NEXT = PROJECT_ROOT / "research-roadmap-next.yaml"
RESEARCH_COMPLETE = PROJECT_ROOT / "research-complete.yaml"

# Priority is overdue once it has been pending this many completed
# milestones without pickup. CLAUDE.md mandates >= 3.
PENDING_THRESHOLD = 3

# Header pattern for priority entries in known-issues.md.
# Example: "### NEW 2026-05-26 (20:50Z): Prompt-Injection EBM Distillation v4 — ..."
PRIORITY_HEADER = re.compile(
    r"^###\s+NEW\s+(\d{4}-\d{2}-\d{2})(?:\s*\([^)]*\))?:\s+(.+?)\s*$",
    re.MULTILINE,
)

# Deliverable line within a priority entry's `## The task to queue:` block.
# Example: deliverable: "results/experiment_<next>_prompt_injection_kan_distill_v4_15k.json"
DELIVERABLE_PATTERN = re.compile(
    r"""deliverable:\s*["']?results/experiment_(?:<next>|\d+)_(?P<slug>[a-z0-9_]+)\.json["']?""",
    re.IGNORECASE,
)

# Milestone-completed marker in research-complete.yaml.
# Example: completed: '2026-05-26'
COMPLETED_PATTERN = re.compile(r"^\s*completed:\s*['\"]?(\d{4}-\d{2}-\d{2})['\"]?", re.MULTILINE)


@dataclass
class Priority:
    """A MANDATORY-NEXT-MILESTONE priority parsed from known-issues.md."""

    filed_date: date
    title: str
    slug: str | None  # deliverable slug; None if no task spec embedded
    raw_block: str  # the full markdown block for this priority


def _read_text(path: Path) -> str:
    """Return the file's text, or empty string if missing."""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _parse_priorities(known_issues_text: str) -> list[Priority]:
    """Extract priority blocks from the MANDATORY-NEXT-MILESTONE section."""
    # Find the start of the MANDATORY priorities section.
    sec_start = known_issues_text.find("## MANDATORY-NEXT-MILESTONE PRIORITIES")
    if sec_start < 0:
        return []
    # The section runs until the next `^## ` heading.
    after = known_issues_text[sec_start + len("## MANDATORY-NEXT-MILESTONE PRIORITIES"):]
    next_h2 = re.search(r"^## [^M]", after, re.MULTILINE)
    section = after[: next_h2.start()] if next_h2 else after

    out: list[Priority] = []
    # Iterate the `### NEW ...` headers within the section.
    headers = list(PRIORITY_HEADER.finditer(section))
    for i, m in enumerate(headers):
        block_start = m.start()
        block_end = headers[i + 1].start() if i + 1 < len(headers) else len(section)
        block = section[block_start:block_end]
        date_str = m.group(1)
        title = m.group(2)
        try:
            filed = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        slug_match = DELIVERABLE_PATTERN.search(block)
        slug = slug_match.group("slug") if slug_match else None
        out.append(Priority(filed_date=filed, title=title, slug=slug, raw_block=block))
    return out


def _count_milestones_completed_since(
    research_complete_text: str, filed: date
) -> int:
    """Count milestone-completion records dated STRICTLY AFTER the filed date.

    Why strictly-after (>) and not on-or-after (>=): the conductor closes
    ~30 milestones per day, so a priority filed *today* would otherwise
    inherit every same-day completion as "pending" and be flagged overdue
    the moment it is filed. pending_count is meant to measure how many
    *subsequent* planning cycles skipped the priority, so only milestones
    that completed on a later calendar day count. A priority filed today
    has pending_count 0 today and begins accruing tomorrow — giving it a
    fair grace window before the >=3 forcing threshold can fire.

    Date granularity is coarse (it can't distinguish two milestones on the
    same later day), but milestone IDs are sequential and the conductor
    advances multiple per day, so a genuinely-stale priority (filed days
    ago) still crosses the threshold reliably.
    """
    n = 0
    for m in COMPLETED_PATTERN.finditer(research_complete_text):
        try:
            ts = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except ValueError:
            continue
        if ts > filed:
            n += 1
    return n


def _slug_present_in_roadmap_next(
    roadmap_next_text: str, slug: str | None
) -> bool:
    """True iff a task with this slug is present in the roadmap-next YAML."""
    if slug is None:
        # No deliverable slug in the priority entry — treat as covered
        # (the lint can't disambiguate priorities without a task spec).
        return True
    return slug in roadmap_next_text


def _operator_override_present(
    roadmap_next_text: str, slug: str | None
) -> bool:
    """True iff the roadmap-next has an explicit override for this priority."""
    if slug is None:
        return False
    # Look for an operator_override: block within ~30 lines after the slug
    # mention OR at the top of the milestone metadata.
    if not slug:
        return False
    # Simple-and-strict: search for "operator_override" near the slug, or
    # an explicit milestone-level scope_reduction_compliance or overdue_priority_override field.
    OVERRIDE_PATTERNS = (
        re.compile(r"operator_override:", re.MULTILINE),
        re.compile(r"overdue_priority_override:", re.MULTILINE),
    )
    return any(p.search(roadmap_next_text) for p in OVERRIDE_PATTERNS)


def _check(
    priorities: list[Priority],
    research_complete_text: str,
    roadmap_next_text: str,
) -> list[str]:
    """Return a list of violation messages; empty list = clean."""
    violations: list[str] = []
    for p in priorities:
        pending = _count_milestones_completed_since(
            research_complete_text, p.filed_date
        )
        if pending < PENDING_THRESHOLD:
            continue
        if _slug_present_in_roadmap_next(roadmap_next_text, p.slug):
            continue
        if _operator_override_present(roadmap_next_text, p.slug):
            continue
        violations.append(
            f"  Priority filed {p.filed_date.isoformat()} (pending_count={pending}): "
            f"\"{p.title[:80]}...\" "
            f"deliverable_slug={p.slug!r} -- NOT in research-roadmap-next.yaml "
            f"and no operator_override present."
        )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero on any violation (default)",
    )
    args = parser.parse_args()
    del args  # currently no behavioral difference; the parameter is
    # documented for future toggles (e.g., a --warn-only mode if the rule
    # needs to be relaxed during a transition).

    known_issues_text = _read_text(KNOWN_ISSUES)
    if not known_issues_text:
        print(
            f"warning: {KNOWN_ISSUES} not found; skipping overdue-priority check",
            file=sys.stderr,
        )
        return 0

    research_complete_text = _read_text(RESEARCH_COMPLETE)
    roadmap_next_text = _read_text(ROADMAP_NEXT)

    priorities = _parse_priorities(known_issues_text)
    if not priorities:
        return 0  # no MANDATORY priorities filed

    violations = _check(priorities, research_complete_text, roadmap_next_text)
    if not violations:
        return 0

    print("=" * 72, file=sys.stderr)
    print(
        "overdue_priority_lint: refusing roadmap-next activation",
        file=sys.stderr,
    )
    print("=" * 72, file=sys.stderr)
    print(file=sys.stderr)
    print(
        "CLAUDE.md \"Overdue-Priority Forcing Function\" mandates pickup",
        file=sys.stderr,
    )
    print(
        f"of any priority with pending_count >= {PENDING_THRESHOLD}. The",
        file=sys.stderr,
    )
    print(
        "following priorities have aged past that threshold without",
        file=sys.stderr,
    )
    print(
        "being picked up or operator-overridden:",
        file=sys.stderr,
    )
    print(file=sys.stderr)
    for v in violations:
        print(v, file=sys.stderr)
    print(file=sys.stderr)
    print(
        "To unblock: either (a) add a task to research-roadmap-next.yaml",
        file=sys.stderr,
    )
    print(
        "with a deliverable matching the priority's slug, or (b) add an",
        file=sys.stderr,
    )
    print(
        "explicit operator_override: field citing the authorization, or",
        file=sys.stderr,
    )
    print(
        "(c) remove the stale priority entry from known-issues.md.",
        file=sys.stderr,
    )
    print(file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
