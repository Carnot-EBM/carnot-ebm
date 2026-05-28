"""Tests for scripts/overdue_priority_lint.py.

Origin: 2026-05-27 .298 planner confirmed CLAUDE.md "Overdue-Priority
Forcing Function" was design-time discipline only — no mechanical
enforcement. The Prompt-Injection KAN v4 priority sat at pending_count=5
without pickup across .294-.298. This lint is the Layer 1 mechanical
fix. These tests pin the contract:

  - known-issues.md priority + 0/1/2 completed milestones since filed:
    pass (not overdue yet)
  - priority + 3+ completed milestones + slug present in roadmap-next:
    pass (picked up)
  - priority + 3+ completed milestones + slug absent + no override:
    refuse
  - priority + 3+ completed milestones + slug absent + operator_override
    field present: pass
  - priority without an embedded task spec (no slug): pass (lint can't
    disambiguate, treat as covered)

Spec coverage: CLAUDE.md "Overdue-Priority Forcing Function".
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import pytest


def _load():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "overdue_priority_lint.py"
    spec = importlib.util.spec_from_file_location(
        "overdue_priority_lint", module_path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["overdue_priority_lint"] = mod
    spec.loader.exec_module(mod)
    return mod


_MOD = _load()


# -----------------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------------


class TestParsePriorities:
    """Parse `### NEW YYYY-MM-DD ...` headers into Priority records."""

    def test_extracts_date_and_title(self) -> None:
        text = """## MANDATORY-NEXT-MILESTONE PRIORITIES

### NEW 2026-05-26 (20:50Z): Prompt-Injection EBM Distillation v4

Origin paragraph.

```yaml
- id: exp<next>
  deliverable: "results/experiment_<next>_prompt_injection_kan_distill_v4_15k.json"
```

## Another section
"""
        priorities = _MOD._parse_priorities(text)
        assert len(priorities) == 1
        p = priorities[0]
        assert p.filed_date == date(2026, 5, 26)
        assert "Prompt-Injection" in p.title
        assert p.slug == "prompt_injection_kan_distill_v4_15k"

    def test_returns_empty_when_no_section(self) -> None:
        assert _MOD._parse_priorities("# Random doc with no priorities") == []

    def test_handles_multiple_priorities(self) -> None:
        text = """## MANDATORY-NEXT-MILESTONE PRIORITIES

### NEW 2026-05-26: First Priority

```yaml
deliverable: "results/experiment_<next>_first_priority.json"
```

### NEW 2026-05-27: Second Priority

```yaml
deliverable: "results/experiment_<next>_second_priority.json"
```

## Other heading
"""
        priorities = _MOD._parse_priorities(text)
        assert len(priorities) == 2
        assert priorities[0].slug == "first_priority"
        assert priorities[1].slug == "second_priority"

    def test_priority_without_deliverable_has_no_slug(self) -> None:
        text = """## MANDATORY-NEXT-MILESTONE PRIORITIES

### NEW 2026-05-26: Discussion-only Priority

This entry has no task spec; just a narrative.

## Next section
"""
        priorities = _MOD._parse_priorities(text)
        assert len(priorities) == 1
        assert priorities[0].slug is None

    def test_stops_at_non_M_section_header(self) -> None:
        """The MANDATORY-* section runs until the next `^## [^M]` header."""
        text = """## MANDATORY-NEXT-MILESTONE PRIORITIES

### NEW 2026-05-26: Real priority

```yaml
deliverable: "results/experiment_<next>_real_one.json"
```

## Resolved Issues

### NEW 2026-05-25: Resolved-but-still-NEW-tagged

```yaml
deliverable: "results/experiment_<next>_resolved.json"
```
"""
        priorities = _MOD._parse_priorities(text)
        assert len(priorities) == 1
        assert priorities[0].slug == "real_one"


# -----------------------------------------------------------------------------
# Counting completed milestones
# -----------------------------------------------------------------------------


class TestCountMilestonesCompletedSince:
    def test_counts_only_dates_after_filed(self) -> None:
        text = """- id: '2026.05.290'
  completed: '2026-05-25'
- id: '2026.05.291'
  completed: '2026-05-26'
- id: '2026.05.292'
  completed: '2026-05-27'
"""
        # filed before all three: 25,26,27 are all strictly after 24 -> 3
        assert _MOD._count_milestones_completed_since(text, date(2026, 5, 24)) == 3
        # filed ON 05-26 (strict >): only 05-27 counts; same-day 05-26 excluded -> 1
        assert _MOD._count_milestones_completed_since(text, date(2026, 5, 26)) == 1
        # filed after all three -> 0
        assert _MOD._count_milestones_completed_since(text, date(2026, 5, 28)) == 0

    def test_same_day_filing_not_counted(self) -> None:
        """A priority filed the same day milestones close is NOT yet overdue —
        strict > means same-day completions don't accrue pending_count.
        This is the bug-fix case: the conductor closes ~30 milestones/day,
        so a freshly-filed priority must not inherit same-day churn."""
        text = "\n".join(f"completed: '2026-05-28'" for _ in range(30))
        # filed today (2026-05-28): zero milestones completed on a LATER day
        assert _MOD._count_milestones_completed_since(text, date(2026, 5, 28)) == 0

    def test_empty_research_complete(self) -> None:
        assert _MOD._count_milestones_completed_since("", date(2026, 5, 1)) == 0


# -----------------------------------------------------------------------------
# Slug presence + override detection
# -----------------------------------------------------------------------------


class TestSlugAndOverrideDetection:
    def test_slug_present_returns_true(self) -> None:
        roadmap = """milestone: "2026.05.299"
tasks:
  - id: exp3221-prompt-injection-kan-distill-v4-15k
    deliverable: "results/experiment_3221_prompt_injection_kan_distill_v4_15k.json"
"""
        assert _MOD._slug_present_in_roadmap_next(
            roadmap, "prompt_injection_kan_distill_v4_15k"
        )

    def test_slug_absent_returns_false(self) -> None:
        roadmap = """milestone: "2026.05.299"
tasks:
  - id: exp3221-other-task
    deliverable: "results/experiment_3221_other.json"
"""
        assert not _MOD._slug_present_in_roadmap_next(
            roadmap, "prompt_injection_kan_distill_v4_15k"
        )

    def test_none_slug_treats_as_covered(self) -> None:
        """A priority without an embedded slug can't be disambiguated; treat as covered."""
        assert _MOD._slug_present_in_roadmap_next("milestone: x", None)

    def test_operator_override_detected(self) -> None:
        roadmap = """milestone: "2026.05.299"
operator_override:
  authorization: "operator directive 2026-05-27"
  rationale: "Deferred to .300 in favor of CUDA chain"
tasks:
  - id: exp3221-other-task
"""
        assert _MOD._operator_override_present(roadmap, "any-slug")

    def test_operator_override_absent(self) -> None:
        roadmap = """milestone: "2026.05.299"
tasks:
  - id: exp3221-other-task
"""
        assert not _MOD._operator_override_present(roadmap, "any-slug")


# -----------------------------------------------------------------------------
# Full _check pipeline
# -----------------------------------------------------------------------------


class TestCheck:
    def _make_priority(
        self, filed: date, slug: str | None = "test_slug"
    ) -> "_MOD.Priority":
        return _MOD.Priority(
            filed_date=filed,
            title=f"Test priority filed {filed}",
            slug=slug,
            raw_block="raw markdown",
        )

    def test_clean_when_pending_below_threshold(self) -> None:
        priorities = [self._make_priority(date(2026, 5, 27))]
        research_complete = "completed: '2026-05-27'\ncompleted: '2026-05-28'\n"
        roadmap = "milestone: 2026.05.300\n"
        assert _MOD._check(priorities, research_complete, roadmap) == []

    def test_clean_when_slug_present_in_roadmap(self) -> None:
        priorities = [self._make_priority(date(2026, 5, 20), slug="my_task_slug")]
        # 5 milestones since 2026-05-20: definitely overdue
        research_complete = "\n".join(
            f"completed: '2026-05-{d:02d}'" for d in (21, 22, 23, 24, 25)
        )
        roadmap = """milestone: "2026.05.300"
tasks:
  - id: exp9999-foo
    deliverable: "results/experiment_9999_my_task_slug.json"
"""
        assert _MOD._check(priorities, research_complete, roadmap) == []

    def test_clean_when_operator_override_present(self) -> None:
        priorities = [self._make_priority(date(2026, 5, 20), slug="my_task_slug")]
        research_complete = "\n".join(
            f"completed: '2026-05-{d:02d}'" for d in (21, 22, 23, 24, 25)
        )
        roadmap = """milestone: "2026.05.300"
operator_override:
  authorization: "operator directive 2026-05-27"
tasks:
  - id: exp9999-other-task
"""
        assert _MOD._check(priorities, research_complete, roadmap) == []

    def test_refuses_when_overdue_and_missing(self) -> None:
        priorities = [self._make_priority(date(2026, 5, 20), slug="my_task_slug")]
        research_complete = "\n".join(
            f"completed: '2026-05-{d:02d}'" for d in (21, 22, 23, 24, 25)
        )
        roadmap = """milestone: "2026.05.300"
tasks:
  - id: exp9999-something-else
    deliverable: "results/experiment_9999_unrelated.json"
"""
        violations = _MOD._check(priorities, research_complete, roadmap)
        assert len(violations) == 1
        assert "my_task_slug" in violations[0]

    def test_boundary_at_threshold(self) -> None:
        """Pending_count exactly 3 triggers (rule is >= 3)."""
        priorities = [self._make_priority(date(2026, 5, 20), slug="boundary")]
        # Exactly 3 milestones since
        research_complete = "\n".join(
            f"completed: '2026-05-{d:02d}'" for d in (21, 22, 23)
        )
        roadmap = "milestone: 2026.05.300\n"  # boundary slug absent
        violations = _MOD._check(priorities, research_complete, roadmap)
        assert len(violations) == 1

    def test_just_below_threshold(self) -> None:
        """Pending_count of 2 does not trigger."""
        priorities = [self._make_priority(date(2026, 5, 20), slug="not_yet")]
        # Only 2 milestones since
        research_complete = "completed: '2026-05-21'\ncompleted: '2026-05-22'\n"
        roadmap = "milestone: 2026.05.300\n"
        assert _MOD._check(priorities, research_complete, roadmap) == []

    def test_no_slug_treats_as_covered(self) -> None:
        """Priorities without an embedded task spec can't be enforced
        — treat as covered to avoid false-positives on narrative-only entries."""
        priorities = [self._make_priority(date(2026, 5, 20), slug=None)]
        research_complete = "\n".join(
            f"completed: '2026-05-{d:02d}'" for d in (21, 22, 23, 24, 25)
        )
        roadmap = "milestone: 2026.05.300\n"
        assert _MOD._check(priorities, research_complete, roadmap) == []
