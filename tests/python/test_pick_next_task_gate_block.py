"""Tests for pick_next_task's GATE_BLOCK handling.

Spec: REQ-INFRA-085, SCENARIO-INFRA-085-A through SCENARIO-INFRA-085-C

Background: on 2026-04-29 (.81 milestone) exp1044 (Triple Integration
v7) gate-blocked indefinitely on exp1039's retired-without-artifact
state. The GATE_BLOCK status was logged but NOT counted toward
MAX_FAILURES_PER_TASK, so pick_next_task kept picking exp1044 every
iteration for ~30 minutes, blocking exp1045+ from being picked.

Fix: GATE_BLOCK now counts toward fail_counts. After 3 consecutive
gate-blocks, MAX_FAILURES retires the task, unblocking the cascade.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


def _parse_fail_counts(log_text: str) -> dict[str, int]:
    """Mirror the failure-counting logic from pick_next_task for unit testing."""
    fail_counts: dict[str, int] = {}
    completed_titles: set[str] = set()

    for line in log_text.splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        title = parts[2].strip()
        status = parts[3].strip()

        if status == "OK":
            completed_titles.add(title)
            fail_counts[title] = 0
        elif status in ("FAIL", "REVERT", "SKIP", "NOOP", "GATE_BLOCK"):
            fail_counts[title] = fail_counts.get(title, 0) + 1
    return fail_counts


def test_gate_block_counts_as_failure():
    """REQ-INFRA-085 / SCENARIO-INFRA-085-A: A single GATE_BLOCK increments fail_count.

    Pre-fix behavior: GATE_BLOCK was silently ignored, fail_count stayed at 0.
    """
    log = "| 2026-04-29 20:03 UTC | Triple Integration | GATE_BLOCK | 1 of 1 gate(s) failed"
    counts = _parse_fail_counts(log)
    assert counts.get("Triple Integration") == 1


def test_three_gate_blocks_hit_max_failures():
    """REQ-INFRA-085 / SCENARIO-INFRA-085-B: 3 consecutive GATE_BLOCKs reach
    MAX_FAILURES_PER_TASK threshold, allowing the task to be retired.

    This is the operational unblock for the .81 exp1044 wedge observed
    2026-04-29 — without the fix, GATE_BLOCK looped indefinitely.
    """
    log = "\n".join(
        [
            "| 2026-04-29 20:03 UTC | Triple Integration | GATE_BLOCK | 1 of 1 gate(s) failed",
            "| 2026-04-29 20:14 UTC | Triple Integration | GATE_BLOCK | 1 of 1 gate(s) failed",
            "| 2026-04-29 20:25 UTC | Triple Integration | GATE_BLOCK | 1 of 1 gate(s) failed",
        ]
    )
    counts = _parse_fail_counts(log)
    assert counts.get("Triple Integration") == 3
    # MAX_FAILURES_PER_TASK = 3 (per scripts/research_conductor.py:839)
    # so 3 GATE_BLOCKs hit the threshold and the task gets retired
    # on next pick_next_task iteration.


def test_gate_block_mixed_with_other_failures():
    """REQ-INFRA-085 / SCENARIO-INFRA-085-C: GATE_BLOCK and SKIP combine
    naturally — both increment the same counter.

    Real-world scenario from .81 exp1042: 1 FAIL + 2 SKIP = 3 failures
    → retired. Same shape if mixed FAIL/SKIP/GATE_BLOCK.
    """
    log = "\n".join(
        [
            "| 2026-04-29 18:13 UTC | DualGPU ROCm | FAIL | Reached max turns",
            "| 2026-04-29 18:45 UTC | DualGPU ROCm | SKIP | Pre-tests failing",
            "| 2026-04-29 19:37 UTC | DualGPU ROCm | GATE_BLOCK | gate failed",
        ]
    )
    counts = _parse_fail_counts(log)
    assert counts.get("DualGPU ROCm") == 3


def test_ok_resets_fail_count_after_gate_block():
    """OK after GATE_BLOCKs resets the fail counter (e.g., upstream fix landed
    and the gate now passes)."""
    log = "\n".join(
        [
            "| 2026-04-29 20:03 UTC | Triple Integration | GATE_BLOCK | gate failed",
            "| 2026-04-29 20:14 UTC | Triple Integration | GATE_BLOCK | gate failed",
            "| 2026-04-29 20:30 UTC | Triple Integration | OK | 300 passed",
        ]
    )
    counts = _parse_fail_counts(log)
    assert counts.get("Triple Integration") == 0


def test_pick_next_task_source_includes_gate_block_in_failure_set():
    """Regression guard: scan pick_next_task source to ensure GATE_BLOCK
    is in the failure-counting set. The pre-2026-04-29 bug was that
    GATE_BLOCK was silently ignored."""
    source = (SCRIPTS_DIR / "research_conductor.py").read_text()
    # The relevant tuple should include GATE_BLOCK. The needle below matched
    # the pre-2026-04-30 tuple; DOOMED_RERUN_BLOCK was appended later and
    # this assertion silently went RED for months (found 2026-08-29 during
    # the gate-cascade fix). Keep it aligned with the live tuple.
    assert (
        '"GATE_BLOCK"' in source
        and 'status in ("FAIL", "REVERT", "SKIP", "NOOP", "GATE_BLOCK", "DOOMED_RERUN_BLOCK")'
        in source
    ), (
        "pick_next_task must count GATE_BLOCK toward fail_counts. "
        "The pre-2026-04-29 bug let exp1044 loop indefinitely."
    )
