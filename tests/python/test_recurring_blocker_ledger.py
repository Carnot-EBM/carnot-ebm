"""REQ-OPS-RECURRING-BLOCKER-6263: a blocker that recurs is a task nobody is doing.

Spec: REQ-OPS-RECURRING-BLOCKER-6263 /
SCENARIO-OPS-RECURRING-BLOCKER-6263-PER-TASK-IDS-DO-NOT-HIDE-RECURRENCE

Unattended operation fails differently: a failure nobody reads never gets fixed. Measured
2026-08-13, `blocked_gate_check_failed` fired 31 times across 14 milestones with nothing
escalating it, and 37 of 58 blocked tasks recorded no diagnostic at all.

The normaliser is the load-bearing part. If it under-collapses, every blocker looks like a
singleton and the tool reports nothing; if it over-collapses, unrelated blockers merge and the
count is meaningless. Both directions are pinned here.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from recurring_blocker_ledger import _is_blocked, normalise  # noqa: E402


def test_per_task_experiment_ids_do_not_hide_recurrence() -> None:
    """THE LOAD-BEARING CASE. Without id-stripping every blocker is a singleton."""
    a = normalise("blocked: Exp6262 readiness controls did not all pass")
    b = normalise("blocked: Exp6301 readiness controls did not all pass")
    assert a == b


def test_milestone_versions_do_not_hide_recurrence() -> None:
    a = normalise("blocked: V540 source freeze checks failed")
    b = normalise("blocked: V547 source freeze checks failed")
    assert a == b


def test_counts_inside_a_message_do_not_hide_recurrence() -> None:
    a = normalise("blocked_safety: safety=0.0; workload_value=0.0")
    b = normalise("blocked_safety: safety=0.3; workload_value=1.0")
    assert a == b


def test_genuinely_different_blockers_stay_separate() -> None:
    """The over-collapse direction. A tool that merges unrelated blockers reports a big
    number that means nothing."""
    a = normalise("blocked: one or more recorded validation commands failed or timed out")
    b = normalise("blocked_missing_receipt: no newer dated gatemate physical receipt")
    assert a != b


def test_blocked_detection_covers_the_shapes_this_corpus_uses() -> None:
    for v in (
        "blocked_gate_check_failed",
        "blocked: readiness controls did not all pass",
        "complete: pre_gate block on exp6388",
        "skipped_doomed_rerun",
    ):
        assert _is_blocked(v), v


def test_a_successful_verdict_is_not_blocked() -> None:
    assert not _is_blocked("complete: bounded ASP energy compiler matches clingo oracle")
    assert not _is_blocked("complete_gate_not_met_no_reliable_signal")
