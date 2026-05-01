"""Tests for failure_ledger_v2 (Issues 1, 5, and dispatch-time manifest).

Spec: REQ-INFRA-070 — failure-ledger v2 fail-count scoping and matcher tightening.

Each test below corresponds to one of the four named requirements in the
exp1104 plan.  They cover the new module's behaviour in isolation; the
conductor wire-in is exercised separately by the existing
test_research_conductor_*.py suite (the new module is opt-in via dynamic
import there, so a plain-import failure here is the strongest possible
regression signal).

Spec: openspec/change-proposals/failure-ledger-v2-and-planner-discipline.md
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from failure_ledger_v2 import (  # noqa: E402
    count_failures_for_task,
    extract_experiment_id,
    is_excluded_by_manifest,
    keywords_overlap,
)


def _log(timestamp: str, title: str, status: str, details: str) -> str:
    """Build one conductor-log line in the canonical 5-pipe format."""
    return f"| {timestamp} | {title[:50]} | {status} | {details} |"


def test_count_by_id_not_title_prefix():
    """Issue 1 fix: a fresh task with the same title as a retired prior task

    must NOT inherit the prior's failure count when the log entries carry
    [id=...] markers identifying the prior as a different experiment.

    Mirrors the .85 incident where exp1096 SemEnergy was retired before its
    first iteration because exp1080 SemEnergy left 4 fails on the same
    title prefix.
    """
    log_lines = [
        # Three failed attempts at the OLD experiment (exp1080)
        _log(
            "2026-04-30 10:00 UTC",
            "SemEnergy Probe v1 — Live Energy Floor",
            "FAIL",
            "x [id=exp1080-semenergy-probe-v1]",
        ),
        _log(
            "2026-04-30 10:30 UTC",
            "SemEnergy Probe v1 — Live Energy Floor",
            "FAIL",
            "y [id=exp1080-semenergy-probe-v1]",
        ),
        _log(
            "2026-04-30 11:00 UTC",
            "SemEnergy Probe v1 — Live Energy Floor",
            "FAIL",
            "z [id=exp1080-semenergy-probe-v1]",
        ),
        # Milestone activation marker — pick_next_task scopes past this in
        # production, but the helper itself is milestone-agnostic; this test
        # passes ALL lines and asserts the per-id bucketing is enough.
        _log("2026-05-01 09:00 UTC", "Milestone 2026.04.85 activated", "OK", "12 tasks queued"),
    ]
    new_task = {
        "id": "exp1096-semenergy-probe-v1",
        "title": "SemEnergy Probe v1 — Live Energy Floor",
    }
    # The new task shares title prefix with the OLD one but has its own id;
    # the legacy title-prefix counter would have returned 3, but the v2
    # counter returns 0 because no log entry carries the new task's id.
    assert count_failures_for_task(new_task, log_lines) == 0


def test_keyword_matcher_requires_two_keywords():
    """Issue 5 fix: titles that share only one substantive token must NOT

    be flagged as a doomed-rerun match.  Single-token "verifier" or
    "adversarial" overlap was the .85 false-positive that blocked
    exp1106 Phase-1a Adversarial Verifier on 18 spurious priors.
    """
    # Single-token overlap on "verifier" only — must NOT match.
    assert not keywords_overlap(
        "Phase 1a Adversarial Verifier Robustness Audit v2",
        "ThinkPRM Step Verifier v4 — Trained on FoVer",
    )
    # Single-token overlap on "adversarial" only — must NOT match.
    assert not keywords_overlap(
        "Phase 1a Adversarial Verifier Robustness Audit v2",
        "Adversarial Distillation Probe v3",
    )
    # Two-token overlap on real scope words — MUST match.
    assert keywords_overlap(
        "SOTA Code Repair v7 — Live HumanEval",
        "Code Repair via GGUF Models v3 (sota)",
    )
    # Two-token overlap on ising + sampler — MUST match.
    assert keywords_overlap(
        "KV260 Ising Sampler v3 — Sequential Hardware",
        "FPGA Ising Sampler Correctness Audit",
    )


def test_manifest_dispatch_time_enforcement_skips_retired(tmp_path):
    """Manifest fix: a task whose extracted experiment_id is listed in

    the YAML exclusion manifest must be reported as excluded at dispatch
    time, regardless of whether the task is also indexed in
    research-complete.yaml or research-roadmap.yaml.
    """
    manifest_yaml = tmp_path / "exclusion_manifest.yaml"
    manifest_yaml.write_text(
        "retired_experiments:\n"
        "  - experiment_id: 906\n"
        "    completed_milestone: '2026.04.79'\n"
        "    reason: 'no-progress 3 milestones'\n"
        "  - experiment_id: 1065\n"
        "    completed_milestone: '2026.04.83'\n"
        "    reason: 'codex backend pause'\n"
    )
    # exp906 — IS retired, MUST be excluded
    excluded, reason = is_excluded_by_manifest(
        {"id": "exp906-foo-bar", "title": "Some retired thing"},
        yaml_manifest_path=manifest_yaml,
    )
    assert excluded, f"expected exp906 excluded, got reason={reason}"
    assert "906" in reason
    # exp1099 — NOT retired, MUST NOT be excluded
    excluded2, _ = is_excluded_by_manifest(
        {"id": "exp1099-rlvr-ssd-integration-v1", "title": "RLVR SSD"},
        yaml_manifest_path=manifest_yaml,
    )
    assert not excluded2
    # Title-only fallback: extract_experiment_id reads "Experiment 906" too
    excluded3, _ = is_excluded_by_manifest(
        {"id": "no-numeric-prefix", "title": "Experiment 906 — Adversarial Cleanup"},
        yaml_manifest_path=manifest_yaml,
    )
    assert excluded3
    # Sanity on the ID extractor itself.
    assert extract_experiment_id({"id": "exp906-foo"}) == 906
    assert extract_experiment_id({"id": "no-prefix", "title": "Exp 906: hi"}) == 906
    assert extract_experiment_id({"id": "no-id", "title": "no number here"}) is None


def test_legacy_log_entries_still_counted_by_prefix_fallback():
    """Backward-compat: log entries written before the [id=...] marker

    schema must still be counted by title-prefix.  This guarantees that
    upgrading the conductor mid-milestone does not lose the failure
    history for tasks that have already accumulated fails under the
    legacy schema.
    """
    log_lines = [
        # Legacy entries (no [id=...] marker)
        _log(
            "2026-04-30 10:00 UTC", "Legacy Title Prefix Task XYZ", "FAIL", "old details no marker"
        ),
        _log("2026-04-30 10:30 UTC", "Legacy Title Prefix Task XYZ", "FAIL", "still no marker"),
        # Modern entry for a different task (must NOT count)
        _log("2026-04-30 11:00 UTC", "Some Other Task", "FAIL", "x [id=exp9999-other-task]"),
        # One more legacy fail for the legacy-titled task
        _log("2026-04-30 11:30 UTC", "Legacy Title Prefix Task XYZ", "GATE_BLOCK", "no marker"),
    ]
    legacy_task = {
        "id": "exp1234-legacy-title-prefix-task-xyz",
        "title": "Legacy Title Prefix Task XYZ",
    }
    # All three legacy lines bucket by title-prefix to the legacy task;
    # the modern line for a different task does NOT.
    assert count_failures_for_task(legacy_task, log_lines) == 3

    # And a SUCCESS in the legacy bucket resets the counter.
    log_lines_with_reset = log_lines + [
        _log("2026-04-30 12:00 UTC", "Legacy Title Prefix Task XYZ", "OK", "passed"),
    ]
    assert count_failures_for_task(legacy_task, log_lines_with_reset) == 0
