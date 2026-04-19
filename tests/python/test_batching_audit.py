"""Tests for BatchingEnforcementAudit and BatchingViolation.

Spec: REQ-INFRA-047, REQ-INFRA-048,
      SCENARIO-INFRA-055, SCENARIO-INFRA-056
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from carnot.pipeline.batching_audit import BatchingEnforcementAudit, BatchingViolation


# ---------------------------------------------------------------------------
# BatchingViolation tests
# ---------------------------------------------------------------------------


def test_batching_violation_is_high_severity_true():
    """REQ-INFRA-047: is_high_severity returns True when severity='high'."""
    v = BatchingViolation(
        script_path="scripts/exp_123.py",
        line_no=42,
        pattern="for q in questions:",
        severity="high",
    )
    assert v.is_high_severity is True


def test_batching_violation_is_high_severity_false_medium():
    """REQ-INFRA-047: is_high_severity returns False when severity='medium'."""
    v = BatchingViolation(
        script_path="scripts/exp_456.py",
        line_no=10,
        pattern="for q in questions:",
        severity="medium",
    )
    assert v.is_high_severity is False


def test_batching_violation_is_high_severity_false_low():
    """REQ-INFRA-047: is_high_severity returns False when severity is not 'high'."""
    v = BatchingViolation(
        script_path="scripts/smoke.py",
        line_no=5,
        pattern="for q in questions:",
        severity="low",
    )
    assert v.is_high_severity is False


def test_batching_violation_dataclass_fields():
    """BatchingViolation stores all fields correctly."""
    v = BatchingViolation(
        script_path="scripts/exp_789.py",
        line_no=99,
        pattern="for sample in samples:",
        severity="high",
    )
    assert v.script_path == "scripts/exp_789.py"
    assert v.line_no == 99
    assert v.pattern == "for sample in samples:"
    assert v.severity == "high"


# ---------------------------------------------------------------------------
# BatchingEnforcementAudit.recommended_batch_size tests
# ---------------------------------------------------------------------------


def test_recommended_batch_size_gsm8k(tmp_path):
    """REQ-INFRA-048 / SCENARIO-INFRA-056: gsm8k -> 8."""
    audit = BatchingEnforcementAudit(str(tmp_path))
    assert audit.recommended_batch_size("gsm8k") == 8


def test_recommended_batch_size_humaneval(tmp_path):
    """REQ-INFRA-048 / SCENARIO-INFRA-056: humaneval -> 4."""
    audit = BatchingEnforcementAudit(str(tmp_path))
    assert audit.recommended_batch_size("humaneval") == 4


def test_recommended_batch_size_default(tmp_path):
    """REQ-INFRA-048 / SCENARIO-INFRA-056: unknown task type -> 8 (default)."""
    audit = BatchingEnforcementAudit(str(tmp_path))
    assert audit.recommended_batch_size("unknown_task") == 8


def test_recommended_batch_size_case_insensitive(tmp_path):
    """REQ-INFRA-048: task_type lookup is case-insensitive."""
    audit = BatchingEnforcementAudit(str(tmp_path))
    assert audit.recommended_batch_size("GSM8K") == 8
    assert audit.recommended_batch_size("HumanEval") == 4


# ---------------------------------------------------------------------------
# BatchingEnforcementAudit.scan tests
# ---------------------------------------------------------------------------


def test_scan_detects_sequential_loop_without_batched_runner(tmp_path):
    """REQ-INFRA-047 / SCENARIO-INFRA-055: sequential loop without BatchedInferenceRunner -> violation."""
    script = tmp_path / "exp_seq.py"
    script.write_text(
        textwrap.dedent("""\
            questions = load_questions(100)
            for q in questions:
                result = infer(q)
        """)
    )
    audit = BatchingEnforcementAudit(str(tmp_path))
    violations = audit.scan()
    assert len(violations) >= 1
    assert violations[0].is_high_severity is True
    assert "questions" in violations[0].pattern


def test_scan_no_violation_when_batched_runner_present(tmp_path):
    """REQ-INFRA-047: file with BatchedInferenceRunner is compliant — no violation."""
    script = tmp_path / "exp_batched.py"
    script.write_text(
        textwrap.dedent("""\
            from scripts.experiment_template import BatchedInferenceRunner
            questions = load_questions(100)
            runner = BatchedInferenceRunner(infer)
            results = runner.run_batch(questions)
        """)
    )
    audit = BatchingEnforcementAudit(str(tmp_path))
    violations = audit.scan()
    assert violations == []


def test_scan_returns_medium_severity_when_batched_runner_present_but_loop_exists(tmp_path):
    """REQ-INFRA-047: BatchedInferenceRunner present but sequential loop also exists -> 'medium'."""
    script = tmp_path / "exp_partial.py"
    script.write_text(
        textwrap.dedent("""\
            from scripts.experiment_template import BatchedInferenceRunner
            # Some questions are still sequential for debugging
            for q in questions:
                debug_infer(q)
        """)
    )
    audit = BatchingEnforcementAudit(str(tmp_path))
    violations = audit.scan()
    assert len(violations) >= 1
    assert violations[0].severity == "medium"
    assert violations[0].is_high_severity is False


def test_scan_empty_directory(tmp_path):
    """scan() returns empty list when scripts_dir has no Python files."""
    audit = BatchingEnforcementAudit(str(tmp_path))
    assert audit.scan() == []


def test_scan_nonexistent_directory():
    """scan() returns empty list when scripts_dir does not exist."""
    audit = BatchingEnforcementAudit("/nonexistent/path/that/does/not/exist")
    assert audit.scan() == []


def test_scan_detects_samples_loop(tmp_path):
    """REQ-INFRA-047: 'for x in samples:' is also detected as a violation."""
    script = tmp_path / "exp_samples.py"
    script.write_text(
        textwrap.dedent("""\
            samples = load_samples(50)
            for s in samples:
                infer(s)
        """)
    )
    audit = BatchingEnforcementAudit(str(tmp_path))
    violations = audit.scan()
    assert len(violations) >= 1


def test_scan_detects_prompts_loop(tmp_path):
    """REQ-INFRA-047: 'for p in prompts:' is detected as a violation."""
    script = tmp_path / "exp_prompts.py"
    script.write_text(
        textwrap.dedent("""\
            prompts = build_prompts(questions)
            for p in prompts:
                output = model(p)
        """)
    )
    audit = BatchingEnforcementAudit(str(tmp_path))
    violations = audit.scan()
    assert len(violations) >= 1


def test_scan_reports_correct_line_number(tmp_path):
    """REQ-INFRA-047: violation line_no matches the actual for-loop line."""
    script = tmp_path / "exp_lineno.py"
    script.write_text(
        textwrap.dedent("""\
            import os
            import sys

            questions = load_questions()
            for q in questions:
                result = infer(q)
        """)
    )
    audit = BatchingEnforcementAudit(str(tmp_path))
    violations = audit.scan()
    assert len(violations) >= 1
    # The for loop is on line 5
    assert violations[0].line_no == 5


def test_scan_multiple_scripts(tmp_path):
    """REQ-INFRA-047: violations are collected across multiple scripts."""
    for i in range(3):
        s = tmp_path / f"exp_{i:03d}.py"
        s.write_text(f"for q in questions:\n    infer(q)\n")
    audit = BatchingEnforcementAudit(str(tmp_path))
    violations = audit.scan()
    # At least one violation per script
    assert len(violations) >= 3
