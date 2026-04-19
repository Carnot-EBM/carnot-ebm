"""Tests for BatchingHookRunner — pre-commit hook integration.

Spec: REQ-INFRA-052, REQ-INFRA-053,
      SCENARIO-INFRA-060, SCENARIO-INFRA-061
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from carnot.pipeline.batching_audit import BatchingViolation
from carnot.pipeline.batching_hook_runner import BatchingHookRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_script(scripts_dir: Path, name: str, content: str) -> Path:
    """Write a Python script to scripts_dir and return its path."""
    p = scripts_dir / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# SCENARIO-INFRA-060: sequential loop in staged file blocked
# ---------------------------------------------------------------------------

class TestBatchingHookRunnerViolation:
    """Test that a staged file with a sequential loop returns a violation.

    Spec: SCENARIO-INFRA-060, REQ-INFRA-052
    """

    def test_staged_file_with_sequential_loop_returns_violation(self) -> None:
        """BatchingHookRunner returns high-severity violation for staged file with sequential loop."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scripts_dir = Path(tmpdir)
            script = _write_script(
                scripts_dir,
                "experiment_bad.py",
                "for q in questions:\n    result = infer(q)\n",
            )
            runner = BatchingHookRunner(
                scripts_dir=str(scripts_dir),
                staged_files=[str(script)],
            )
            violations = runner.run(raise_on_violation=False)
            assert len(violations) == 1
            assert violations[0].is_high_severity
            assert "experiment_bad.py" in violations[0].script_path

    def test_run_logs_violations_when_raise_on_violation_true(self, caplog: pytest.LogCaptureFixture) -> None:
        """run(raise_on_violation=True) logs ERROR-level messages for each violation.

        Spec: REQ-INFRA-052
        """
        import logging
        with tempfile.TemporaryDirectory() as tmpdir:
            scripts_dir = Path(tmpdir)
            script = _write_script(
                scripts_dir,
                "experiment_bad2.py",
                "for item in questions:\n    process(item)\n",
            )
            runner = BatchingHookRunner(
                scripts_dir=str(scripts_dir),
                staged_files=[str(script)],
            )
            with caplog.at_level(logging.ERROR, logger="carnot.pipeline.batching_hook_runner"):
                violations = runner.run(raise_on_violation=True)
            assert violations
            assert any("BATCHING VIOLATION" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# SCENARIO-INFRA-061: compliant script not blocked
# ---------------------------------------------------------------------------

class TestBatchingHookRunnerCompliant:
    """Test that a compliant script using BatchedInferenceRunner returns no violations.

    Spec: SCENARIO-INFRA-061, REQ-INFRA-052
    """

    def test_compliant_script_returns_no_violations(self) -> None:
        """Staged file using BatchedInferenceRunner has no high-severity violations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scripts_dir = Path(tmpdir)
            script = _write_script(
                scripts_dir,
                "experiment_good.py",
                "from scripts.experiment_template import BatchedInferenceRunner\n"
                "runner = BatchedInferenceRunner(infer, batch_size=8)\n"
                "results = runner.run_batch(questions)\n",
            )
            runner = BatchingHookRunner(
                scripts_dir=str(scripts_dir),
                staged_files=[str(script)],
            )
            violations = runner.run(raise_on_violation=False)
            assert violations == []


# ---------------------------------------------------------------------------
# REQ-INFRA-053: idempotency — no staged files → empty result
# ---------------------------------------------------------------------------

class TestBatchingHookRunnerNoStagedFiles:
    """Test that no staged files returns empty list (idempotency).

    Spec: REQ-INFRA-053
    """

    def test_no_staged_files_returns_empty_list(self) -> None:
        """BatchingHookRunner returns [] when staged_files is empty.

        Spec: REQ-INFRA-053
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            scripts_dir = Path(tmpdir)
            # Even if there are violating scripts on disk, no staged_files → nothing flagged.
            _write_script(
                scripts_dir,
                "experiment_unstaged.py",
                "for q in questions:\n    infer(q)\n",
            )
            runner = BatchingHookRunner(
                scripts_dir=str(scripts_dir),
                staged_files=[],
            )
            violations = runner.run(raise_on_violation=False)
            assert violations == []

    def test_non_staged_file_with_violation_not_reported(self) -> None:
        """Violation in a non-staged file is not reported (idempotency).

        Spec: REQ-INFRA-053
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            scripts_dir = Path(tmpdir)
            _write_script(
                scripts_dir,
                "experiment_old.py",
                "for q in questions:\n    infer(q)\n",
            )
            staged_script = _write_script(
                scripts_dir,
                "experiment_new_clean.py",
                "# clean script — no sequential loops\n",
            )
            runner = BatchingHookRunner(
                scripts_dir=str(scripts_dir),
                staged_files=[str(staged_script)],
            )
            violations = runner.run(raise_on_violation=False)
            assert violations == []


# ---------------------------------------------------------------------------
# filter_new_violations — direct unit tests
# ---------------------------------------------------------------------------

class TestFilterNewViolations:
    """Unit tests for BatchingHookRunner.filter_new_violations.

    Spec: REQ-INFRA-053
    """

    def test_filter_returns_only_staged_violations(self) -> None:
        """filter_new_violations returns only violations in staged_files.

        Spec: REQ-INFRA-053
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            staged_path = str(Path(tmpdir) / "scripts" / "experiment_staged.py")
            unstaged_path = str(Path(tmpdir) / "scripts" / "experiment_old.py")

            violations = [
                BatchingViolation(
                    script_path=staged_path,
                    line_no=5,
                    pattern="for q in questions:",
                    severity="high",
                ),
                BatchingViolation(
                    script_path=unstaged_path,
                    line_no=10,
                    pattern="for sample in samples:",
                    severity="high",
                ),
            ]

            runner = BatchingHookRunner(
                scripts_dir=str(Path(tmpdir) / "scripts"),
                staged_files=[staged_path],
            )
            result = runner.filter_new_violations(violations)
            assert len(result) == 1
            assert result[0].script_path == staged_path

    def test_filter_excludes_medium_severity(self) -> None:
        """filter_new_violations excludes medium-severity violations.

        Medium severity means BatchedInferenceRunner is already present — partial
        migration.  Only high-severity (no runner at all) blocks the commit.

        Spec: REQ-INFRA-052
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            staged_path = str(Path(tmpdir) / "experiment_partial.py")
            violations = [
                BatchingViolation(
                    script_path=staged_path,
                    line_no=3,
                    pattern="for q in questions:",
                    severity="medium",
                ),
            ]
            runner = BatchingHookRunner(
                scripts_dir=str(Path(tmpdir)),
                staged_files=[staged_path],
            )
            result = runner.filter_new_violations(violations)
            assert result == []

    def test_filter_none_violations_runs_audit(self) -> None:
        """filter_new_violations(None) runs audit internally.

        Spec: REQ-INFRA-052
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            scripts_dir = Path(tmpdir)
            script = _write_script(
                scripts_dir,
                "exp_auto.py",
                "for q in questions:\n    infer(q)\n",
            )
            runner = BatchingHookRunner(
                scripts_dir=str(scripts_dir),
                staged_files=[str(script)],
            )
            result = runner.filter_new_violations(None)
            assert len(result) == 1
            assert result[0].is_high_severity

    def test_filter_empty_violations_returns_empty(self) -> None:
        """filter_new_violations([]) returns [].

        Spec: REQ-INFRA-053
        """
        runner = BatchingHookRunner(scripts_dir="/tmp", staged_files=["/tmp/foo.py"])
        result = runner.filter_new_violations([])
        assert result == []

    def test_nonexistent_scripts_dir_returns_empty(self) -> None:
        """BatchingHookRunner with nonexistent scripts_dir returns no violations.

        Spec: REQ-INFRA-052
        """
        runner = BatchingHookRunner(
            scripts_dir="/nonexistent/path/scripts",
            staged_files=["/nonexistent/path/scripts/exp.py"],
        )
        violations = runner.run(raise_on_violation=False)
        assert violations == []
