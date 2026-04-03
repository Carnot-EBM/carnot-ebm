"""Tests for the automatic rollback mechanism.

Spec coverage: REQ-AUTO-007, SCENARIO-AUTO-003
"""

from unittest.mock import MagicMock, patch

from carnot.autoresearch.experiment_log import ExperimentLog
from carnot.autoresearch.rollback import (
    RollbackConfig,
    RollbackResult,
    git_get_head_commit,
    git_revert_commit,
    monitor_and_rollback,
)


class TestGitOperations:
    """Tests for REQ-AUTO-007: git operations."""

    @patch("carnot.autoresearch.rollback.subprocess.run")
    def test_get_head_commit(self, mock_run: MagicMock) -> None:
        """REQ-AUTO-007: gets current HEAD commit hash."""
        mock_run.return_value = MagicMock(stdout="abc123def\n")
        commit = git_get_head_commit()
        assert commit == "abc123def"

    @patch("carnot.autoresearch.rollback.subprocess.run")
    def test_revert_success(self, mock_run: MagicMock) -> None:
        """REQ-AUTO-007: successful git revert returns True."""
        mock_run.return_value = MagicMock(returncode=0)
        assert git_revert_commit("abc123") is True

    @patch("carnot.autoresearch.rollback.subprocess.run")
    def test_revert_failure(self, mock_run: MagicMock) -> None:
        """REQ-AUTO-007: failed git revert returns False."""
        mock_run.return_value = MagicMock(returncode=1)
        assert git_revert_commit("abc123") is False


class TestMonitorAndRollback:
    """Tests for REQ-AUTO-007, SCENARIO-AUTO-003: monitoring and rollback."""

    @patch("carnot.autoresearch.rollback.time.sleep")
    def test_stable_deployment(self, mock_sleep: MagicMock) -> None:
        """REQ-AUTO-007: stable metrics pass monitoring window."""
        # Energy stays at baseline — no regression
        measure_fn = MagicMock(return_value=-5.0)
        # Use very short window so the test completes quickly.
        # Also tests that config=None uses defaults by not passing it
        # in a separate test — here we pass explicit config for speed.
        config = RollbackConfig(
            regression_threshold=0.05,
            monitoring_window_seconds=0.01,  # very short for testing
            poll_interval_seconds=0.001,
        )

        result = monitor_and_rollback(
            hypothesis_id="test-001",
            deployed_commit="abc123",
            baseline_energy=-5.0,
            measure_fn=measure_fn,
            config=config,
        )

        assert result.stable
        assert not result.rolled_back
        assert "without regression" in result.reason

    @patch("carnot.autoresearch.rollback.git_revert_commit", return_value=True)
    @patch("carnot.autoresearch.rollback.time.sleep")
    def test_regression_triggers_rollback(
        self, mock_sleep: MagicMock, mock_revert: MagicMock
    ) -> None:
        """SCENARIO-AUTO-003: energy regression triggers automatic rollback."""
        # Energy degrades significantly
        measure_fn = MagicMock(return_value=-2.0)  # much worse than -5.0
        config = RollbackConfig(
            regression_threshold=0.05,
            monitoring_window_seconds=10,
            poll_interval_seconds=0.001,
        )

        result = monitor_and_rollback(
            hypothesis_id="test-001",
            deployed_commit="abc123",
            baseline_energy=-5.0,
            measure_fn=measure_fn,
            config=config,
        )

        assert not result.stable
        assert result.rolled_back
        assert result.reverted_commit == "abc123"
        assert result.baseline_energy == -5.0
        assert result.final_energy == -2.0
        assert result.regression_amount is not None
        assert result.regression_amount > 0
        mock_revert.assert_called_once_with("abc123", ".")

    @patch("carnot.autoresearch.rollback.git_revert_commit", return_value=False)
    @patch("carnot.autoresearch.rollback.time.sleep")
    def test_failed_revert(
        self, mock_sleep: MagicMock, mock_revert: MagicMock
    ) -> None:
        """REQ-AUTO-007: handles failed git revert gracefully."""
        measure_fn = MagicMock(return_value=-2.0)
        config = RollbackConfig(
            regression_threshold=0.05,
            monitoring_window_seconds=10,
            poll_interval_seconds=0.001,
        )

        result = monitor_and_rollback(
            hypothesis_id="test-001",
            deployed_commit="abc123",
            baseline_energy=-5.0,
            measure_fn=measure_fn,
            config=config,
        )

        assert not result.stable
        assert not result.rolled_back  # revert failed
        assert result.reverted_commit is None

    @patch("carnot.autoresearch.rollback.git_revert_commit", return_value=True)
    @patch("carnot.autoresearch.rollback.time.sleep")
    def test_regression_logged(
        self, mock_sleep: MagicMock, mock_revert: MagicMock
    ) -> None:
        """REQ-AUTO-007: regression is logged to experiment log."""
        log = ExperimentLog()
        measure_fn = MagicMock(return_value=-2.0)
        config = RollbackConfig(
            regression_threshold=0.05,
            monitoring_window_seconds=10,
            poll_interval_seconds=0.001,
        )

        monitor_and_rollback(
            hypothesis_id="test-001",
            deployed_commit="abc123",
            baseline_energy=-5.0,
            measure_fn=measure_fn,
            config=config,
            experiment_log=log,
        )

        assert len(log) == 1
        entry = log.entries[0]
        assert entry.outcome == "rejected"
        assert "regression" in entry.eval_reason.lower()
        assert entry.id == "rollback-test-001"

    @patch("carnot.autoresearch.rollback.time.sleep")
    def test_default_config(self, mock_sleep: MagicMock) -> None:
        """REQ-AUTO-007: works with config=None (uses defaults)."""
        # Trigger an immediate regression so the loop exits quickly
        # rather than running for 3600s with the default window.
        measure_fn = MagicMock(return_value=999.0)  # way above baseline

        with patch("carnot.autoresearch.rollback.git_revert_commit", return_value=True):
            result = monitor_and_rollback(
                hypothesis_id="test",
                deployed_commit="abc",
                baseline_energy=-5.0,
                measure_fn=measure_fn,
                # config=None exercises the default path
            )

        assert not result.stable
        assert result.rolled_back

    @patch("carnot.autoresearch.rollback.time.sleep")
    def test_within_threshold_stable(self, mock_sleep: MagicMock) -> None:
        """REQ-AUTO-007: small energy increase within threshold is stable."""
        # Energy is slightly worse but within 5% threshold
        # baseline=-5.0, threshold=5%, max_allowed=-5.0 + 0.25 = -4.75
        measure_fn = MagicMock(return_value=-4.8)  # within threshold
        config = RollbackConfig(
            regression_threshold=0.05,
            monitoring_window_seconds=0.01,
            poll_interval_seconds=0.001,
        )

        result = monitor_and_rollback(
            hypothesis_id="test-001",
            deployed_commit="abc123",
            baseline_energy=-5.0,
            measure_fn=measure_fn,
            config=config,
        )

        assert result.stable
        assert not result.rolled_back
