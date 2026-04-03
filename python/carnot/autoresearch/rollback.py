"""Automatic rollback mechanism for production autoresearch deployments.

**Researcher summary:**
    Monitors production energy metrics after deploying an accepted hypothesis.
    If metrics degrade beyond a threshold within a monitoring window, automatically
    reverts to the last known-good state via git. Logs regression details and
    adds the hypothesis to the rejected registry.

**Detailed explanation for engineers:**
    After the autoresearch loop accepts a hypothesis and it's transpiled to Rust
    and deployed, we need to verify it actually works in production. This module
    provides the monitoring and rollback infrastructure:

    **The monitoring loop:**
    1. Record the baseline energy metrics before deployment
    2. Deploy the new code (merge the hypothesis branch)
    3. Periodically measure production energy metrics
    4. If energy degrades > threshold for > monitoring_window: ROLLBACK
    5. If monitoring window passes without degradation: mark as STABLE

    **What "rollback" means:**
    - Revert the git commit that introduced the hypothesis
    - Restore the previous baseline record
    - Log the regression with full metrics
    - Add the hypothesis to the rejected registry so it's never re-proposed

    **Why git-based rollback?**
    The production code is in a git repo. Reverting a commit is atomic, fast
    (~1 second), and leaves a clear audit trail. No need for blue-green
    deployments or container orchestration for the initial implementation.

Spec: REQ-AUTO-007
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from carnot.autoresearch.baselines import BaselineRecord
from carnot.autoresearch.experiment_log import ExperimentEntry, ExperimentLog

logger = logging.getLogger(__name__)


@dataclass
class RollbackConfig:
    """Configuration for the production monitoring and rollback system.

    **Researcher summary:**
        Regression threshold, monitoring window, poll interval, and
        git working directory for revert operations.

    **Detailed explanation for engineers:**

    Attributes:
        regression_threshold: Maximum allowed energy increase (as a fraction).
            Default 0.05 (5%). If production energy increases by more than
            this relative to the baseline, rollback is triggered.
            Uses absolute comparison for correct handling of negative energies.
        monitoring_window_seconds: How long to monitor before declaring
            the deployment stable. Default 3600 (1 hour).
        poll_interval_seconds: How often to check production metrics.
            Default 60 (1 minute).
        git_dir: Path to the git repository for revert operations.
            Default "." (current directory).

    Spec: REQ-AUTO-007
    """

    regression_threshold: float = 0.05
    monitoring_window_seconds: float = 3600
    poll_interval_seconds: float = 60
    git_dir: str = "."


@dataclass
class RollbackResult:
    """Result of the monitoring/rollback process.

    **For engineers:**
        Reports whether the deployment was stable or rolled back,
        with timing and metric details.

    Spec: REQ-AUTO-007
    """

    stable: bool
    rolled_back: bool
    reason: str
    monitoring_duration_seconds: float = 0.0
    final_energy: float | None = None
    baseline_energy: float | None = None
    regression_amount: float | None = None
    reverted_commit: str | None = None


def git_get_head_commit(git_dir: str = ".") -> str:
    """Get the current HEAD commit hash.

    Spec: REQ-AUTO-007
    """
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=git_dir,
        timeout=10,
    )
    return result.stdout.strip()


def git_revert_commit(commit: str, git_dir: str = ".") -> bool:
    """Revert a specific git commit (non-interactive).

    Returns True if the revert succeeded, False otherwise.

    **For engineers:**
        Uses ``git revert --no-edit`` to create a new commit that undoes
        the specified commit. This is safer than ``git reset --hard``
        because it preserves history and can itself be reverted.

    Spec: REQ-AUTO-007
    """
    result = subprocess.run(
        ["git", "revert", "--no-edit", commit],
        capture_output=True,
        text=True,
        cwd=git_dir,
        timeout=30,
    )
    return result.returncode == 0


def monitor_and_rollback(
    hypothesis_id: str,
    deployed_commit: str,
    baseline_energy: float,
    measure_fn: Callable[[], float],
    config: RollbackConfig | None = None,
    experiment_log: ExperimentLog | None = None,
) -> RollbackResult:
    """Monitor production metrics and rollback if degraded.

    **Researcher summary:**
        Polls ``measure_fn()`` at regular intervals. If energy exceeds
        baseline + threshold, reverts the deployed commit and logs the
        regression. Returns stable/rolled-back result.

    **Detailed explanation for engineers:**
        This is the safety net for the autoresearch pipeline. After deploying
        a new hypothesis to production:

        1. Call ``measure_fn()`` periodically to get current energy
        2. Compare against ``baseline_energy`` + tolerance
        3. If energy is too high → revert the git commit and log failure
        4. If monitoring window passes without problems → declare stable

        The ``measure_fn`` is a callback you provide that returns the current
        production energy metric. This could be:
        - Running the benchmark suite and returning mean energy
        - Querying a monitoring system
        - Evaluating the model on a held-out test set

    Args:
        hypothesis_id: ID of the deployed hypothesis (for logging).
        deployed_commit: Git commit hash to revert if rollback needed.
        baseline_energy: The energy before this hypothesis was deployed.
        measure_fn: Callable that returns current production energy.
            Called at each poll interval.
        config: Rollback configuration. Uses defaults if None.
        experiment_log: If provided, regression is logged here.

    Returns:
        RollbackResult indicating stable or rolled-back, with details.

    Spec: REQ-AUTO-007
    """
    if config is None:
        config = RollbackConfig()

    # Compute the energy threshold that triggers rollback.
    # Uses absolute tolerance for correct handling of negative energies.
    abs_threshold = abs(baseline_energy) * config.regression_threshold
    max_allowed_energy = baseline_energy + abs_threshold

    start = time.monotonic()

    while True:
        elapsed = time.monotonic() - start

        # --- Check if monitoring window has passed ---
        if elapsed >= config.monitoring_window_seconds:
            logger.info(
                "Monitoring window passed (%.1fs). Deployment stable.",
                elapsed,
            )
            return RollbackResult(
                stable=True,
                rolled_back=False,
                reason="Monitoring window passed without regression",
                monitoring_duration_seconds=elapsed,
            )

        # --- Measure current production energy ---
        current_energy = measure_fn()

        # --- Check for regression ---
        if current_energy > max_allowed_energy:
            regression = current_energy - baseline_energy
            logger.warning(
                "REGRESSION DETECTED: energy %.4f > threshold %.4f "
                "(baseline=%.4f, regression=%.4f). Rolling back commit %s.",
                current_energy,
                max_allowed_energy,
                baseline_energy,
                regression,
                deployed_commit,
            )

            # --- Rollback ---
            reverted = git_revert_commit(deployed_commit, config.git_dir)
            reverted_commit = deployed_commit if reverted else None

            if not reverted:
                logger.error("Git revert FAILED for commit %s", deployed_commit)

            # --- Log the regression ---
            if experiment_log is not None:
                experiment_log.append(ExperimentEntry(
                    id=f"rollback-{hypothesis_id}",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    hypothesis_code="",
                    hypothesis_description=f"Rollback of {hypothesis_id}",
                    sandbox_success=False,
                    eval_verdict="FAIL",
                    eval_reason=f"Production regression: energy {current_energy:.4f} > threshold {max_allowed_energy:.4f}",
                    outcome="rejected",
                ))

            return RollbackResult(
                stable=False,
                rolled_back=reverted,
                reason=f"Energy regression: {current_energy:.4f} > {max_allowed_energy:.4f}",
                monitoring_duration_seconds=elapsed,
                final_energy=current_energy,
                baseline_energy=baseline_energy,
                regression_amount=regression,
                reverted_commit=reverted_commit,
            )

        # --- Wait for next poll ---
        time.sleep(config.poll_interval_seconds)
