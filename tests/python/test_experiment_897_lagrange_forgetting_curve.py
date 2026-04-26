"""Tests for Exp 897: Lagrange forgetting curve for constraint memory lifecycle.

Covers:
- tick() applies exponential decay to constraint weights (REQ-FR11-007)
- Expiry at weight < 1e-4 removes constraint from memory (REQ-FR11-007)
- get_replay_candidates() returns correct IDs for aging+active constraints (REQ-FR11-007)
- 10-session relay shows forgetting improves or matches constraint precision vs baseline

Spec: REQ-FR11-007, SCENARIO-FR11-007
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from carnot.pipeline.lagrange_updater import ConstraintRecord, LagrangeAdaptiveUpdater


# ---------------------------------------------------------------------------
# REQ-FR11-007: tick() applies exponential decay
# ---------------------------------------------------------------------------


class TestTickDecay:
    """tick() MUST multiply each constraint weight by exp(-forgetting_lambda).

    Spec: REQ-FR11-007
    """

    def test_single_step_decay(self) -> None:
        """REQ-FR11-007: After tick(1), weight = initial_weight * exp(-lambda)."""
        updater = LagrangeAdaptiveUpdater(forgetting_lambda=0.05)
        updater.update("c1", violated=True)
        w_before = updater.constraint_weights["c1"]

        updater.tick(1)

        expected = w_before * math.exp(-0.05)
        assert updater.constraint_weights["c1"] == pytest.approx(expected, rel=1e-6)

    def test_multiple_steps_decay(self) -> None:
        """REQ-FR11-007: tick(n) is equivalent to n single-step ticks in terms of final weight."""
        updater_a = LagrangeAdaptiveUpdater(forgetting_lambda=0.1)
        updater_b = LagrangeAdaptiveUpdater(forgetting_lambda=0.1)
        updater_a.update("c1", violated=True)
        updater_b.update("c1", violated=True)

        # Single tick(5) vs five tick(1) calls
        updater_a.tick(5)
        for _ in range(5):
            updater_b.tick(1)

        assert updater_a.constraint_weights.get("c1", 0.0) == pytest.approx(
            updater_b.constraint_weights.get("c1", 0.0), rel=1e-6
        )

    def test_age_incremented(self) -> None:
        """REQ-FR11-007: tick() increments constraint_ages by the step count."""
        updater = LagrangeAdaptiveUpdater()
        updater.update("c1", violated=False)
        assert updater.constraint_ages["c1"] == 0

        updater.tick(3)
        assert updater.constraint_ages["c1"] == 3

    def test_no_forgetting_when_lambda_zero(self) -> None:
        """Edge case: forgetting_lambda=0 means exp(0)=1, so weight never decays."""
        updater = LagrangeAdaptiveUpdater(forgetting_lambda=0.0)
        updater.update("c1", violated=True)
        w = updater.constraint_weights["c1"]

        for _ in range(50):
            updater.tick(1)

        assert updater.constraint_weights["c1"] == pytest.approx(w, rel=1e-9)


# ---------------------------------------------------------------------------
# REQ-FR11-007: expiry at weight < 1e-4
# ---------------------------------------------------------------------------


class TestExpiry:
    """Constraints with weight < 1e-4 MUST be removed from active memory.

    Spec: REQ-FR11-007
    """

    def test_constraint_expires_after_enough_ticks(self) -> None:
        """REQ-FR11-007: A constraint decays to below 1e-4 and is removed."""
        updater = LagrangeAdaptiveUpdater(forgetting_lambda=0.5, weight_init=1.0)
        updater.update("c1", violated=False)  # weight starts at 1.0

        # After k ticks: weight = 1.0 * exp(-0.5 * k)
        # Need weight < 1e-4: exp(-0.5k) < 1e-4 → k > ln(1e4)/0.5 ≈ 18.4 → k=19
        for _ in range(20):
            updater.tick(1)

        assert "c1" not in updater.constraint_weights
        assert "c1" not in updater.constraint_ages
        assert updater.total_expired >= 1

    def test_expired_returns_id_list(self) -> None:
        """tick() returns the list of expired constraint IDs."""
        updater = LagrangeAdaptiveUpdater(forgetting_lambda=1.0, weight_init=1.0)
        updater.update("c1", violated=False)  # weight=1.0
        updater.update("c2", violated=False)  # weight=1.0

        # After 10 ticks with lambda=1: weight = exp(-10) ≈ 4.5e-5 < 1e-4 → should expire
        expired = updater.tick(10)

        assert "c1" in expired
        assert "c2" in expired

    def test_unexpired_constraint_stays(self) -> None:
        """A constraint that is renewed by violations does not expire prematurely."""
        updater = LagrangeAdaptiveUpdater(forgetting_lambda=0.05, weight_init=1.0)
        updater.update("c1", violated=True)  # weight = 1.1

        # With lambda=0.05 and initial weight 1.1, need >> 100 ticks to reach 1e-4.
        # ln(1.1/1e-4) / 0.05 ≈ 188 ticks.  After 50 ticks it should still be alive.
        for _ in range(50):
            updater.tick(1)

        assert "c1" in updater.constraint_weights

    def test_update_after_expiry_re_adds(self) -> None:
        """Calling update() after a constraint was expired re-initializes it."""
        updater = LagrangeAdaptiveUpdater(forgetting_lambda=1.0, weight_init=1.0)
        updater.update("c1", violated=False)
        updater.tick(10)  # expires c1

        assert "c1" not in updater.constraint_weights

        # Re-add by calling update again — should be accepted as a new constraint
        updater.update("c1", violated=True)
        assert "c1" in updater.constraint_weights
        assert updater.constraint_weights["c1"] == pytest.approx(1.1, rel=1e-6)


# ---------------------------------------------------------------------------
# REQ-FR11-007: get_replay_candidates returns correct IDs
# ---------------------------------------------------------------------------


class TestGetReplayCandidates:
    """get_replay_candidates() MUST return only aging+active constraint IDs.

    Spec: REQ-FR11-007, SCENARIO-FR11-007
    """

    def _make_updater_with_aged_constraint(
        self, cid: str, n_violations: int, n_total: int
    ) -> LagrangeAdaptiveUpdater:
        """Helper: create updater, add a constraint, age it below REPLAY_WEIGHT_THRESHOLD."""
        updater = LagrangeAdaptiveUpdater(
            forgetting_lambda=0.5, replay_threshold=0.8, weight_init=1.0
        )
        # Record violation history before aging
        for i in range(n_total):
            updater.update(cid, violated=(i < n_violations))
        # Decay weight below REPLAY_WEIGHT_THRESHOLD (0.1) without expiring it.
        # At lambda=0.5, after 3 ticks: weight_base * exp(-1.5) ≈ 0.22 * weight_base
        # We need weight < 0.1; boost initial weight first then decay:
        # Force via direct record manipulation so the violation history is preserved.
        updater._records[cid].weight = 0.05
        updater.constraint_weights[cid] = 0.05
        return updater

    def test_high_violation_rate_returned(self) -> None:
        """SCENARIO-FR11-007: constraint with violation_rate=0.9 IS a replay candidate."""
        updater = self._make_updater_with_aged_constraint("c1", n_violations=9, n_total=10)
        candidates = updater.get_replay_candidates()
        assert "c1" in candidates

    def test_low_violation_rate_not_returned(self) -> None:
        """SCENARIO-FR11-007: constraint with violation_rate=0.1 is NOT a replay candidate."""
        updater = self._make_updater_with_aged_constraint("c2", n_violations=1, n_total=10)
        candidates = updater.get_replay_candidates()
        assert "c2" not in candidates

    def test_healthy_weight_not_returned(self) -> None:
        """Constraints with weight >= REPLAY_WEIGHT_THRESHOLD are not aging-out; skip them."""
        updater = LagrangeAdaptiveUpdater(replay_threshold=0.5)
        # Add constraint with high weight (not aging)
        updater.update("c1", violated=True)
        # c1.weight = 1.1 >> 0.1, so even with high violation rate it's not a replay candidate
        candidates = updater.get_replay_candidates()
        assert "c1" not in candidates

    def test_external_violation_rates_override(self) -> None:
        """External violation_rates dict overrides internal empirical rates."""
        updater = LagrangeAdaptiveUpdater(replay_threshold=0.8)
        updater.update("c1", violated=False)  # internal rate = 0.0
        updater._records["c1"].weight = 0.05
        updater.constraint_weights["c1"] = 0.05

        # External rate says c1 is being violated 90% of the time
        candidates = updater.get_replay_candidates(violation_rates={"c1": 0.9})
        assert "c1" in candidates

    def test_apply_replay_resets_weight(self) -> None:
        """apply_replay() resets candidates' weights to weight_init and returns count."""
        updater = LagrangeAdaptiveUpdater(replay_threshold=0.8, weight_init=1.0)
        for i in range(10):
            updater.update("c1", violated=(i < 9))  # violation_rate=0.9
        updater._records["c1"].weight = 0.05
        updater.constraint_weights["c1"] = 0.05

        count = updater.apply_replay()

        assert count == 1
        assert updater.constraint_weights["c1"] == pytest.approx(1.0, rel=1e-6)
        assert updater.total_replay_events == 1


# ---------------------------------------------------------------------------
# Benchmark: constraint_precision improves with forgetting (10-session relay)
# ---------------------------------------------------------------------------


class TestForgettingRelay:
    """10-session relay benchmark shows forgetting improves or matches constraint precision.

    Spec: REQ-FR11-007
    """

    def _run_relay(
        self,
        forgetting_lambda: float,
        n_sessions: int = 10,
        n_questions: int = 20,
        seed: int = 42,
    ) -> dict:
        """Simulate a relay: each session adds new constraints (some become stale).

        Returns a dict with constraint_count_final and constraint_precision_final.
        """
        import random

        rng = random.Random(seed)
        updater = LagrangeAdaptiveUpdater(
            forgetting_lambda=forgetting_lambda,
            replay_threshold=0.8,
        )
        step = 0

        for session in range(n_sessions):
            for q in range(n_questions):
                # Each question produces 1-3 constraints.  First 5 sessions use
                # "early" constraint IDs; later sessions use new ones to model drift.
                n_constraints = rng.randint(1, 3)
                for c in range(n_constraints):
                    cid = f"s{session}_q{q}_c{c}"
                    # Only constraints from the most recent 2 sessions tend to recur.
                    # Simulate recurring errors for recent constraints:
                    is_recent = session >= n_sessions - 2
                    violated = rng.random() < (0.8 if is_recent else 0.2)
                    updater.update(cid, violated=violated)

                step += 1
                updater.apply_replay()
                updater.tick(1)

        return {
            "constraint_count": updater.n_constraints,
            "constraint_precision": updater.constraint_precision,
            "total_replay_events": updater.total_replay_events,
            "total_expired": updater.total_expired,
        }

    def test_forgetting_reduces_stale_constraints(self) -> None:
        """With forgetting (lambda=0.05), constraint count at session 10 is lower than baseline."""
        baseline = self._run_relay(forgetting_lambda=0.0)
        with_forgetting = self._run_relay(forgetting_lambda=0.05)

        assert with_forgetting["constraint_count"] <= baseline["constraint_count"], (
            f"Expected forgetting to reduce constraint count: "
            f"forget={with_forgetting['constraint_count']} "
            f"baseline={baseline['constraint_count']}"
        )

    def test_forgetting_precision_not_worse_than_baseline(self) -> None:
        """Constraint precision with forgetting MUST be >= baseline - 0.02 (neutral or better).

        Spec: REQ-FR11-007 acceptance criteria — forgetting_neutral or forgetting_improves_precision
        """
        baseline = self._run_relay(forgetting_lambda=0.0)
        with_forgetting = self._run_relay(forgetting_lambda=0.05)

        precision_delta = (
            with_forgetting["constraint_precision"] - baseline["constraint_precision"]
        )
        assert precision_delta >= -0.02, (
            f"Forgetting hurt precision by more than 0.02: delta={precision_delta:.4f}"
        )

    def test_deliverable_json_exists(self) -> None:
        """The experiment deliverable JSON must exist and contain required fields."""
        result_path = Path("results/experiment_897_lagrange_forgetting_curve.json")
        assert result_path.exists(), f"Deliverable not found at {result_path}"

        with result_path.open() as f:
            data = json.load(f)

        required_fields = [
            "experiment",
            "honest_verdict",
            "constraint_precision_no_forget",
            "constraint_precision_with_forget",
            "memory_size_no_forget",
            "memory_size_with_forget",
            "forgetting_rate_best_lambda",
            "replay_events",
        ]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        assert data["honest_verdict"] in {
            "forgetting_improves_precision",
            "forgetting_neutral",
            "forgetting_hurts_precision",
        }
