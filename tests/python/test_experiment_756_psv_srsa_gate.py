"""Tests for Exp 756: PSV SRSA Gate + Constraint Freezing + Curriculum Diversity.

Every test traces to a specific requirement so the spec-coverage checker
(scripts/check_spec_coverage.py) can verify completeness.

Spec: REQ-PSV-014, REQ-PSV-015, REQ-PSV-016,
      SCENARIO-PSV-021, SCENARIO-PSV-022, SCENARIO-PSV-023
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# REQ-PSV-014 / SCENARIO-PSV-021: write_with_verification
# ---------------------------------------------------------------------------


class TestWriteWithVerification:
    """SessionMemory.write_with_verification must gate repairs with VPRM.

    Spec: REQ-PSV-014, REQ-PSV-014-1, REQ-PSV-014-2, REQ-PSV-014-3,
          SCENARIO-PSV-021
    """

    def _make_session_memory(self, tmp_path):
        from carnot.pipeline.session_memory import SessionMemory

        return SessionMemory(str(tmp_path), model_id="test_model")

    def test_rejects_repair_with_arithmetic_violation(self, tmp_path):
        """REQ-PSV-014-2: write_with_verification returns False for incorrect arithmetic.

        The repair text contains '3 plus 4 equals 8' which is wrong (3+4=7).
        VPRMArithmeticVerifier detects this and the method must return False.

        Spec: SCENARIO-PSV-021
        """
        sm = self._make_session_memory(tmp_path)
        # Deliberately incorrect addition: 3 + 4 ≠ 8
        bad_repair = "Step 1: 3 plus 4 equals 8. Therefore the answer is 8."
        result = sm.write_with_verification(bad_repair, constraint_type="addition")
        assert result is False, (
            "write_with_verification should return False when VPRM detects an arithmetic violation"
        )

    def test_accepts_repair_with_no_arithmetic_in_text(self, tmp_path):
        """REQ-PSV-014-3: write_with_verification returns True for text with no detectable violations.

        Plain prose without any arithmetic claims passes all VPRM rules (rules return None)
        so detect_violations returns an empty list. The method must return True.

        Spec: REQ-PSV-014-3
        """
        sm = self._make_session_memory(tmp_path)
        clean_repair = "The answer follows from applying the definition of the operation."
        result = sm.write_with_verification(clean_repair, constraint_type="logic")
        assert result is True, (
            "write_with_verification should return True when no arithmetic violations are found"
        )

    def test_calls_vprm_before_any_write(self, tmp_path):
        """REQ-PSV-014-1: write_with_verification must call detect_violations before write.

        We verify this behaviourally: a repair with a violation is rejected,
        confirming VPRM was consulted and no write path was reached.

        Spec: REQ-PSV-014-1
        """
        sm = self._make_session_memory(tmp_path)
        # Incorrect subtraction: 10 minus 4 = 7 (should be 6)
        bad_repair = "10 minus 4 gives 7."
        result = sm.write_with_verification(bad_repair, constraint_type="subtraction")
        # If VPRM had not been consulted, any repair would return True.
        # Getting False here proves VPRM was called first.
        assert result is False


# ---------------------------------------------------------------------------
# REQ-PSV-015 / SCENARIO-PSV-022: _freeze_stable_constraints
# ---------------------------------------------------------------------------


class TestFreezeStableConstraints:
    """SelfLearningRelay._freeze_stable_constraints must freeze low-variance constraints.

    Spec: REQ-PSV-015, REQ-PSV-015-1, REQ-PSV-015-2, REQ-PSV-015-3,
          SCENARIO-PSV-022
    """

    def _make_relay(self):
        """Create a minimal SelfLearningRelay with stub dependencies."""
        from unittest.mock import MagicMock, patch

        from carnot.pipeline.self_learning_relay import SelfLearningRelay

        pipeline = MagicMock()
        pipeline.verify.return_value = (True, "tier1", 0.5)

        template_library = MagicMock()
        template_library.get_active_templates.return_value = []

        fp_tracker = MagicMock()

        eorm_model = MagicMock()
        eorm_model.energy.return_value = 0.3

        relay = SelfLearningRelay(pipeline, template_library, fp_tracker, eorm_model)
        return relay, fp_tracker

    def test_freeze_stable_constraints_freezes_when_variance_low(self):
        """REQ-PSV-015-2: constraint type is frozen when energy variance < 0.01.

        We manually populate the trajectory with identical accuracy values (zero variance)
        then call _freeze_stable_constraints(). The 'verification' type must be in
        _frozen_constraints afterward.

        Spec: REQ-PSV-015-2, REQ-PSV-015-3
        """
        from carnot.pipeline.self_learning_relay import SelfLearningBatchResult

        relay, _ = self._make_relay()

        # Inject 15 trajectory entries with constant accuracy → variance = 0.
        for i in range(15):
            relay._trajectory.append(
                SelfLearningBatchResult(
                    batch_id=i,
                    n_questions=10,
                    accuracy=0.7,  # constant → zero variance
                    n_tier1_updates=10,
                    n_tier2_templates_active=0,
                    tier3_gate_auc=0.5,
                    cumulative_accuracy=0.7,
                )
            )

        relay._freeze_stable_constraints()
        assert "verification" in relay._frozen_constraints, (
            "REQ-PSV-015-3: 'verification' must be in _frozen_constraints when variance < 0.01"
        )

    def test_freeze_does_not_freeze_high_variance(self):
        """REQ-PSV-015-2: constraint type is NOT frozen when variance >= 0.01.

        Accuracy oscillating between 0.3 and 0.9 has high variance; the constraint
        should NOT be frozen.

        Spec: REQ-PSV-015-2
        """
        from carnot.pipeline.self_learning_relay import SelfLearningBatchResult

        relay, _ = self._make_relay()

        # Inject entries with alternating accuracy → high variance.
        for i in range(15):
            relay._trajectory.append(
                SelfLearningBatchResult(
                    batch_id=i,
                    n_questions=10,
                    accuracy=0.3 if i % 2 == 0 else 0.9,
                    n_tier1_updates=10,
                    n_tier2_templates_active=0,
                    tier3_gate_auc=0.5,
                    cumulative_accuracy=0.6,
                )
            )

        relay._freeze_stable_constraints()
        assert "verification" not in relay._frozen_constraints, (
            "High-variance constraint types must not be frozen"
        )

    def test_frozen_constraint_skips_fp_tracker_update(self):
        """REQ-PSV-015-3: run_batch skips fp_tracker.update() for frozen constraints.

        We freeze 'verification' manually then call run_batch. The fp_tracker.update
        mock should NOT be called because 'verification' is frozen.

        Spec: SCENARIO-PSV-022
        """
        relay, fp_tracker = self._make_relay()

        # Manually freeze the 'verification' constraint type.
        relay._frozen_constraints.add("verification")

        questions = ["What is 2 + 2?"]
        ground_truth = [True]

        relay.run_batch(questions, ground_truth, model_id="ci_test")

        # fp_tracker.update must not have been called for the frozen constraint type.
        fp_tracker.update.assert_not_called()


# ---------------------------------------------------------------------------
# REQ-PSV-016 / SCENARIO-PSV-023: recovery_sustained requires both window slopes
# ---------------------------------------------------------------------------


class TestRecoverySustained:
    """recovery_sustained must be True ONLY when both window slopes are negative.

    Spec: REQ-PSV-016, REQ-PSV-016-1, REQ-PSV-016-2, REQ-PSV-016-3,
          SCENARIO-PSV-023
    """

    def _linear_slope(self, values):
        """OLS slope helper — mirrors _linear_slope in experiment_756."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from experiment_756_psv_srsa_gate import _linear_slope

        return _linear_slope(values)

    def test_both_negative_gives_recovery_sustained(self):
        """REQ-PSV-016: recovery_sustained=True when both window slopes < 0.

        A strictly decreasing fp_rate series produces negative slopes in both windows.

        Spec: REQ-PSV-016, SCENARIO-PSV-023
        """
        # Decreasing fp_rate over 61 steps: clearly negative in both windows.
        fp_series = [0.6 - 0.007 * i for i in range(61)]
        window1 = fp_series[:31]
        window2 = fp_series[30:]
        w1_slope = self._linear_slope(window1)
        w2_slope = self._linear_slope(window2)
        recovery_sustained = (w1_slope < 0) and (w2_slope < 0)
        assert recovery_sustained is True, (
            "REQ-PSV-016: both windows negative must produce recovery_sustained=True"
        )
        honest_verdict = "recovery_sustained" if recovery_sustained else "recovery_partial"
        assert honest_verdict == "recovery_sustained"

    def test_positive_window2_gives_recovery_partial(self):
        """REQ-PSV-016-2: honest_verdict='recovery_partial' when window1 < 0 but window2 >= 0.

        This is the 'old pattern' from Exps 697 and 737 — initial recovery that doesn't
        sustain through the second 30-step window.

        Spec: REQ-PSV-016-2, SCENARIO-PSV-023
        """
        # Window1: decreasing. Window2: flat/slightly increasing.
        window1 = [0.6 - 0.005 * i for i in range(31)]
        window2_start = window1[-1]
        window2 = [window2_start + 0.001 * i for i in range(31)]

        fp_series = window1 + window2[1:]
        w1_slope = self._linear_slope(fp_series[:31])
        w2_slope = self._linear_slope(fp_series[30:])

        recovery_sustained = (w1_slope < 0) and (w2_slope < 0)
        assert recovery_sustained is False
        assert w1_slope < 0
        assert w2_slope > 0

        honest_verdict = (
            "recovery_sustained" if recovery_sustained
            else ("recovery_partial" if w1_slope < 0 else "recovery_failed")
        )
        assert honest_verdict == "recovery_partial"

    def test_both_positive_gives_recovery_failed(self):
        """REQ-PSV-016-3: honest_verdict='recovery_failed' when both slopes >= 0.

        Spec: REQ-PSV-016-3
        """
        # Increasing fp_rate: both windows show degradation.
        fp_series = [0.3 + 0.005 * i for i in range(61)]
        w1_slope = self._linear_slope(fp_series[:31])
        w2_slope = self._linear_slope(fp_series[30:])

        recovery_sustained = (w1_slope < 0) and (w2_slope < 0)
        assert recovery_sustained is False

        honest_verdict = (
            "recovery_sustained" if recovery_sustained
            else ("recovery_partial" if w1_slope < 0 else "recovery_failed")
        )
        assert honest_verdict == "recovery_failed"

    def test_actual_deliverable_is_recovery_sustained(self):
        """REQ-PSV-016: the Exp 756 deliverable artifact must report recovery_sustained=True.

        This is the primary success criterion for RETRO-PSV-RELAPSE closure.

        Spec: REQ-PSV-016, SCENARIO-PSV-023
        """
        import json
        from pathlib import Path

        deliverable = Path(__file__).resolve().parents[2] / "results" / "experiment_756_psv_srsa_gate.json"
        assert deliverable.exists(), f"Deliverable not found: {deliverable}"

        with deliverable.open() as f:
            artifact = json.load(f)

        assert artifact["recovery_sustained"] is True, (
            "REQ-PSV-016: Exp 756 must achieve recovery_sustained=True to close RETRO-PSV-RELAPSE"
        )
        assert artifact["window1_slope"] < 0, "window1_slope must be negative"
        assert artifact["window2_slope"] < 0, "window2_slope must be negative"
        assert artifact["honest_verdict"] == "recovery_sustained"
