"""Tests for AdversarialV2Result alias and Exp 479 thesis scenarios.

Spec: REQ-BENCH-031, REQ-BENCH-032, REQ-BENCH-033,
      SCENARIO-BENCH-050, SCENARIO-BENCH-051, SCENARIO-BENCH-052
"""

from __future__ import annotations

from carnot.pipeline.adversarial_benchmark_result import (
    AdversarialBenchmarkResult,
    AdversarialV2Result,
)


# ---------------------------------------------------------------------------
# Alias identity
# ---------------------------------------------------------------------------


class TestAdversarialV2ResultIsAlias:
    """AdversarialV2Result must be the same class as AdversarialBenchmarkResult."""

    def test_alias_is_same_class(self):
        # REQ-BENCH-032: AdversarialV2Result is the canonical result type for Exp 479
        assert AdversarialV2Result is AdversarialBenchmarkResult

    def test_instantiation_via_alias(self):
        r = AdversarialV2Result(
            model_id="TestModel",
            standard_acc=0.70,
            adversarial_baseline_acc=0.55,
            adversarial_carnot_acc=0.65,
            n_questions=50,
            carnot_standard_improvement=0.05,
        )
        assert r.standard_acc == 0.70


# ---------------------------------------------------------------------------
# SCENARIO-BENCH-052: thesis_confirmed=True when adversarial improvement (0.10) > standard (0.05)
# ---------------------------------------------------------------------------


class TestThesisConfirmedScenario052:
    """SCENARIO-BENCH-052: adversarial improvement 0.10 > standard improvement 0.05 → confirmed."""

    def test_thesis_confirmed_when_adversarial_exceeds_standard(self):
        # adversarial improvement = 0.65 - 0.55 = 0.10 > carnot_standard_improvement = 0.05
        r = AdversarialV2Result(
            model_id="TestModel",
            standard_acc=0.70,
            adversarial_baseline_acc=0.55,
            adversarial_carnot_acc=0.65,
            n_questions=50,
            carnot_standard_improvement=0.05,
        )
        assert abs(r.carnot_adversarial_improvement - 0.10) < 1e-9
        assert r.thesis_confirmed is True

    def test_thesis_not_confirmed_when_equal(self):
        # Tie: adversarial improvement == standard improvement → False (must EXCEED)
        r = AdversarialV2Result(
            model_id="TestModel",
            standard_acc=0.70,
            adversarial_baseline_acc=0.55,
            adversarial_carnot_acc=0.65,
            n_questions=50,
            carnot_standard_improvement=0.10,
        )
        assert r.thesis_confirmed is False


# ---------------------------------------------------------------------------
# adversarial_drop computed correctly — SCENARIO-BENCH-051
# ---------------------------------------------------------------------------


class TestAdversarialDropScenario051:
    """SCENARIO-BENCH-051: adversarial_drop = standard_acc - adversarial_baseline_acc."""

    def test_drop_computed_correctly(self):
        r = AdversarialV2Result(
            model_id="TestModel",
            standard_acc=0.70,
            adversarial_baseline_acc=0.55,
            adversarial_carnot_acc=0.65,
            n_questions=50,
        )
        assert abs(r.adversarial_drop - 0.15) < 1e-9

    def test_drop_zero_when_equal(self):
        r = AdversarialV2Result(
            model_id="TestModel",
            standard_acc=0.60,
            adversarial_baseline_acc=0.60,
            adversarial_carnot_acc=0.60,
            n_questions=50,
        )
        assert r.adversarial_drop == 0.0

    def test_drop_negative_when_adversarial_higher(self):
        r = AdversarialV2Result(
            model_id="TestModel",
            standard_acc=0.50,
            adversarial_baseline_acc=0.65,
            adversarial_carnot_acc=0.65,
            n_questions=50,
        )
        assert r.adversarial_drop < 0.0


# ---------------------------------------------------------------------------
# ci_95_adversarial stays in [0, 1] — REQ-BENCH-033
# ---------------------------------------------------------------------------


class TestCI95ForV2Result:
    """ci_95_adversarial bounds must be in [0, 1] for any valid input."""

    def test_ci_bounds_valid(self):
        r = AdversarialV2Result(
            model_id="TestModel",
            standard_acc=0.70,
            adversarial_baseline_acc=0.55,
            adversarial_carnot_acc=0.65,
            n_questions=50,
        )
        lo, hi = r.ci_95_adversarial
        assert 0.0 <= lo <= hi <= 1.0
