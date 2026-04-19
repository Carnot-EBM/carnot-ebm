"""Tests for AdversarialV3Result alias and Exp 490 thesis scenarios.

AdversarialV3Result is an alias for AdversarialBenchmarkResult (same class as V2).
These tests cover the new spec requirements REQ-BENCH-040/041/042 and
SCENARIO-BENCH-059/060/061.

Spec: REQ-BENCH-040, REQ-BENCH-041, REQ-BENCH-042,
      SCENARIO-BENCH-059, SCENARIO-BENCH-060, SCENARIO-BENCH-061
"""

from __future__ import annotations

from carnot.pipeline.adversarial_benchmark_result import (
    AdversarialBenchmarkResult,
    AdversarialV3Result,
)


# ---------------------------------------------------------------------------
# Alias identity
# ---------------------------------------------------------------------------


class TestAdversarialV3ResultIsAlias:
    """AdversarialV3Result must be the same class as AdversarialBenchmarkResult."""

    def test_alias_is_same_class(self):
        # REQ-BENCH-042: AdversarialV3Result is the canonical result type for Exp 490
        assert AdversarialV3Result is AdversarialBenchmarkResult

    def test_instantiation_via_alias(self):
        r = AdversarialV3Result(
            model_id="TestModel",
            standard_acc=0.70,
            adversarial_baseline_acc=0.55,
            adversarial_carnot_acc=0.65,
            n_questions=50,
            carnot_standard_improvement=0.05,
        )
        assert r.standard_acc == 0.70
        assert r.model_id == "TestModel"


# ---------------------------------------------------------------------------
# SCENARIO-BENCH-060: thesis_confirmed=True when adversarial improvement > standard
# ---------------------------------------------------------------------------


class TestThesisConfirmedScenario060:
    """SCENARIO-BENCH-060: adversarial improvement 0.10 > standard improvement 0.05 → confirmed."""

    def test_thesis_confirmed_when_adversarial_exceeds_standard(self):
        # REQ-BENCH-042: thesis_confirmed requires adversarial_improvement > standard_improvement
        # adversarial improvement = 0.65 - 0.55 = 0.10, carnot_standard_improvement = 0.05
        r = AdversarialV3Result(
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
        # Tie: adversarial improvement == standard improvement → False (must EXCEED strictly)
        r = AdversarialV3Result(
            model_id="TestModel",
            standard_acc=0.70,
            adversarial_baseline_acc=0.55,
            adversarial_carnot_acc=0.65,
            n_questions=50,
            carnot_standard_improvement=0.10,
        )
        assert r.thesis_confirmed is False

    def test_thesis_not_confirmed_when_adversarial_below_standard(self):
        r = AdversarialV3Result(
            model_id="TestModel",
            standard_acc=0.70,
            adversarial_baseline_acc=0.55,
            adversarial_carnot_acc=0.65,
            n_questions=50,
            carnot_standard_improvement=0.20,
        )
        assert r.thesis_confirmed is False


# ---------------------------------------------------------------------------
# SCENARIO-BENCH-061: adversarial_drop = standard_acc - adversarial_baseline_acc
# ---------------------------------------------------------------------------


class TestAdversarialDropScenario061:
    """SCENARIO-BENCH-061: adversarial_drop = standard_acc - adversarial_baseline_acc."""

    def test_drop_computed_correctly(self):
        # standard 0.70 - adversarial_baseline 0.55 = 0.15 drop (Apple finding: all LLMs regress)
        r = AdversarialV3Result(
            model_id="TestModel",
            standard_acc=0.70,
            adversarial_baseline_acc=0.55,
            adversarial_carnot_acc=0.65,
            n_questions=50,
        )
        assert abs(r.adversarial_drop - 0.15) < 1e-9

    def test_drop_zero_when_equal(self):
        r = AdversarialV3Result(
            model_id="TestModel",
            standard_acc=0.60,
            adversarial_baseline_acc=0.60,
            adversarial_carnot_acc=0.60,
            n_questions=50,
        )
        assert r.adversarial_drop == 0.0

    def test_drop_negative_when_adversarial_higher(self):
        # Unusual but honest: model performs better on adversarial than standard
        r = AdversarialV3Result(
            model_id="TestModel",
            standard_acc=0.50,
            adversarial_baseline_acc=0.65,
            adversarial_carnot_acc=0.65,
            n_questions=50,
        )
        assert r.adversarial_drop < 0.0


# ---------------------------------------------------------------------------
# carnot_adversarial_improvement property — REQ-BENCH-042
# ---------------------------------------------------------------------------


class TestCarnotAdversarialImprovement:
    """carnot_adversarial_improvement = adversarial_carnot_acc - adversarial_baseline_acc."""

    def test_improvement_positive(self):
        r = AdversarialV3Result(
            model_id="TestModel",
            standard_acc=0.70,
            adversarial_baseline_acc=0.55,
            adversarial_carnot_acc=0.65,
            n_questions=50,
        )
        assert abs(r.carnot_adversarial_improvement - 0.10) < 1e-9

    def test_improvement_zero(self):
        r = AdversarialV3Result(
            model_id="TestModel",
            standard_acc=0.70,
            adversarial_baseline_acc=0.60,
            adversarial_carnot_acc=0.60,
            n_questions=50,
        )
        assert r.carnot_adversarial_improvement == 0.0

    def test_improvement_negative(self):
        # Repair hurt accuracy — honest to report
        r = AdversarialV3Result(
            model_id="TestModel",
            standard_acc=0.70,
            adversarial_baseline_acc=0.60,
            adversarial_carnot_acc=0.55,
            n_questions=50,
        )
        assert r.carnot_adversarial_improvement < 0.0


# ---------------------------------------------------------------------------
# ci_95_adversarial stays in [0, 1] — REQ-BENCH-040
# ---------------------------------------------------------------------------


class TestCI95ForV3Result:
    """ci_95_adversarial bounds must be in [0, 1] for any valid input."""

    def test_ci_bounds_valid(self):
        r = AdversarialV3Result(
            model_id="TestModel",
            standard_acc=0.70,
            adversarial_baseline_acc=0.55,
            adversarial_carnot_acc=0.65,
            n_questions=50,
        )
        lo, hi = r.ci_95_adversarial
        assert 0.0 <= lo <= hi <= 1.0

    def test_ci_bounds_valid_at_extremes(self):
        # Near p=0 and p=1 — Wilson CI must stay in [0, 1]
        r_low = AdversarialV3Result(
            model_id="M",
            standard_acc=0.0,
            adversarial_baseline_acc=0.0,
            adversarial_carnot_acc=0.0,
            n_questions=50,
        )
        lo, hi = r_low.ci_95_adversarial
        assert 0.0 <= lo <= hi <= 1.0

        r_high = AdversarialV3Result(
            model_id="M",
            standard_acc=1.0,
            adversarial_baseline_acc=1.0,
            adversarial_carnot_acc=1.0,
            n_questions=50,
        )
        lo, hi = r_high.ci_95_adversarial
        assert 0.0 <= lo <= hi <= 1.0


# ---------------------------------------------------------------------------
# to_dict serialization — REQ-BENCH-041
# ---------------------------------------------------------------------------


class TestToDict:
    """to_dict() must include all required fields for the v4 artifact schema."""

    def test_to_dict_has_all_fields(self):
        r = AdversarialV3Result(
            model_id="TestModel",
            standard_acc=0.70,
            adversarial_baseline_acc=0.55,
            adversarial_carnot_acc=0.65,
            n_questions=50,
            carnot_standard_improvement=0.05,
        )
        d = r.to_dict()
        required_fields = [
            "model_id",
            "standard_acc",
            "adversarial_baseline_acc",
            "adversarial_carnot_acc",
            "n_questions",
            "carnot_standard_improvement",
            "adversarial_drop",
            "carnot_adversarial_improvement",
            "thesis_confirmed",
            "ci_95_adversarial",
        ]
        for field in required_fields:
            assert field in d, f"Missing field: {field}"

    def test_to_dict_thesis_confirmed_true(self):
        r = AdversarialV3Result(
            model_id="TestModel",
            standard_acc=0.70,
            adversarial_baseline_acc=0.55,
            adversarial_carnot_acc=0.65,
            n_questions=50,
            carnot_standard_improvement=0.05,
        )
        d = r.to_dict()
        assert d["thesis_confirmed"] is True
        assert abs(d["adversarial_drop"] - 0.15) < 1e-9
        assert abs(d["carnot_adversarial_improvement"] - 0.10) < 1e-9
