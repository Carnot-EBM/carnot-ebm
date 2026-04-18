"""Tests for AdversarialBenchmarkResult — three-condition adversarial benchmark metric.

Spec: REQ-BENCH-020, REQ-BENCH-021, REQ-BENCH-022,
      SCENARIO-BENCH-039, SCENARIO-BENCH-040, SCENARIO-BENCH-041
"""

from __future__ import annotations

import math

import pytest

from carnot.pipeline.adversarial_benchmark_result import (
    AdversarialBenchmarkResult,
    _wilson_ci,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make(
    standard_acc: float = 0.70,
    adversarial_baseline_acc: float = 0.55,
    adversarial_carnot_acc: float = 0.65,
    n_questions: int = 50,
    carnot_standard_improvement: float = 0.05,
    model_id: str = "TestModel",
) -> AdversarialBenchmarkResult:
    return AdversarialBenchmarkResult(
        model_id=model_id,
        standard_acc=standard_acc,
        adversarial_baseline_acc=adversarial_baseline_acc,
        adversarial_carnot_acc=adversarial_carnot_acc,
        n_questions=n_questions,
        carnot_standard_improvement=carnot_standard_improvement,
    )


# ---------------------------------------------------------------------------
# adversarial_drop — SCENARIO-BENCH-039
# ---------------------------------------------------------------------------


class TestAdversarialDrop:
    """SCENARIO-BENCH-039: adversarial_drop is positive when adversarial < standard."""

    def test_drop_positive_when_adversarial_lower(self):
        r = _make(standard_acc=0.70, adversarial_baseline_acc=0.55)
        assert abs(r.adversarial_drop - 0.15) < 1e-9

    def test_drop_zero_when_equal(self):
        r = _make(standard_acc=0.60, adversarial_baseline_acc=0.60)
        assert r.adversarial_drop == 0.0

    def test_drop_negative_when_adversarial_higher(self):
        # Unusual but valid: model is somehow better on adversarial
        r = _make(standard_acc=0.50, adversarial_baseline_acc=0.60)
        assert abs(r.adversarial_drop - (-0.10)) < 1e-9


# ---------------------------------------------------------------------------
# carnot_adversarial_improvement
# ---------------------------------------------------------------------------


class TestCarnotAdversarialImprovement:
    """REQ-BENCH-021: Carnot improvement = adversarial_carnot_acc - adversarial_baseline_acc."""

    def test_positive_improvement(self):
        r = _make(adversarial_baseline_acc=0.55, adversarial_carnot_acc=0.65)
        assert abs(r.carnot_adversarial_improvement - 0.10) < 1e-9

    def test_zero_improvement(self):
        r = _make(adversarial_baseline_acc=0.55, adversarial_carnot_acc=0.55)
        assert r.carnot_adversarial_improvement == 0.0

    def test_negative_improvement(self):
        # Pipeline degraded performance — must NOT clamp to 0
        r = _make(adversarial_baseline_acc=0.60, adversarial_carnot_acc=0.50)
        assert abs(r.carnot_adversarial_improvement - (-0.10)) < 1e-9


# ---------------------------------------------------------------------------
# thesis_confirmed — SCENARIO-BENCH-040
# ---------------------------------------------------------------------------


class TestThesisConfirmed:
    """SCENARIO-BENCH-040: thesis_confirmed requires adversarial improvement > standard."""

    def test_confirmed_when_adversarial_improvement_exceeds_standard(self):
        # adversarial improvement=0.10 > standard_improvement=0.05
        r = _make(
            adversarial_baseline_acc=0.55,
            adversarial_carnot_acc=0.65,
            carnot_standard_improvement=0.05,
        )
        assert r.thesis_confirmed is True

    def test_not_confirmed_when_standard_improvement_exceeds_adversarial(self):
        # adversarial improvement=0.10 < standard_improvement=0.15
        r = _make(
            adversarial_baseline_acc=0.55,
            adversarial_carnot_acc=0.65,
            carnot_standard_improvement=0.15,
        )
        assert r.thesis_confirmed is False

    def test_not_confirmed_when_equal(self):
        # Tie goes to 'not confirmed' (must EXCEED, not merely match)
        r = _make(
            adversarial_baseline_acc=0.55,
            adversarial_carnot_acc=0.65,
            carnot_standard_improvement=0.10,
        )
        assert r.thesis_confirmed is False

    def test_confirmed_when_standard_improvement_is_zero_and_carnot_improves(self):
        r = _make(
            adversarial_baseline_acc=0.55,
            adversarial_carnot_acc=0.60,
            carnot_standard_improvement=0.0,
        )
        assert r.thesis_confirmed is True

    def test_not_confirmed_when_no_improvement_at_all(self):
        r = _make(
            adversarial_baseline_acc=0.55,
            adversarial_carnot_acc=0.55,
            carnot_standard_improvement=0.0,
        )
        assert r.thesis_confirmed is False


# ---------------------------------------------------------------------------
# ci_95_adversarial
# ---------------------------------------------------------------------------


class TestCI95Adversarial:
    """Wilson CI stays in [0,1] and has correct width for n=50."""

    def test_bounds_in_range(self):
        r = _make(adversarial_carnot_acc=0.60, n_questions=50)
        lo, hi = r.ci_95_adversarial
        assert 0.0 <= lo <= hi <= 1.0

    def test_ci_near_zero(self):
        r = _make(adversarial_carnot_acc=0.0, n_questions=50)
        lo, hi = r.ci_95_adversarial
        assert lo >= 0.0
        assert hi >= 0.0

    def test_ci_near_one(self):
        r = _make(adversarial_carnot_acc=1.0, n_questions=50)
        lo, hi = r.ci_95_adversarial
        assert lo <= 1.0
        assert hi <= 1.0

    def test_ci_midpoint_contains_p(self):
        # The midpoint of the CI should be close to the observed proportion
        r = _make(adversarial_carnot_acc=0.64, n_questions=50)
        lo, hi = r.ci_95_adversarial
        mid = (lo + hi) / 2
        assert abs(mid - 0.64) < 0.05  # Wilson center is shifted slightly


# ---------------------------------------------------------------------------
# _wilson_ci helper directly
# ---------------------------------------------------------------------------


class TestWilsonCI:
    """Unit tests for the _wilson_ci helper function."""

    def test_n1_does_not_crash(self):
        # n=0 is clamped to 1 internally
        lo, hi = _wilson_ci(0.5, 0)
        assert 0.0 <= lo <= hi <= 1.0

    def test_known_value(self):
        # p=0.5, n=100: Wilson CI ≈ (0.404, 0.596)
        lo, hi = _wilson_ci(0.5, 100)
        assert abs(lo - 0.404) < 0.01
        assert abs(hi - 0.596) < 0.01


# ---------------------------------------------------------------------------
# to_dict serialization
# ---------------------------------------------------------------------------


class TestToDict:
    """to_dict returns all required fields as JSON-serializable values."""

    def test_all_keys_present(self):
        r = _make()
        d = r.to_dict()
        required = {
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
        }
        assert required.issubset(d.keys())

    def test_ci_95_adversarial_is_list(self):
        r = _make()
        d = r.to_dict()
        assert isinstance(d["ci_95_adversarial"], list)
        assert len(d["ci_95_adversarial"]) == 2

    def test_thesis_confirmed_is_bool(self):
        r = _make()
        d = r.to_dict()
        assert isinstance(d["thesis_confirmed"], bool)

    def test_computed_fields_match_properties(self):
        r = _make()
        d = r.to_dict()
        assert abs(d["adversarial_drop"] - r.adversarial_drop) < 1e-12
        assert abs(d["carnot_adversarial_improvement"] - r.carnot_adversarial_improvement) < 1e-12
        assert d["thesis_confirmed"] == r.thesis_confirmed


# ---------------------------------------------------------------------------
# Default carnot_standard_improvement
# ---------------------------------------------------------------------------


class TestDefaultStandardImprovement:
    """carnot_standard_improvement defaults to 0.0 when not provided."""

    def test_default_zero(self):
        r = AdversarialBenchmarkResult(
            model_id="M",
            standard_acc=0.70,
            adversarial_baseline_acc=0.55,
            adversarial_carnot_acc=0.65,
            n_questions=50,
        )
        assert r.carnot_standard_improvement == 0.0

    def test_any_positive_improvement_confirms_thesis_with_default(self):
        r = AdversarialBenchmarkResult(
            model_id="M",
            standard_acc=0.70,
            adversarial_baseline_acc=0.55,
            adversarial_carnot_acc=0.60,
            n_questions=50,
        )
        # improvement=0.05 > 0.0 default → confirmed
        assert r.thesis_confirmed is True
