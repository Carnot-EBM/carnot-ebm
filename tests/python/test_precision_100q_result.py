"""Tests for carnot.pipeline.precision_100q_result.Precision100qResult.

Spec: REQ-BENCH-014, SCENARIO-BENCH-034
"""

from __future__ import annotations

import math

import pytest

from carnot.pipeline.precision_100q_result import Precision100qResult


def _make(
    pre: float = 0.40,
    post: float = 0.45,
    n: int = 100,
    extractor: str = "vericot,vprm",
    mode: str = "live_gpu",
) -> Precision100qResult:
    return Precision100qResult(
        model_id="TestModel",
        pre_accuracy=pre,
        post_accuracy=post,
        n_questions=n,
        extractor_used=extractor,
        inference_mode=mode,
    )


class TestPrecision100qResult:
    """Tests for Precision100qResult data class — REQ-BENCH-014, SCENARIO-BENCH-034."""

    # -----------------------------------------------------------------------
    # signed_improvement
    # -----------------------------------------------------------------------

    def test_signed_improvement_positive(self) -> None:
        """Improvement = post - pre when post > pre."""
        r = _make(pre=0.40, post=0.45)
        assert abs(r.signed_improvement - 0.05) < 1e-9

    def test_signed_improvement_negative(self) -> None:
        """Regression: signed_improvement < 0 when post < pre."""
        r = _make(pre=0.50, post=0.45)
        assert abs(r.signed_improvement - (-0.05)) < 1e-9

    def test_signed_improvement_zero(self) -> None:
        """No change: signed_improvement == 0.0."""
        r = _make(pre=0.45, post=0.45)
        assert r.signed_improvement == 0.0

    # -----------------------------------------------------------------------
    # is_positive
    # -----------------------------------------------------------------------

    def test_is_positive_when_improvement(self) -> None:
        """is_positive True when post > pre."""
        assert _make(pre=0.40, post=0.45).is_positive is True

    def test_is_positive_false_when_zero(self) -> None:
        """is_positive False when improvement == 0."""
        assert _make(pre=0.45, post=0.45).is_positive is False

    def test_is_positive_false_when_regression(self) -> None:
        """is_positive False when post < pre."""
        assert _make(pre=0.50, post=0.45).is_positive is False

    # -----------------------------------------------------------------------
    # confidence_interval_95 — SCENARIO-BENCH-034
    # -----------------------------------------------------------------------

    def test_ci_95_at_50pct_is_within_40_60(self) -> None:
        """SCENARIO-BENCH-034: n=100, p=0.50 → CI within (0.40, 0.60)."""
        r = _make(post=0.50, n=100)
        lo, hi = r.confidence_interval_95
        assert lo > 0.40
        assert hi < 0.60

    def test_ci_95_at_5pct_width_less_than_10pp(self) -> None:
        """SCENARIO-BENCH-034: n=100, p=0.05 → Wilson CI width < 0.10."""
        r = _make(post=0.05, n=100)
        lo, hi = r.confidence_interval_95
        assert (hi - lo) < 0.10

    def test_ci_95_lower_not_negative(self) -> None:
        """CI lower bound is always >= 0.0 (Wilson shrinks at extremes)."""
        r = _make(post=0.01, n=100)
        lo, hi = r.confidence_interval_95
        assert lo >= 0.0

    def test_ci_95_upper_not_above_one(self) -> None:
        """CI upper bound is always <= 1.0."""
        r = _make(post=0.99, n=100)
        lo, hi = r.confidence_interval_95
        assert hi <= 1.0

    def test_ci_95_lower_le_post_accuracy(self) -> None:
        """CI lower <= post_accuracy <= CI upper."""
        r = _make(post=0.60, n=100)
        lo, hi = r.confidence_interval_95
        assert lo <= r.post_accuracy <= hi

    def test_ci_95_with_n1_does_not_crash(self) -> None:
        """n=1 extreme case: CI is valid, no ZeroDivisionError."""
        r = _make(post=1.0, n=1)
        lo, hi = r.confidence_interval_95
        assert 0.0 <= lo <= hi <= 1.0

    def test_ci_95_narrows_with_more_questions(self) -> None:
        """Wider at n=50, narrower at n=100, narrower still at n=500."""
        p = 0.50
        width_50 = _make(post=p, n=50).confidence_interval_95
        width_100 = _make(post=p, n=100).confidence_interval_95
        width_500 = _make(post=p, n=500).confidence_interval_95
        assert (width_50[1] - width_50[0]) > (width_100[1] - width_100[0])
        assert (width_100[1] - width_100[0]) > (width_500[1] - width_500[0])

    # -----------------------------------------------------------------------
    # to_dict
    # -----------------------------------------------------------------------

    def test_to_dict_contains_required_fields(self) -> None:
        """to_dict() contains all required fields for the artifact schema."""
        r = _make(pre=0.40, post=0.50, n=100, extractor="vericot", mode="live_gpu")
        d = r.to_dict()

        required = {
            "model_id", "pre_accuracy", "post_accuracy", "n_questions",
            "extractor_used", "inference_mode", "signed_improvement",
            "confidence_interval_95", "is_positive",
        }
        assert required.issubset(d.keys())

    def test_to_dict_confidence_interval_is_list(self) -> None:
        """confidence_interval_95 is a list [lo, hi] for JSON serialization."""
        d = _make().to_dict()
        assert isinstance(d["confidence_interval_95"], list)
        assert len(d["confidence_interval_95"]) == 2

    def test_to_dict_signed_improvement_matches_property(self) -> None:
        """to_dict() signed_improvement equals the property value."""
        r = _make(pre=0.40, post=0.45)
        d = r.to_dict()
        assert abs(d["signed_improvement"] - r.signed_improvement) < 1e-12

    def test_to_dict_is_positive_matches_property(self) -> None:
        """to_dict() is_positive equals the property value."""
        r = _make(pre=0.40, post=0.45)
        d = r.to_dict()
        assert d["is_positive"] == r.is_positive

    def test_to_dict_values_are_json_serializable(self) -> None:
        """to_dict() can be serialized to JSON without error."""
        import json
        r = _make()
        json.dumps(r.to_dict())  # must not raise

    def test_to_dict_inference_mode_preserved(self) -> None:
        """inference_mode is preserved in to_dict()."""
        r = _make(mode="synthetic")
        assert r.to_dict()["inference_mode"] == "synthetic"

    def test_to_dict_extractor_used_preserved(self) -> None:
        """extractor_used is preserved in to_dict()."""
        r = _make(extractor="none")
        assert r.to_dict()["extractor_used"] == "none"

    # -----------------------------------------------------------------------
    # Edge cases
    # -----------------------------------------------------------------------

    def test_perfect_accuracy(self) -> None:
        """post_accuracy=1.0 — is_positive True when pre < 1.0."""
        r = _make(pre=0.90, post=1.0, n=100)
        assert r.is_positive is True
        lo, hi = r.confidence_interval_95
        assert 0.0 <= lo <= 1.0

    def test_zero_accuracy(self) -> None:
        """post_accuracy=0.0 — signed_improvement negative when pre > 0."""
        r = _make(pre=0.50, post=0.0, n=100)
        assert r.is_positive is False
        assert r.signed_improvement < 0.0
        lo, hi = r.confidence_interval_95
        assert lo >= 0.0
