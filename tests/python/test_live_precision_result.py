"""Tests for LivePrecisionResult — signed-improvement data class for Exp 451.

Verifies:
  - signed_improvement = post_accuracy - pre_accuracy (exact arithmetic)
  - is_positive = True when post > pre (strict greater-than)
  - is_positive = False when post == pre (zero improvement is NOT positive)
  - is_positive = False when post < pre (regression)
  - to_dict() returns all six expected keys with correct values
  - Dataclass field assignment and attribute access

Spec: REQ-BENCH-013, SCENARIO-BENCH-031, SCENARIO-BENCH-032
"""

from __future__ import annotations

import pytest

from carnot.pipeline.live_precision_result import LivePrecisionResult


class TestSignedImprovement:
    """REQ-BENCH-013: signed_improvement = post - pre, unclamped."""

    def test_positive_improvement(self) -> None:
        """REQ-BENCH-013: post > pre yields positive signed_improvement."""
        r = LivePrecisionResult(model_id="TestModel", pre_accuracy=0.60, post_accuracy=0.75)
        assert abs(r.signed_improvement - 0.15) < 1e-9

    def test_zero_improvement(self) -> None:
        """REQ-BENCH-013: post == pre yields signed_improvement == 0.0."""
        r = LivePrecisionResult(model_id="TestModel", pre_accuracy=0.50, post_accuracy=0.50)
        assert r.signed_improvement == pytest.approx(0.0)

    def test_negative_improvement(self) -> None:
        """REQ-BENCH-013: post < pre yields negative signed_improvement (honest regression)."""
        r = LivePrecisionResult(model_id="TestModel", pre_accuracy=0.80, post_accuracy=0.70)
        assert abs(r.signed_improvement - (-0.10)) < 1e-9

    def test_full_range_zero_pre(self) -> None:
        """REQ-BENCH-013: pre=0.0 post=1.0 yields signed_improvement=1.0."""
        r = LivePrecisionResult(model_id="M", pre_accuracy=0.0, post_accuracy=1.0)
        assert r.signed_improvement == pytest.approx(1.0)

    def test_full_range_zero_post(self) -> None:
        """REQ-BENCH-013: pre=1.0 post=0.0 yields signed_improvement=-1.0."""
        r = LivePrecisionResult(model_id="M", pre_accuracy=1.0, post_accuracy=0.0)
        assert r.signed_improvement == pytest.approx(-1.0)


class TestIsPositive:
    """REQ-BENCH-013 / SCENARIO-BENCH-031: is_positive semantics."""

    def test_true_when_post_greater_than_pre(self) -> None:
        """SCENARIO-BENCH-031: is_positive=True when post > pre."""
        r = LivePrecisionResult(model_id="M", pre_accuracy=0.70, post_accuracy=0.75)
        assert r.is_positive is True

    def test_false_when_equal(self) -> None:
        """REQ-BENCH-013: zero improvement is NOT positive."""
        r = LivePrecisionResult(model_id="M", pre_accuracy=0.70, post_accuracy=0.70)
        assert r.is_positive is False

    def test_false_when_regression(self) -> None:
        """REQ-BENCH-013: regression (post < pre) is NOT positive."""
        r = LivePrecisionResult(model_id="M", pre_accuracy=0.80, post_accuracy=0.60)
        assert r.is_positive is False

    def test_tiny_positive_improvement(self) -> None:
        """REQ-BENCH-013: even a tiny positive delta (1e-10) counts as positive."""
        r = LivePrecisionResult(model_id="M", pre_accuracy=0.0, post_accuracy=1e-10)
        assert r.is_positive is True

    def test_tiny_negative_improvement(self) -> None:
        """REQ-BENCH-013: tiny regression is not positive."""
        r = LivePrecisionResult(model_id="M", pre_accuracy=1e-10, post_accuracy=0.0)
        assert r.is_positive is False


class TestToDict:
    """LivePrecisionResult.to_dict() — JSON-serializable representation."""

    def test_keys_present(self) -> None:
        """to_dict() returns all six expected keys."""
        r = LivePrecisionResult(model_id="Gemma4", pre_accuracy=0.60, post_accuracy=0.75)
        d = r.to_dict()
        assert set(d.keys()) == {
            "model_id",
            "pre_accuracy",
            "post_accuracy",
            "signed_improvement",
            "is_positive",
        }

    def test_values_correct(self) -> None:
        """to_dict() values match dataclass fields and computed properties."""
        r = LivePrecisionResult(model_id="Qwen", pre_accuracy=0.40, post_accuracy=0.50)
        d = r.to_dict()
        assert d["model_id"] == "Qwen"
        assert d["pre_accuracy"] == pytest.approx(0.40)
        assert d["post_accuracy"] == pytest.approx(0.50)
        assert d["signed_improvement"] == pytest.approx(0.10)
        assert d["is_positive"] is True

    def test_regression_values_in_dict(self) -> None:
        """to_dict() preserves negative signed_improvement (no clamping)."""
        r = LivePrecisionResult(model_id="M", pre_accuracy=0.80, post_accuracy=0.70)
        d = r.to_dict()
        assert d["signed_improvement"] == pytest.approx(-0.10)
        assert d["is_positive"] is False


class TestDataclassFields:
    """LivePrecisionResult dataclass field access."""

    def test_model_id_stored(self) -> None:
        """model_id field stores the provided string."""
        r = LivePrecisionResult(model_id="google/gemma-4-E4B-it", pre_accuracy=0.0, post_accuracy=0.0)
        assert r.model_id == "google/gemma-4-E4B-it"

    def test_pre_and_post_stored(self) -> None:
        """pre_accuracy and post_accuracy fields are stored as-is."""
        r = LivePrecisionResult(model_id="M", pre_accuracy=0.123, post_accuracy=0.456)
        assert r.pre_accuracy == pytest.approx(0.123)
        assert r.post_accuracy == pytest.approx(0.456)
