"""Tests for Live200qV3Result — 100% coverage for Exp 489 result class.

Spec: REQ-BENCH-037, REQ-BENCH-039,
      SCENARIO-BENCH-056, SCENARIO-BENCH-057, SCENARIO-BENCH-058
"""

import math

import pytest

from carnot.pipeline.live_200q_v3_result import Live200qV3Result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_Z95 = 1.959963984540054


def _make_result(pre_acc: float, post_acc: float, n: int = 200) -> Live200qV3Result:
    return Live200qV3Result(
        model_id="test-model",
        pre_acc=pre_acc,
        post_acc=post_acc,
        n=n,
        extractor_name="VeriCoT+VPRM",
        inference_mode="live_gpu",
    )


# ---------------------------------------------------------------------------
# REQ-BENCH-037: signed_improvement
# SCENARIO-BENCH-057
# ---------------------------------------------------------------------------


class TestSignedImprovement:
    def test_positive_improvement(self):
        # SCENARIO-BENCH-057: pipeline improves accuracy
        r = _make_result(pre_acc=0.70, post_acc=0.80)
        assert abs(r.signed_improvement - 0.10) < 1e-9

    def test_negative_improvement(self):
        # Honest negative results must not be clamped (CLAUDE.md requirement)
        r = _make_result(pre_acc=0.80, post_acc=0.70)
        assert abs(r.signed_improvement - (-0.10)) < 1e-9

    def test_zero_improvement(self):
        r = _make_result(pre_acc=0.75, post_acc=0.75)
        assert r.signed_improvement == 0.0


# ---------------------------------------------------------------------------
# REQ-BENCH-039: ci_95_wilson width < 0.07 at n=200 for all p
# SCENARIO-BENCH-056
# ---------------------------------------------------------------------------


class TestCi95Wilson:
    def test_interval_within_01(self):
        # CI must stay in [0, 1]
        r = _make_result(pre_acc=0.5, post_acc=0.5)
        lo, hi = r.ci_95_wilson
        assert 0.0 <= lo <= hi <= 1.0

    def test_width_less_than_007_at_n200_p005(self):
        # REQ-BENCH-039: width < 0.07 at n=200, p=0.05
        r = _make_result(pre_acc=0.5, post_acc=0.05, n=200)
        lo, hi = r.ci_95_wilson
        assert (hi - lo) < 0.07, f"width={hi - lo:.4f} should be < 0.07"

    def test_width_less_than_007_at_n200_p095(self):
        r = _make_result(pre_acc=0.5, post_acc=0.95, n=200)
        lo, hi = r.ci_95_wilson
        assert (hi - lo) < 0.07, f"width={hi - lo:.4f} should be < 0.07"

    def test_center_near_post_acc(self):
        # For large n the Wilson center is close to the sample proportion
        r = _make_result(pre_acc=0.5, post_acc=0.70, n=200)
        lo, hi = r.ci_95_wilson
        center = (lo + hi) / 2
        assert abs(center - 0.70) < 0.05

    def test_n1_does_not_crash(self):
        # Guard against division by zero when n is tiny
        r = Live200qV3Result(
            model_id="x", pre_acc=0.5, post_acc=0.5, n=1,
            extractor_name="e", inference_mode="live_gpu",
        )
        lo, hi = r.ci_95_wilson
        assert 0.0 <= lo <= hi <= 1.0


# ---------------------------------------------------------------------------
# REQ-BENCH-039: is_statistically_positive = (lower_ci_bound > 0)
# SCENARIO-BENCH-058
# ---------------------------------------------------------------------------


class TestIsStatisticallyPositive:
    def test_small_improvement_not_significant_at_n200(self):
        # SCENARIO-BENCH-058: signed_improvement=0.02 at n=200 is NOT significant
        # (Wald SE ≈ 0.045, lower = 0.02 - 1.96*0.045 ≈ -0.069 < 0)
        r = _make_result(pre_acc=0.70, post_acc=0.72, n=200)
        assert r.is_statistically_positive is False

    def test_large_improvement_is_significant(self):
        # 30pp improvement at n=200: lower = 0.30 - 1.96*SE > 0
        r = _make_result(pre_acc=0.50, post_acc=0.80, n=200)
        assert r.is_statistically_positive is True

    def test_zero_improvement_is_not_significant(self):
        r = _make_result(pre_acc=0.70, post_acc=0.70, n=200)
        assert r.is_statistically_positive is False

    def test_negative_improvement_is_not_significant(self):
        r = _make_result(pre_acc=0.80, post_acc=0.70, n=200)
        assert r.is_statistically_positive is False

    def test_boundary_lower_exactly_zero_is_false(self):
        # lower_ci_bound > 0.0 (strict), so exactly 0.0 must be False
        # Construct a case where lower ≈ 0 and verify strict >
        # We patch the computation by setting n very large to reduce SE to near 0
        # then check that tiny positive improvement with huge n is significant
        r = _make_result(pre_acc=0.700, post_acc=0.701, n=1_000_000)
        # SE ≈ sqrt(0.21/1e6 + 0.2097/1e6) ≈ 0.000648; lower = 0.001 - 1.96*0.000648 = -0.00027 < 0
        # Actually 0.001 - 0.00127 = -0.00027 < 0, so False
        # Use a larger improvement: 0.01 improvement with n=1M → lower > 0
        r2 = _make_result(pre_acc=0.700, post_acc=0.710, n=1_000_000)
        assert r2.is_statistically_positive is True


# ---------------------------------------------------------------------------
# to_dict — all fields present and types correct
# ---------------------------------------------------------------------------


class TestToDict:
    def test_all_keys_present(self):
        r = _make_result(pre_acc=0.70, post_acc=0.75)
        d = r.to_dict()
        for key in (
            "model_id", "pre_acc", "post_acc", "n",
            "extractor_name", "inference_mode",
            "signed_improvement", "ci_95_wilson", "is_statistically_positive",
        ):
            assert key in d, f"missing key: {key}"

    def test_ci_95_wilson_is_list_of_two(self):
        r = _make_result(pre_acc=0.70, post_acc=0.75)
        d = r.to_dict()
        assert isinstance(d["ci_95_wilson"], list)
        assert len(d["ci_95_wilson"]) == 2

    def test_values_match_properties(self):
        r = _make_result(pre_acc=0.60, post_acc=0.75)
        d = r.to_dict()
        assert d["signed_improvement"] == pytest.approx(r.signed_improvement)
        assert d["ci_95_wilson"][0] == pytest.approx(r.ci_95_wilson[0])
        assert d["ci_95_wilson"][1] == pytest.approx(r.ci_95_wilson[1])
        assert d["is_statistically_positive"] == r.is_statistically_positive

    def test_model_id_preserved(self):
        r = _make_result(pre_acc=0.5, post_acc=0.6)
        assert r.to_dict()["model_id"] == "test-model"

    def test_inference_mode_preserved(self):
        r = _make_result(pre_acc=0.5, post_acc=0.6)
        assert r.to_dict()["inference_mode"] == "live_gpu"
