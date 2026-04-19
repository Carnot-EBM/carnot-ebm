"""Tests for Live200qV4Result — 100% coverage for Exp 503 result class.

Spec: REQ-BENCH-046, REQ-BENCH-047, REQ-BENCH-048,
      SCENARIO-BENCH-065, SCENARIO-BENCH-066, SCENARIO-BENCH-067
"""

import pytest

from carnot.pipeline.live_200q_v4_result import Live200qV4Result

_Z95 = 1.959963984540054


def _make(pre_acc: float, post_acc: float, n: int = 200) -> Live200qV4Result:
    return Live200qV4Result(
        model_id="test-model",
        pre_acc=pre_acc,
        post_acc=post_acc,
        n=n,
        extractor_name="VeriCoT+VPRM",
        inference_mode="live_gpu",
    )


# ---------------------------------------------------------------------------
# REQ-BENCH-046: signed_improvement
# ---------------------------------------------------------------------------


class TestSignedImprovement:
    def test_positive(self):
        # SCENARIO-BENCH-065: pipeline improves accuracy
        r = _make(pre_acc=0.70, post_acc=0.80)
        assert abs(r.signed_improvement - 0.10) < 1e-9

    def test_negative(self):
        # Honest negative results must not be clamped (CLAUDE.md requirement)
        r = _make(pre_acc=0.80, post_acc=0.70)
        assert abs(r.signed_improvement - (-0.10)) < 1e-9

    def test_zero(self):
        r = _make(pre_acc=0.75, post_acc=0.75)
        assert r.signed_improvement == 0.0


# ---------------------------------------------------------------------------
# REQ-BENCH-047: ci_95_wilson width < 0.07 at n=200, p near boundary
# SCENARIO-BENCH-065
# ---------------------------------------------------------------------------


class TestCi95Wilson:
    def test_bounds_in_01(self):
        # CI must stay in [0, 1]
        r = _make(pre_acc=0.5, post_acc=0.5)
        lo, hi = r.ci_95_wilson
        assert 0.0 <= lo <= hi <= 1.0

    def test_width_less_than_007_at_n200_p005(self):
        # REQ-BENCH-047: width < 0.07 at n=200, p=0.05 (near-boundary case)
        # Wilson CI at p=0.10 has full width ~0.084; at p=0.05 it shrinks to ~0.062
        r = _make(pre_acc=0.5, post_acc=0.05, n=200)
        lo, hi = r.ci_95_wilson
        assert (hi - lo) < 0.07, f"width={hi - lo:.4f} should be < 0.07"

    def test_width_less_than_007_at_n200_p095(self):
        r = _make(pre_acc=0.5, post_acc=0.95, n=200)
        lo, hi = r.ci_95_wilson
        assert (hi - lo) < 0.07, f"width={hi - lo:.4f} should be < 0.07"

    def test_center_near_post_acc_for_large_n(self):
        # For n=200 the Wilson center is close to the sample proportion
        r = _make(pre_acc=0.5, post_acc=0.70, n=200)
        lo, hi = r.ci_95_wilson
        center = (lo + hi) / 2
        assert abs(center - 0.70) < 0.05

    def test_n1_no_crash(self):
        # Guard against division by zero when n is tiny
        r = Live200qV4Result(
            model_id="x", pre_acc=0.5, post_acc=0.5, n=1,
            extractor_name="e", inference_mode="live_gpu",
        )
        lo, hi = r.ci_95_wilson
        assert 0.0 <= lo <= hi <= 1.0

    def test_post_acc_zero(self):
        # Boundary: p=0 should not produce negative CI
        r = _make(pre_acc=0.0, post_acc=0.0)
        lo, hi = r.ci_95_wilson
        assert lo >= 0.0

    def test_post_acc_one(self):
        # Boundary: p=1 should not produce CI > 1
        r = _make(pre_acc=1.0, post_acc=1.0)
        lo, hi = r.ci_95_wilson
        assert hi <= 1.0


# ---------------------------------------------------------------------------
# REQ-BENCH-047: is_statistically_positive = (lower_wald_bound > 0)
# SCENARIO-BENCH-066
# ---------------------------------------------------------------------------


class TestIsStatisticallyPositive:
    def test_small_improvement_not_significant_at_n200(self):
        # SCENARIO-BENCH-066: signed_improvement=0.02, n=200 → NOT significant
        # SE ≈ 0.045, lower = 0.02 - 1.96*0.045 ≈ -0.069 < 0
        r = _make(pre_acc=0.70, post_acc=0.72, n=200)
        assert r.is_statistically_positive is False

    def test_large_improvement_is_significant(self):
        # 30pp improvement at n=200: lower = 0.30 - 1.96*SE > 0
        r = _make(pre_acc=0.50, post_acc=0.80, n=200)
        assert r.is_statistically_positive is True

    def test_zero_improvement_not_significant(self):
        r = _make(pre_acc=0.70, post_acc=0.70, n=200)
        assert r.is_statistically_positive is False

    def test_negative_improvement_not_significant(self):
        r = _make(pre_acc=0.80, post_acc=0.70, n=200)
        assert r.is_statistically_positive is False

    def test_large_n_makes_small_improvement_significant(self):
        # With n=1_000_000 a 0.01 improvement is clearly significant
        r = _make(pre_acc=0.700, post_acc=0.710, n=1_000_000)
        assert r.is_statistically_positive is True

    def test_boundary_strict_greater_than_zero(self):
        # lower_ci_bound > 0.0 (strict), so zero itself must be False
        # With n=200 and a small improvement the lower bound is negative
        r = _make(pre_acc=0.700, post_acc=0.701, n=200)
        assert r.is_statistically_positive is False


# ---------------------------------------------------------------------------
# REQ-BENCH-048: to_dict — all fields present and correct types
# SCENARIO-BENCH-067
# ---------------------------------------------------------------------------


class TestToDict:
    _REQUIRED_KEYS = (
        "model_id", "pre_acc", "post_acc", "n",
        "extractor_name", "inference_mode",
        "signed_improvement", "ci_95_wilson", "is_statistically_positive",
    )

    def test_all_keys_present(self):
        r = _make(pre_acc=0.70, post_acc=0.75)
        d = r.to_dict()
        for key in self._REQUIRED_KEYS:
            assert key in d, f"missing key: {key}"

    def test_ci_95_wilson_is_list_of_two(self):
        r = _make(pre_acc=0.70, post_acc=0.75)
        d = r.to_dict()
        assert isinstance(d["ci_95_wilson"], list)
        assert len(d["ci_95_wilson"]) == 2

    def test_values_match_properties(self):
        r = _make(pre_acc=0.60, post_acc=0.75)
        d = r.to_dict()
        assert d["signed_improvement"] == pytest.approx(r.signed_improvement)
        assert d["ci_95_wilson"][0] == pytest.approx(r.ci_95_wilson[0])
        assert d["ci_95_wilson"][1] == pytest.approx(r.ci_95_wilson[1])
        assert d["is_statistically_positive"] == r.is_statistically_positive

    def test_model_id_preserved(self):
        r = _make(pre_acc=0.5, post_acc=0.6)
        assert r.to_dict()["model_id"] == "test-model"

    def test_inference_mode_preserved(self):
        r = _make(pre_acc=0.5, post_acc=0.6)
        assert r.to_dict()["inference_mode"] == "live_gpu"

    def test_extractor_name_preserved(self):
        r = _make(pre_acc=0.5, post_acc=0.6)
        assert r.to_dict()["extractor_name"] == "VeriCoT+VPRM"

    def test_n_preserved(self):
        r = _make(pre_acc=0.5, post_acc=0.6, n=200)
        assert r.to_dict()["n"] == 200
