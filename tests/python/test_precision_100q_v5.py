"""Tests for Precision100qV5Result (Exp 488).

Every test traces to a spec requirement or scenario:
    REQ-BENCH-034, REQ-BENCH-035, REQ-BENCH-036
    SCENARIO-BENCH-053, SCENARIO-BENCH-054, SCENARIO-BENCH-055
"""

from __future__ import annotations

import json

import pytest

from carnot.pipeline.precision_100q_v5_result import Precision100qV5Result


def _make(pre: float = 0.60, post: float = 0.65, n: int = 100, gpu_id: int = 0) -> Precision100qV5Result:
    return Precision100qV5Result(
        model_id="TestModel",
        pre_accuracy=pre,
        post_accuracy=post,
        n=n,
        extractor_used="vericot,vprm",
        inference_mode="live_gpu",
        gpu_id=gpu_id,
    )


class TestPrecision100qV5ResultBasic:
    """REQ-BENCH-034, SCENARIO-BENCH-055"""

    def test_signed_improvement_positive(self):
        r = _make(pre=0.60, post=0.65)
        assert abs(r.signed_improvement - 0.05) < 1e-9

    def test_signed_improvement_negative(self):
        r = _make(pre=0.70, post=0.65)
        assert r.signed_improvement < 0

    def test_signed_improvement_zero(self):
        r = _make(pre=0.65, post=0.65)
        assert r.signed_improvement == 0.0

    def test_is_positive_true_when_improvement(self):
        # REQ-BENCH-034: is_positive=True when signed_improvement=0.10, n=100
        r = _make(pre=0.60, post=0.70)
        assert r.is_positive is True

    def test_is_positive_false_when_zero(self):
        r = _make(pre=0.65, post=0.65)
        assert r.is_positive is False

    def test_is_positive_false_when_negative(self):
        r = _make(pre=0.70, post=0.65)
        assert r.is_positive is False


class TestPrecision100qV5ResultWilsonCI:
    """REQ-BENCH-034: CI width < 0.10 at n=100, p=0.05"""

    def test_ci_95_wilson_returns_tuple(self):
        r = _make()
        lo, hi = r.ci_95_wilson
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_ci_95_wilson_ordered(self):
        r = _make()
        lo, hi = r.ci_95_wilson
        assert lo < hi

    def test_ci_95_wilson_within_unit_interval(self):
        r = _make()
        lo, hi = r.ci_95_wilson
        assert 0.0 <= lo <= 1.0
        assert 0.0 <= hi <= 1.0

    def test_ci_95_wilson_width_lt_010_at_n100_p005(self):
        # REQ-BENCH-034: at n=100, p=0.05 the width must be < 0.10
        r = _make(post=0.05, n=100)
        lo, hi = r.ci_95_wilson
        assert (hi - lo) < 0.10

    def test_ci_95_wilson_at_extreme_zero(self):
        r = _make(pre=0.0, post=0.0, n=100)
        lo, hi = r.ci_95_wilson
        assert lo >= 0.0
        assert hi >= 0.0

    def test_ci_95_wilson_at_extreme_one(self):
        r = _make(pre=1.0, post=1.0, n=100)
        lo, hi = r.ci_95_wilson
        assert lo <= 1.0
        assert hi <= 1.0

    def test_ci_95_wilson_n_guard_no_zerodiv(self):
        # n=0 must not raise ZeroDivisionError (guarded by max(n,1))
        r = Precision100qV5Result(
            model_id="m", pre_accuracy=0.5, post_accuracy=0.5,
            n=0, extractor_used="none", inference_mode="synthetic", gpu_id=0,
        )
        lo, hi = r.ci_95_wilson
        assert 0.0 <= lo <= hi <= 1.0


class TestPrecision100qV5ResultToDict:
    """SCENARIO-BENCH-055: to_dict() returns all required fields"""

    def test_to_dict_keys_present(self):
        r = _make()
        d = r.to_dict()
        for key in ("model_id", "pre_accuracy", "post_accuracy", "n",
                    "extractor_used", "inference_mode", "gpu_id",
                    "signed_improvement", "ci_95_wilson", "is_positive"):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_gpu_id_recorded(self):
        # REQ-BENCH-035: gpu_id must be present in to_dict() for auditability
        r = _make(gpu_id=1)
        assert r.to_dict()["gpu_id"] == 1

    def test_to_dict_ci_95_wilson_is_list(self):
        # JSON serialisability requires a list, not a tuple
        r = _make()
        d = r.to_dict()
        assert isinstance(d["ci_95_wilson"], list)
        assert len(d["ci_95_wilson"]) == 2

    def test_to_dict_is_json_serializable(self):
        r = _make()
        json.dumps(r.to_dict())  # must not raise

    def test_to_dict_values_match_properties(self):
        r = _make(pre=0.60, post=0.70, n=100)
        d = r.to_dict()
        assert d["signed_improvement"] == pytest.approx(r.signed_improvement)
        assert d["is_positive"] == r.is_positive
        lo, hi = r.ci_95_wilson
        assert d["ci_95_wilson"][0] == pytest.approx(lo)
        assert d["ci_95_wilson"][1] == pytest.approx(hi)
