"""Tests for Precision100qV6Result — 100% coverage on the new module.

Spec: REQ-BENCH-043, REQ-BENCH-044, REQ-BENCH-045,
      SCENARIO-BENCH-062, SCENARIO-BENCH-063, SCENARIO-BENCH-064
"""

from __future__ import annotations

import pytest

from carnot.pipeline.precision_100q_v6_result import Precision100qV6Result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_result(
    pre: float = 0.60,
    post: float = 0.65,
    n: int = 100,
    gpu_id: int = 0,
    extractor_used: str = "vericot",
    inference_mode: str = "live_gpu",
) -> Precision100qV6Result:
    return Precision100qV6Result(
        model_id="Gemma4-INT4",
        pre_accuracy=pre,
        post_accuracy=post,
        n=n,
        extractor_used=extractor_used,
        inference_mode=inference_mode,
        gpu_id=gpu_id,
    )


# ---------------------------------------------------------------------------
# REQ-BENCH-043: ci_95_wilson width < 0.10 at n=100
# SCENARIO-BENCH-064
# ---------------------------------------------------------------------------


def test_ci_95_wilson_width_at_n100():
    """REQ-BENCH-043: Wilson CI width < 0.10 at n=100, p=0.05."""
    result = _make_result(post=0.05, n=100)
    lo, hi = result.ci_95_wilson
    width = hi - lo
    assert width < 0.10, f"Expected width < 0.10 but got {width:.4f}"


def test_ci_95_wilson_width_midrange_n100():
    """Wilson CI width at p=0.50, n=100 is ~0.19 — verify it's within the expected range.

    At p=0.50 the Wilson CI is maximally wide (~0.19).  The width < 0.10 spec
    (REQ-BENCH-043) specifically applies to extreme p values (e.g. p=0.05) where the
    interval is narrower.  This test confirms the midrange width stays under 0.25 as
    a sanity bound.
    """
    result = _make_result(post=0.50, n=100)
    lo, hi = result.ci_95_wilson
    assert hi - lo < 0.25


def test_ci_95_wilson_bounds_clamped():
    """Wilson CI is always within [0.0, 1.0]."""
    # Extreme low probability
    r_low = _make_result(post=0.0, n=100)
    lo, hi = r_low.ci_95_wilson
    assert lo >= 0.0
    assert hi <= 1.0

    # Extreme high probability
    r_high = _make_result(post=1.0, n=100)
    lo, hi = r_high.ci_95_wilson
    assert lo >= 0.0
    assert hi <= 1.0


def test_ci_95_wilson_lower_lt_upper():
    """Lower bound must be strictly less than upper bound for non-degenerate n."""
    result = _make_result(post=0.65, n=100)
    lo, hi = result.ci_95_wilson
    assert lo < hi


# ---------------------------------------------------------------------------
# REQ-BENCH-043: is_positive=True when signed_improvement > 0
# SCENARIO-BENCH-064
# ---------------------------------------------------------------------------


def test_is_positive_true_when_positive_improvement():
    """REQ-BENCH-043: is_positive=True when signed_improvement=0.05, n=100."""
    result = _make_result(pre=0.60, post=0.65)
    assert result.is_positive is True


def test_is_positive_false_when_zero_improvement():
    """is_positive=False when pre_accuracy == post_accuracy (no improvement)."""
    result = _make_result(pre=0.65, post=0.65)
    assert result.is_positive is False


def test_is_positive_false_when_regression():
    """is_positive=False when post < pre (pipeline regression)."""
    result = _make_result(pre=0.70, post=0.65)
    assert result.is_positive is False


# ---------------------------------------------------------------------------
# signed_improvement
# ---------------------------------------------------------------------------


def test_signed_improvement_positive():
    result = _make_result(pre=0.60, post=0.65)
    assert abs(result.signed_improvement - 0.05) < 1e-9


def test_signed_improvement_zero():
    result = _make_result(pre=0.65, post=0.65)
    assert result.signed_improvement == 0.0


def test_signed_improvement_negative():
    result = _make_result(pre=0.70, post=0.60)
    assert abs(result.signed_improvement - (-0.10)) < 1e-9


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


def test_to_dict_contains_all_keys():
    """SCENARIO-BENCH-064: to_dict contains all required schema keys."""
    result = _make_result()
    d = result.to_dict()
    required_keys = {
        "model_id",
        "pre_accuracy",
        "post_accuracy",
        "n",
        "extractor_used",
        "inference_mode",
        "gpu_id",
        "signed_improvement",
        "ci_95_wilson",
        "is_positive",
    }
    assert required_keys.issubset(d.keys())


def test_to_dict_ci_95_wilson_is_list():
    """ci_95_wilson must be a two-element list (JSON-serializable, not tuple)."""
    result = _make_result()
    d = result.to_dict()
    assert isinstance(d["ci_95_wilson"], list)
    assert len(d["ci_95_wilson"]) == 2


def test_to_dict_is_positive_reflects_improvement():
    r_pos = _make_result(pre=0.60, post=0.65)
    assert r_pos.to_dict()["is_positive"] is True

    r_neg = _make_result(pre=0.65, post=0.60)
    assert r_neg.to_dict()["is_positive"] is False


def test_to_dict_gpu_id_recorded():
    """REQ-BENCH-044: gpu_id is embedded in the artifact dict for auditability."""
    r0 = _make_result(gpu_id=0)
    assert r0.to_dict()["gpu_id"] == 0

    r1 = _make_result(gpu_id=1)
    assert r1.to_dict()["gpu_id"] == 1


# ---------------------------------------------------------------------------
# n=1 edge case (guard against division by zero in ci formula)
# ---------------------------------------------------------------------------


def test_ci_n1_does_not_raise():
    """ci_95_wilson must not raise even for n=1 (guard against div-by-zero)."""
    result = _make_result(post=0.5, n=1)
    lo, hi = result.ci_95_wilson
    assert 0.0 <= lo <= hi <= 1.0


# ---------------------------------------------------------------------------
# inference_mode and extractor_used passthrough
# ---------------------------------------------------------------------------


def test_inference_mode_passthrough():
    result = _make_result(inference_mode="synthetic")
    assert result.to_dict()["inference_mode"] == "synthetic"


def test_extractor_used_passthrough():
    result = _make_result(extractor_used="none")
    assert result.to_dict()["extractor_used"] == "none"
