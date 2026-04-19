"""Tests for live_200q_v7_helpers — 100% coverage for Exp 528 new helpers.

Spec: REQ-BENCH-019, SCENARIO-BENCH-041, SCENARIO-BENCH-042
"""

from __future__ import annotations

import pytest

from carnot.pipeline.live_200q_v7_helpers import (
    build_200q_v7_artifact,
    compute_wilson_ci,
)


# ---------------------------------------------------------------------------
# compute_wilson_ci
# ---------------------------------------------------------------------------


class TestComputeWilsonCi:
    """REQ-BENCH-019: Wilson 95% CI lower bound > 0 is the publishable-claim gate."""

    def test_zero_total_returns_zero_zero(self):
        # SCENARIO-BENCH-041: guard against division by zero
        lo, hi = compute_wilson_ci(0, 0)
        assert lo == 0.0
        assert hi == 0.0

    def test_all_correct_upper_bound_one(self):
        # p=1.0 should not produce an upper bound > 1
        lo, hi = compute_wilson_ci(100, 100)
        assert hi <= 1.0
        assert lo >= 0.0

    def test_all_wrong_lower_bound_zero(self):
        # p=0.0 should not produce a negative lower bound
        lo, hi = compute_wilson_ci(0, 200)
        assert lo >= 0.0

    def test_bounds_in_unit_interval(self):
        lo, hi = compute_wilson_ci(80, 200)
        assert 0.0 <= lo <= hi <= 1.0

    def test_center_near_sample_proportion(self):
        # Wilson center should be close to the MLE proportion for large n
        lo, hi = compute_wilson_ci(140, 200)
        center = (lo + hi) / 2.0
        assert abs(center - 0.70) < 0.05

    def test_width_shrinks_with_larger_n(self):
        lo_small, hi_small = compute_wilson_ci(50, 100)
        lo_big, hi_big = compute_wilson_ci(100, 200)
        assert (hi_big - lo_big) < (hi_small - lo_small)

    def test_returns_tuple_of_floats(self):
        result = compute_wilson_ci(50, 100)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_nonstandard_confidence_fallback(self):
        # When scipy is absent the helper falls back to z=1.96; should still return valid CI
        lo, hi = compute_wilson_ci(80, 200, confidence=0.90)
        assert 0.0 <= lo <= hi <= 1.0

    def test_default_confidence_is_0_95(self):
        # Default call should equal explicit confidence=0.95
        lo_default, hi_default = compute_wilson_ci(80, 200)
        lo_explicit, hi_explicit = compute_wilson_ci(80, 200, confidence=0.95)
        assert abs(lo_default - lo_explicit) < 1e-9
        assert abs(hi_default - hi_explicit) < 1e-9


# ---------------------------------------------------------------------------
# build_200q_v7_artifact
# ---------------------------------------------------------------------------


class TestBuild200qV7Artifact:
    """REQ-BENCH-019: All schema fields present and RETRO-038 logic correct."""

    _REQUIRED_KEYS = (
        "schema",
        "inference_mode",
        "n_questions",
        "baseline_accuracy",
        "pipeline_accuracy",
        "signed_improvement",
        "wilson_95ci_lower",
        "wilson_95ci_upper",
        "is_statistically_positive",
        "retro_038_closed",
        "cot_pairs_written",
        "honest_verdict",
    )

    def _live_results(self, ci_lower: float = 0.05, ci_upper: float = 0.25) -> dict:
        return {
            "n_questions": 200,
            "baseline_accuracy": 0.70,
            "pipeline_accuracy": 0.85,
            "wilson_95ci_lower": ci_lower,
            "wilson_95ci_upper": ci_upper,
        }

    # -- All required schema keys present -----------------------------------

    def test_all_required_keys_present_live(self):
        # SCENARIO-BENCH-041: live path must include every field
        art = build_200q_v7_artifact(self._live_results(), "live_gpu", "results/exp528_cot_pairs.json")
        for key in self._REQUIRED_KEYS:
            assert key in art, f"missing key: {key}"

    def test_all_required_keys_present_deferred(self):
        # SCENARIO-BENCH-042: gpu_required path must also include every field
        art = build_200q_v7_artifact({}, "gpu_required", None)
        for key in self._REQUIRED_KEYS:
            assert key in art, f"missing key: {key}"

    # -- Schema -----------------------------------------------------------

    def test_schema_field(self):
        art = build_200q_v7_artifact(self._live_results(), "live_gpu", None)
        assert art["schema"] == "carnot.live_200q.v7"

    # -- Deferred (gpu_required) path ------------------------------------

    def test_deferred_honest_verdict(self):
        # SCENARIO-BENCH-042: no GPU → gpu_required verdict
        art = build_200q_v7_artifact({}, "gpu_required", None)
        assert art["honest_verdict"] == "gpu_required"
        assert art["retro_038_closed"] is False
        assert art["is_statistically_positive"] is False

    def test_deferred_cot_pairs_null(self):
        art = build_200q_v7_artifact({}, "gpu_required", None)
        assert art["cot_pairs_written"] is None

    def test_deferred_defaults_to_zero(self):
        art = build_200q_v7_artifact({}, "gpu_required", None)
        assert art["n_questions"] == 0
        assert art["baseline_accuracy"] == 0.0
        assert art["pipeline_accuracy"] == 0.0

    # -- Live path with significance (RETRO-038 closed) -----------------

    def test_first_publishable_claim_verdict(self):
        # SCENARIO-BENCH-041: ci_lower > 0 AND live_gpu → first_publishable_claim
        art = build_200q_v7_artifact(self._live_results(ci_lower=0.01), "live_gpu", "p.json")
        assert art["honest_verdict"] == "first_publishable_claim"
        assert art["retro_038_closed"] is True
        assert art["is_statistically_positive"] is True

    def test_cot_pairs_written_captured(self):
        art = build_200q_v7_artifact(self._live_results(ci_lower=0.01), "live_gpu", "results/exp528_cot_pairs.json")
        assert art["cot_pairs_written"] == "results/exp528_cot_pairs.json"

    def test_n_questions_captured(self):
        art = build_200q_v7_artifact(self._live_results(), "live_gpu", None)
        assert art["n_questions"] == 200

    # -- Live path without significance ---------------------------------

    def test_live_no_significance_verdict(self):
        # ci_lower <= 0 even with live_gpu → live_no_significance, RETRO-038 stays open
        art = build_200q_v7_artifact(self._live_results(ci_lower=-0.01), "live_gpu", None)
        assert art["honest_verdict"] == "live_no_significance"
        assert art["retro_038_closed"] is False
        assert art["is_statistically_positive"] is False

    def test_ci_lower_exactly_zero_not_significant(self):
        # The criterion is STRICTLY > 0; zero itself is not sufficient
        art = build_200q_v7_artifact(self._live_results(ci_lower=0.0), "live_gpu", None)
        assert art["is_statistically_positive"] is False

    # -- Arithmetic correctness -----------------------------------------

    def test_signed_improvement_computed_correctly(self):
        results = {
            "n_questions": 200,
            "baseline_accuracy": 0.60,
            "pipeline_accuracy": 0.75,
            "wilson_95ci_lower": 0.05,
            "wilson_95ci_upper": 0.25,
        }
        art = build_200q_v7_artifact(results, "live_gpu", None)
        assert abs(art["signed_improvement"] - 0.15) < 1e-9

    def test_signed_improvement_can_be_negative(self):
        # Honest reporting: regression must not be clamped to zero
        results = {
            "n_questions": 200,
            "baseline_accuracy": 0.80,
            "pipeline_accuracy": 0.70,
            "wilson_95ci_lower": -0.15,
            "wilson_95ci_upper": -0.05,
        }
        art = build_200q_v7_artifact(results, "live_gpu", None)
        assert art["signed_improvement"] < 0.0
