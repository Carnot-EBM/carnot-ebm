"""Tests for adversarial_v5_result helpers (Exp 516 — RETRO-039 robustness claim).

Every test references the spec requirement it validates, as required by CLAUDE.md.

Spec: REQ-BENCH-052, REQ-BENCH-053,
      SCENARIO-BENCH-037, SCENARIO-BENCH-038
"""

from __future__ import annotations

import pytest

from carnot.pipeline.adversarial_v5_result import (
    build_adversarial_v5_artifact,
    compute_robustness_delta,
)


# ---------------------------------------------------------------------------
# Tests for compute_robustness_delta
# ---------------------------------------------------------------------------


class TestComputeRobustnessDelta:
    """Spec: REQ-BENCH-052, SCENARIO-BENCH-037"""

    def test_positive_when_carnot_degrades_less(self):
        """Baseline drops 0.15, pipeline drops 0.08 -> delta = 0.07 > 0.

        Spec: SCENARIO-BENCH-037
        """
        delta = compute_robustness_delta(
            baseline_std=0.80,
            baseline_adv=0.65,
            pipeline_std=0.78,
            pipeline_adv=0.70,
        )
        assert abs(delta - 0.07) < 1e-9

    def test_negative_when_carnot_degrades_more(self):
        """Baseline drops 0.10, pipeline drops 0.15 -> delta = -0.05 < 0.

        Spec: REQ-BENCH-052
        """
        delta = compute_robustness_delta(
            baseline_std=0.80,
            baseline_adv=0.70,
            pipeline_std=0.80,
            pipeline_adv=0.65,
        )
        assert abs(delta - (-0.05)) < 1e-9

    def test_zero_when_equal_degradation(self):
        """Both baseline and pipeline drop by 0.10 -> delta = 0.

        Spec: REQ-BENCH-052
        """
        delta = compute_robustness_delta(
            baseline_std=0.80,
            baseline_adv=0.70,
            pipeline_std=0.80,
            pipeline_adv=0.70,
        )
        assert abs(delta) < 1e-9

    def test_exact_formula(self):
        """Verify the formula: (baseline_std - baseline_adv) - (pipeline_std - pipeline_adv).

        Spec: REQ-BENCH-052
        """
        # With known values: (0.90 - 0.75) - (0.85 - 0.82) = 0.15 - 0.03 = 0.12
        delta = compute_robustness_delta(0.90, 0.75, 0.85, 0.82)
        assert abs(delta - 0.12) < 1e-9

    def test_no_degradation_both_sides(self):
        """Both sides unchanged under adversarial conditions -> delta = 0.

        Spec: REQ-BENCH-052
        """
        delta = compute_robustness_delta(0.80, 0.80, 0.78, 0.78)
        assert abs(delta) < 1e-9

    def test_all_zeros(self):
        """All-zero inputs -> delta = 0.

        Spec: REQ-BENCH-052
        """
        delta = compute_robustness_delta(0.0, 0.0, 0.0, 0.0)
        assert delta == 0.0

    def test_perfect_baseline_robustness(self):
        """Baseline has zero drop, pipeline also zero -> delta = 0.

        Spec: REQ-BENCH-052
        """
        delta = compute_robustness_delta(0.80, 0.80, 0.75, 0.75)
        assert abs(delta) < 1e-9


# ---------------------------------------------------------------------------
# Tests for build_adversarial_v5_artifact
# ---------------------------------------------------------------------------


class TestBuildAdversarialV5Artifact:
    """Spec: REQ-BENCH-053, SCENARIO-BENCH-038"""

    def _base_results(self) -> dict:
        return {
            "baseline_standard_accuracy": 0.80,
            "baseline_adversarial_accuracy": 0.65,
            "pipeline_standard_accuracy": 0.78,
            "pipeline_adversarial_accuracy": 0.70,
        }

    def test_schema_field_present(self):
        """Artifact must carry schema='carnot.adversarial_v5.v1'.

        Spec: SCENARIO-BENCH-038
        """
        artifact = build_adversarial_v5_artifact(self._base_results(), "live_gpu")
        assert artifact["schema"] == "carnot.adversarial_v5.v1"

    def test_thesis_confirmed_when_positive_delta_live_gpu(self):
        """Positive delta + live_gpu -> thesis_confirmed and retro_039_confirmed=True.

        Spec: SCENARIO-BENCH-038
        """
        artifact = build_adversarial_v5_artifact(self._base_results(), "live_gpu")
        assert artifact["honest_verdict"] == "thesis_confirmed"
        assert artifact["retro_039_confirmed"] is True

    def test_thesis_rejected_when_negative_delta_live_gpu(self):
        """Negative delta + live_gpu -> thesis_rejected and retro_039_confirmed=False.

        Spec: SCENARIO-BENCH-038
        """
        results = {
            "baseline_standard_accuracy": 0.80,
            "baseline_adversarial_accuracy": 0.75,   # baseline barely drops
            "pipeline_standard_accuracy": 0.80,
            "pipeline_adversarial_accuracy": 0.60,   # pipeline drops more
        }
        artifact = build_adversarial_v5_artifact(results, "live_gpu")
        assert artifact["honest_verdict"] == "thesis_rejected"
        assert artifact["retro_039_confirmed"] is False

    def test_gpu_required_when_not_live(self):
        """Non-live inference_mode -> honest_verdict='gpu_required', retro_039_confirmed=False.

        Spec: SCENARIO-BENCH-038
        """
        artifact = build_adversarial_v5_artifact(self._base_results(), "simulated")
        assert artifact["honest_verdict"] == "gpu_required"
        assert artifact["retro_039_confirmed"] is False

    def test_gpu_required_for_any_non_live_mode(self):
        """Any inference_mode other than 'live_gpu' produces gpu_required verdict.

        Spec: SCENARIO-BENCH-038
        """
        for mode in ("gpu_required", "blocked", "simulated", ""):
            artifact = build_adversarial_v5_artifact(self._base_results(), mode)
            assert artifact["honest_verdict"] == "gpu_required"
            assert artifact["retro_039_confirmed"] is False

    def test_robustness_delta_is_computed_correctly(self):
        """Artifact's robustness_delta equals compute_robustness_delta output.

        Spec: REQ-BENCH-053
        """
        r = self._base_results()
        expected = compute_robustness_delta(
            r["baseline_standard_accuracy"],
            r["baseline_adversarial_accuracy"],
            r["pipeline_standard_accuracy"],
            r["pipeline_adversarial_accuracy"],
        )
        artifact = build_adversarial_v5_artifact(r, "live_gpu")
        assert abs(artifact["robustness_delta"] - expected) < 1e-9

    def test_all_four_accuracy_fields_present(self):
        """All four accuracy values must appear in the artifact.

        Spec: REQ-BENCH-053
        """
        artifact = build_adversarial_v5_artifact(self._base_results(), "live_gpu")
        for key in (
            "baseline_standard_accuracy",
            "baseline_adversarial_accuracy",
            "pipeline_standard_accuracy",
            "pipeline_adversarial_accuracy",
        ):
            assert key in artifact, f"Missing key: {key}"

    def test_extra_results_keys_passed_through(self):
        """Extra keys in results (batch_log, n_questions) appear in artifact.

        Spec: REQ-BENCH-053
        """
        results = dict(self._base_results())
        results["n_questions"] = 50
        results["batch_log"] = [{"batch_id": 0, "batch_size": 8}]
        artifact = build_adversarial_v5_artifact(results, "live_gpu")
        assert artifact["n_questions"] == 50
        assert artifact["batch_log"] == [{"batch_id": 0, "batch_size": 8}]

    def test_inference_mode_preserved_in_artifact(self):
        """inference_mode from argument appears in artifact verbatim.

        Spec: REQ-BENCH-053
        """
        artifact = build_adversarial_v5_artifact(self._base_results(), "live_gpu")
        assert artifact["inference_mode"] == "live_gpu"

    def test_zero_delta_is_thesis_rejected_live_gpu(self):
        """Zero delta (equal degradation) is not thesis_confirmed.

        Spec: SCENARIO-BENCH-038
        """
        results = {
            "baseline_standard_accuracy": 0.80,
            "baseline_adversarial_accuracy": 0.70,
            "pipeline_standard_accuracy": 0.80,
            "pipeline_adversarial_accuracy": 0.70,
        }
        artifact = build_adversarial_v5_artifact(results, "live_gpu")
        assert artifact["honest_verdict"] == "thesis_rejected"
        assert artifact["retro_039_confirmed"] is False

    def test_missing_accuracy_keys_default_to_zero(self):
        """Empty results dict uses 0.0 for missing accuracy keys.

        Spec: REQ-BENCH-053 — resilience to partial result dicts.
        """
        artifact = build_adversarial_v5_artifact({}, "live_gpu")
        assert artifact["robustness_delta"] == 0.0
        assert artifact["retro_039_confirmed"] is False
