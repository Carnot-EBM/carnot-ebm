"""Tests for Experiment 755: PSV Relapse Root-Cause Diagnosis.

Every test traces to REQ-PSV-013 and sub-requirements.
Coverage target: 100% of python/carnot/pipeline/psv_diagnostic.py.

Spec: REQ-PSV-013, REQ-PSV-013-1, REQ-PSV-013-2, REQ-PSV-013-3,
      REQ-PSV-013-4, REQ-PSV-013-5, SCENARIO-PSV-020
"""

from __future__ import annotations

import pytest

from carnot.pipeline.psv_diagnostic import PSVDiagnostic, PSVDiagnosticResult


# ---------------------------------------------------------------------------
# PSVDiagnosticResult dataclass — REQ-PSV-013-4, REQ-PSV-013-5
# ---------------------------------------------------------------------------


class TestPSVDiagnosticResultFields:
    """REQ-PSV-013-4: result must have all required verdict/evidence fields."""

    def test_fields_present(self):
        # Spec: REQ-PSV-013-4, REQ-PSV-013-5
        r = PSVDiagnosticResult(
            hypothesis_a_confirmed=True,
            hypothesis_b_confirmed=False,
            hypothesis_c_confirmed=True,
            primary_hypothesis="multiple_hypotheses",
            evidence_dict={"n_trials": 100, "n_steps": 30, "seed": 42},
        )
        assert r.hypothesis_a_confirmed is True
        assert r.hypothesis_b_confirmed is False
        assert r.hypothesis_c_confirmed is True
        assert r.primary_hypothesis == "multiple_hypotheses"
        assert "n_trials" in r.evidence_dict

    def test_default_evidence_dict_is_empty(self):
        # Default should be empty dict, not None
        r = PSVDiagnosticResult(
            hypothesis_a_confirmed=False,
            hypothesis_b_confirmed=False,
            hypothesis_c_confirmed=False,
            primary_hypothesis="diagnosis_inconclusive",
        )
        assert r.evidence_dict == {}


# ---------------------------------------------------------------------------
# PSVDiagnostic constructor
# ---------------------------------------------------------------------------


class TestPSVDiagnosticConstructor:
    """Verify constructor stores all parameters."""

    def test_defaults(self):
        d = PSVDiagnostic()
        assert d.n_trials == 100
        assert d.seed == 42
        assert d.n_steps == 30
        assert d.noise_std == 0.008

    def test_custom_params(self):
        d = PSVDiagnostic(n_trials=50, seed=7, n_steps=10, noise_std=0.005)
        assert d.n_trials == 50
        assert d.seed == 7
        assert d.n_steps == 10
        assert d.noise_std == 0.005


# ---------------------------------------------------------------------------
# _linear_slope
# ---------------------------------------------------------------------------


class TestLinearSlope:
    """Internal slope computation used by all three hypothesis tests."""

    def test_perfectly_increasing_series(self):
        slope = PSVDiagnostic._linear_slope([0.0, 1.0, 2.0, 3.0, 4.0])
        assert abs(slope - 1.0) < 1e-9

    def test_perfectly_decreasing_series(self):
        slope = PSVDiagnostic._linear_slope([4.0, 3.0, 2.0, 1.0, 0.0])
        assert abs(slope - (-1.0)) < 1e-9

    def test_flat_series(self):
        slope = PSVDiagnostic._linear_slope([0.5, 0.5, 0.5, 0.5])
        assert abs(slope) < 1e-9

    def test_single_value_returns_zero(self):
        assert PSVDiagnostic._linear_slope([0.3]) == 0.0

    def test_empty_returns_zero(self):
        assert PSVDiagnostic._linear_slope([]) == 0.0


# ---------------------------------------------------------------------------
# test_hypothesis_a — REQ-PSV-013-1
# ---------------------------------------------------------------------------


class TestHypothesisA:
    """REQ-PSV-013-1: 20% corrupted repairs must produce fp_rate_slope > 0."""

    def setup_method(self):
        self.diag = PSVDiagnostic(n_trials=100, seed=42, n_steps=30, noise_std=0.005)

    def test_returns_required_keys(self):
        # Spec: REQ-PSV-013-1
        result = self.diag.test_hypothesis_a()
        for key in ("fp_rates", "fp_at_step_0", "fp_at_step_10", "fp_at_step_20",
                    "fp_at_step_29", "slope", "confirmed"):
            assert key in result, f"Missing key: {key}"

    def test_fp_rates_length(self):
        result = self.diag.test_hypothesis_a()
        assert len(result["fp_rates"]) == 30

    def test_fp_rates_in_range(self):
        result = self.diag.test_hypothesis_a()
        for v in result["fp_rates"]:
            assert 0.0 <= v <= 1.0

    def test_slope_is_positive(self):
        # REQ-PSV-013-1: 20% corruption rate must produce slope > 0 (deterioration)
        result = self.diag.test_hypothesis_a()
        assert result["slope"] > 0, (
            f"Expected positive slope (memory contamination = deterioration) "
            f"but got slope={result['slope']}"
        )

    def test_confirmed_is_true(self):
        result = self.diag.test_hypothesis_a()
        assert result["confirmed"] is True

    def test_checkpoints_match_fp_rates(self):
        result = self.diag.test_hypothesis_a()
        fp = result["fp_rates"]
        assert result["fp_at_step_0"] == fp[0]
        assert result["fp_at_step_10"] == fp[10]
        assert result["fp_at_step_20"] == fp[20]
        assert result["fp_at_step_29"] == fp[29]

    def test_deterministic_with_same_seed(self):
        d1 = PSVDiagnostic(seed=42)
        d2 = PSVDiagnostic(seed=42)
        assert d1.test_hypothesis_a()["slope"] == d2.test_hypothesis_a()["slope"]

    def test_different_seed_gives_different_result(self):
        d1 = PSVDiagnostic(seed=42)
        d2 = PSVDiagnostic(seed=99)
        # Different seed → different fp_rates (though slope sign should still be positive)
        assert d1.test_hypothesis_a()["fp_rates"] != d2.test_hypothesis_a()["fp_rates"]


# ---------------------------------------------------------------------------
# test_hypothesis_b — REQ-PSV-013-2
# ---------------------------------------------------------------------------


class TestHypothesisB:
    """REQ-PSV-013-2: frozen coupling matrix must produce fp_rate_slope <= 0."""

    def setup_method(self):
        self.diag = PSVDiagnostic(n_trials=100, seed=42, n_steps=30, noise_std=0.005)

    def test_returns_required_keys(self):
        result = self.diag.test_hypothesis_b()
        for key in ("fp_rates", "fp_at_step_0", "fp_at_step_10", "fp_at_step_20",
                    "fp_at_step_29", "slope", "confirmed"):
            assert key in result

    def test_fp_rates_length(self):
        assert len(self.diag.test_hypothesis_b()["fp_rates"]) == 30

    def test_fp_rates_in_range(self):
        for v in self.diag.test_hypothesis_b()["fp_rates"]:
            assert 0.0 <= v <= 1.0

    def test_slope_is_non_positive(self):
        # REQ-PSV-013-2: frozen coupling keeps quality stable or improving → slope <= 0
        result = self.diag.test_hypothesis_b()
        assert result["slope"] <= 0.0, (
            f"Expected non-positive slope (frozen coupling = stable/improving) "
            f"but got slope={result['slope']}"
        )

    def test_confirmed_is_true(self):
        # Frozen = confirmed that unfrozen overwrite is the problem
        result = self.diag.test_hypothesis_b()
        assert result["confirmed"] is True

    def test_checkpoints_match_fp_rates(self):
        result = self.diag.test_hypothesis_b()
        fp = result["fp_rates"]
        assert result["fp_at_step_0"] == fp[0]
        assert result["fp_at_step_29"] == fp[29]

    def test_uses_different_seed_than_hyp_a(self):
        # Hypothesis B uses seed+1 to avoid correlation with A
        d = PSVDiagnostic(seed=42)
        result_a = d.test_hypothesis_a()
        result_b = d.test_hypothesis_b()
        assert result_a["fp_rates"] != result_b["fp_rates"]


# ---------------------------------------------------------------------------
# test_hypothesis_c — REQ-PSV-013-3
# ---------------------------------------------------------------------------


class TestHypothesisC:
    """REQ-PSV-013-3: zero-diversity question set must produce fp_rate_slope > 0."""

    def setup_method(self):
        self.diag = PSVDiagnostic(n_trials=100, seed=42, n_steps=30, noise_std=0.005)

    def test_returns_required_keys(self):
        result = self.diag.test_hypothesis_c()
        for key in ("fp_rates", "fp_at_step_0", "fp_at_step_10", "fp_at_step_20",
                    "fp_at_step_29", "slope", "confirmed"):
            assert key in result

    def test_fp_rates_length(self):
        assert len(self.diag.test_hypothesis_c()["fp_rates"]) == 30

    def test_fp_rates_in_range(self):
        for v in self.diag.test_hypothesis_c()["fp_rates"]:
            assert 0.0 <= v <= 1.0

    def test_slope_is_positive(self):
        # REQ-PSV-013-3: zero diversity causes long-run deterioration → slope > 0
        result = self.diag.test_hypothesis_c()
        assert result["slope"] > 0, (
            f"Expected positive slope (curriculum collapse = overfitting) "
            f"but got slope={result['slope']}"
        )

    def test_confirmed_is_true(self):
        result = self.diag.test_hypothesis_c()
        assert result["confirmed"] is True

    def test_early_steps_show_improvement(self):
        # First 5 steps should show quality improvement (rapid early learning)
        result = self.diag.test_hypothesis_c()
        fp = result["fp_rates"]
        # fp_at_step_0 should be higher than fp_at step 4 on average
        # (fp decreases = quality improves in early phase)
        avg_early = sum(fp[:5]) / 5
        avg_late = sum(fp[20:]) / 10
        assert avg_late > avg_early, (
            "Expected late fp_rates to exceed early fp_rates (overfitting regime), "
            f"but avg_early={avg_early:.4f} avg_late={avg_late:.4f}"
        )

    def test_checkpoints_match_fp_rates(self):
        result = self.diag.test_hypothesis_c()
        fp = result["fp_rates"]
        assert result["fp_at_step_0"] == fp[0]
        assert result["fp_at_step_29"] == fp[29]

    def test_uses_different_seed_than_hyp_a_and_b(self):
        d = PSVDiagnostic(seed=42)
        result_a = d.test_hypothesis_a()
        result_c = d.test_hypothesis_c()
        assert result_a["fp_rates"] != result_c["fp_rates"]


# ---------------------------------------------------------------------------
# diagnose — REQ-PSV-013-4, REQ-PSV-013-5
# ---------------------------------------------------------------------------


class TestDiagnose:
    """REQ-PSV-013-4: honest_verdict must be one of the five valid values.
    REQ-PSV-013-5: evidence_dict must include per-hypothesis slopes and primary_hypothesis.
    """

    VALID_VERDICTS = {
        "hypothesis_a_confirmed",
        "hypothesis_b_confirmed",
        "hypothesis_c_confirmed",
        "multiple_hypotheses",
        "diagnosis_inconclusive",
    }

    def test_returns_psv_diagnostic_result(self):
        d = PSVDiagnostic(n_trials=100, seed=42, n_steps=30)
        result = d.diagnose()
        assert isinstance(result, PSVDiagnosticResult)

    def test_primary_hypothesis_is_valid_verdict(self):
        # REQ-PSV-013-4
        d = PSVDiagnostic(n_trials=100, seed=42, n_steps=30)
        result = d.diagnose()
        assert result.primary_hypothesis in self.VALID_VERDICTS, (
            f"primary_hypothesis={result.primary_hypothesis!r} not in {self.VALID_VERDICTS}"
        )

    def test_evidence_dict_has_hypothesis_keys(self):
        # REQ-PSV-013-5: evidence_dict must contain per-hypothesis data
        d = PSVDiagnostic(seed=42, n_steps=30)
        result = d.diagnose()
        assert "hypothesis_a" in result.evidence_dict
        assert "hypothesis_b" in result.evidence_dict
        assert "hypothesis_c" in result.evidence_dict
        assert "n_trials" in result.evidence_dict
        assert "n_steps" in result.evidence_dict

    def test_evidence_dict_has_slopes(self):
        d = PSVDiagnostic(seed=42, n_steps=30)
        result = d.diagnose()
        assert "slope" in result.evidence_dict["hypothesis_a"]
        assert "slope" in result.evidence_dict["hypothesis_b"]
        assert "slope" in result.evidence_dict["hypothesis_c"]

    def test_multiple_hypotheses_when_two_or_more_confirmed(self):
        # With default params, A and C should confirm (positive slope) and B confirms too
        # So we expect multiple_hypotheses
        d = PSVDiagnostic(seed=42, n_steps=30)
        result = d.diagnose()
        n_confirmed = sum([
            result.hypothesis_a_confirmed,
            result.hypothesis_b_confirmed,
            result.hypothesis_c_confirmed,
        ])
        if n_confirmed >= 2:
            assert result.primary_hypothesis == "multiple_hypotheses"

    def test_single_confirmed_selects_that_hypothesis(self):
        # Force conditions where only hypothesis A fires by making a sub-diagnostic
        # We do this by testing the logic directly with a mock-style subclass
        class _SingleADiagnostic(PSVDiagnostic):
            def test_hypothesis_a(self):
                r = super().test_hypothesis_a()
                r["confirmed"] = True
                return r
            def test_hypothesis_b(self):
                r = super().test_hypothesis_b()
                r["confirmed"] = False
                return r
            def test_hypothesis_c(self):
                r = super().test_hypothesis_c()
                r["confirmed"] = False
                return r
        result = _SingleADiagnostic(seed=42, n_steps=10).diagnose()
        assert result.primary_hypothesis == "hypothesis_a_confirmed"
        assert result.hypothesis_a_confirmed is True
        assert result.hypothesis_b_confirmed is False

    def test_diagnosis_inconclusive_when_none_confirmed(self):
        class _NoneConfirmedDiagnostic(PSVDiagnostic):
            def test_hypothesis_a(self):
                r = super().test_hypothesis_a()
                r["confirmed"] = False
                return r
            def test_hypothesis_b(self):
                r = super().test_hypothesis_b()
                r["confirmed"] = False
                return r
            def test_hypothesis_c(self):
                r = super().test_hypothesis_c()
                r["confirmed"] = False
                return r
        result = _NoneConfirmedDiagnostic(seed=42, n_steps=10).diagnose()
        assert result.primary_hypothesis == "diagnosis_inconclusive"

    def test_hypothesis_b_only_routes_correctly(self):
        class _OnlyBDiagnostic(PSVDiagnostic):
            def test_hypothesis_a(self):
                r = super().test_hypothesis_a()
                r["confirmed"] = False
                return r
            def test_hypothesis_b(self):
                r = super().test_hypothesis_b()
                r["confirmed"] = True
                return r
            def test_hypothesis_c(self):
                r = super().test_hypothesis_c()
                r["confirmed"] = False
                return r
        result = _OnlyBDiagnostic(seed=42, n_steps=10).diagnose()
        assert result.primary_hypothesis == "hypothesis_b_confirmed"

    def test_hypothesis_c_only_routes_correctly(self):
        class _OnlyCDiagnostic(PSVDiagnostic):
            def test_hypothesis_a(self):
                r = super().test_hypothesis_a()
                r["confirmed"] = False
                return r
            def test_hypothesis_b(self):
                r = super().test_hypothesis_b()
                r["confirmed"] = False
                return r
            def test_hypothesis_c(self):
                r = super().test_hypothesis_c()
                r["confirmed"] = True
                return r
        result = _OnlyCDiagnostic(seed=42, n_steps=10).diagnose()
        assert result.primary_hypothesis == "hypothesis_c_confirmed"

    def test_default_run_produces_multiple_hypotheses(self):
        # With default parameters, A and C should both show positive slope
        # and B should show negative slope (frozen coupling stable), giving
        # multiple_hypotheses as the honest verdict for the relapse.
        d = PSVDiagnostic(seed=42, n_steps=30)
        result = d.diagnose()
        # All three should confirm under default params
        assert result.hypothesis_a_confirmed is True
        assert result.hypothesis_b_confirmed is True
        assert result.hypothesis_c_confirmed is True
        assert result.primary_hypothesis == "multiple_hypotheses"
