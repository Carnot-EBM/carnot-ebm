"""Tests for Exp 1689: CASAL vs Langevin scaling comparison.

Spec coverage: REQ-SAMPLE-1689, REQ-SAMPLE-1689-1, REQ-SAMPLE-1689-2,
               REQ-SAMPLE-1689-3, REQ-SAMPLE-1689-4, SCENARIO-SAMPLE-1689
"""
from __future__ import annotations

import numpy as np
import pytest

from carnot.phase3.casal_scaling import (
    build_casal_scaling_artifact,
    count_violations,
    energy_value,
    make_random_ebm,
    run_casal_comparison,
)

# Use a short step count for fast unit tests; the full 1000-step run is done
# by the artifact-writing script, not here.
_FAST_STEPS = 50


class TestMakeRandomEBM:
    """REQ-SAMPLE-1689-1: Random EBM construction is reproducible and well-formed."""

    def test_n16_shapes(self) -> None:
        """SCENARIO-SAMPLE-1689: n=16 EBM has correct array shapes."""
        ebm = make_random_ebm(16, seed=0)
        assert ebm.variables == 16
        assert ebm.coupling.shape == (16, 16)
        assert ebm.bias.shape == (16,)

    def test_n32_shapes(self) -> None:
        """SCENARIO-SAMPLE-1689: n=32 EBM has correct array shapes."""
        ebm = make_random_ebm(32, seed=0)
        assert ebm.variables == 32
        assert ebm.coupling.shape == (32, 32)
        assert ebm.bias.shape == (32,)

    def test_coupling_is_symmetric(self) -> None:
        """REQ-SAMPLE-1689-1: Coupling matrix J must be symmetric for EBM validity."""
        ebm = make_random_ebm(16, seed=7)
        np.testing.assert_allclose(ebm.coupling, ebm.coupling.T, atol=1e-12)

    def test_different_seeds_give_different_ebms(self) -> None:
        """REQ-SAMPLE-1689-1: Distinct seeds produce distinct coupling matrices."""
        ebm_a = make_random_ebm(16, seed=0)
        ebm_b = make_random_ebm(16, seed=1)
        assert not np.allclose(ebm_a.coupling, ebm_b.coupling)

    def test_same_seed_is_reproducible(self) -> None:
        """REQ-SAMPLE-1689-1: Same seed always yields the same EBM."""
        ebm_a = make_random_ebm(16, seed=42)
        ebm_b = make_random_ebm(16, seed=42)
        np.testing.assert_array_equal(ebm_a.coupling, ebm_b.coupling)
        np.testing.assert_array_equal(ebm_a.bias, ebm_b.bias)


class TestEnergyValue:
    """REQ-SAMPLE-1689-1: Energy function is finite and consistent."""

    def test_energy_is_finite_at_zero(self) -> None:
        """SCENARIO-SAMPLE-1689: Energy at origin equals -h^T 0 = 0 (bias term is 0 at x=0)."""
        ebm = make_random_ebm(16, seed=42)
        e = energy_value(ebm, np.zeros(16))
        assert np.isfinite(e)
        # At x=0: E(0) = -0.5*0^T J 0 - h^T 0 = 0
        assert e == pytest.approx(0.0, abs=1e-12)

    def test_energy_is_finite_at_unit_vector(self) -> None:
        """SCENARIO-SAMPLE-1689: Energy is finite at tanh-bounded state."""
        ebm = make_random_ebm(16, seed=3)
        x = np.tanh(np.random.default_rng(0).standard_normal(16))
        assert np.isfinite(energy_value(ebm, x))


class TestCountViolations:
    """REQ-SAMPLE-1689-2: Constraint violation counting is correct."""

    def test_no_violations_inside_limit(self) -> None:
        """SCENARIO-SAMPLE-1689: All components below limit gives zero violations."""
        x = np.full(8, 0.5)
        assert count_violations(x, 0.9) == 0

    def test_violations_at_exact_boundary(self) -> None:
        """REQ-SAMPLE-1689-2: Component exactly at limit is not a violation (|x|=limit is OK)."""
        x = np.array([0.9, 0.5])
        assert count_violations(x, 0.9) == 0

    def test_violations_above_limit(self) -> None:
        """SCENARIO-SAMPLE-1689: Components exceeding limit are counted correctly."""
        x = np.array([0.95, 0.5, -0.95, 0.8, -0.91])
        assert count_violations(x, 0.9) == 3

    def test_all_violating(self) -> None:
        """REQ-SAMPLE-1689-2: All-violating array returns n."""
        x = np.full(5, 0.99)
        assert count_violations(x, 0.9) == 5


class TestRunCasalComparison:
    """REQ-SAMPLE-1689-1, REQ-SAMPLE-1689-3: Both samplers run and produce valid outputs."""

    def test_n16_both_finite(self) -> None:
        """SCENARIO-SAMPLE-1689: Both samplers produce finite results on n=16."""
        result = run_casal_comparison(n=16, n_steps=_FAST_STEPS, seed=42)
        assert result["casal_finite"], "CASAL result is not finite"
        assert result["langevin_finite"], "Langevin result is not finite"
        assert np.isfinite(result["casal_energy"])
        assert np.isfinite(result["langevin_energy"])

    def test_n32_both_finite(self) -> None:
        """SCENARIO-SAMPLE-1689: Both samplers produce finite results on n=32."""
        result = run_casal_comparison(n=32, n_steps=_FAST_STEPS, seed=42)
        assert result["casal_finite"], "CASAL result is not finite on n=32"
        assert result["langevin_finite"], "Langevin result is not finite on n=32"

    def test_required_fields_present(self) -> None:
        """REQ-SAMPLE-1689-2: Result dict contains all required comparison fields."""
        result = run_casal_comparison(n=16, n_steps=_FAST_STEPS, seed=0)
        required = [
            "n", "n_steps", "initial_energy",
            "casal_energy", "langevin_energy",
            "casal_violations", "langevin_violations",
            "casal_time_s", "langevin_time_s",
            "speedup_ratio", "casal_finite", "langevin_finite",
        ]
        for field in required:
            assert field in result, f"Missing field: {field}"

    def test_violation_counts_non_negative(self) -> None:
        """REQ-SAMPLE-1689-2: Violation counts are non-negative integers."""
        result = run_casal_comparison(n=16, n_steps=_FAST_STEPS, seed=0)
        assert isinstance(result["casal_violations"], int)
        assert isinstance(result["langevin_violations"], int)
        assert result["casal_violations"] >= 0
        assert result["langevin_violations"] >= 0

    def test_n_and_n_steps_recorded(self) -> None:
        """REQ-SAMPLE-1689-1: Result records the n and n_steps used."""
        result = run_casal_comparison(n=16, n_steps=_FAST_STEPS, seed=0)
        assert result["n"] == 16
        assert result["n_steps"] == _FAST_STEPS

    def test_speedup_ratio_is_finite(self) -> None:
        """REQ-SAMPLE-1689-3: speedup_ratio is finite (no division by zero)."""
        result = run_casal_comparison(n=16, n_steps=_FAST_STEPS, seed=42)
        assert np.isfinite(result["speedup_ratio"])


class TestBuildCasalScalingArtifact:
    """REQ-SAMPLE-1689-2, REQ-SAMPLE-1689-4: Artifact structure satisfies schema requirements."""

    def _make_results(self) -> tuple[dict, dict]:
        r16 = run_casal_comparison(n=16, n_steps=_FAST_STEPS, seed=42)
        r32 = run_casal_comparison(n=32, n_steps=_FAST_STEPS, seed=42)
        return r16, r32

    def test_required_schema_fields_present(self) -> None:
        """REQ-SAMPLE-1689-2: Artifact contains all required schema fields."""
        r16, r32 = self._make_results()
        artifact = build_casal_scaling_artifact(r16, r32)
        for field in [
            "schema", "casal_energy", "langevin_energy",
            "speedup_ratio", "acceptance_gate_passed",
        ]:
            assert field in artifact, f"Missing required artifact field: {field}"

    def test_schema_string(self) -> None:
        """REQ-SAMPLE-1689-2: Schema field identifies this experiment."""
        r16, r32 = self._make_results()
        artifact = build_casal_scaling_artifact(r16, r32)
        assert artifact["schema"] == "carnot.experiment_1689_casal_scaling.v1"

    def test_acceptance_gate_passes_when_finite(self) -> None:
        """REQ-SAMPLE-1689-4: Gate is True when both samplers produce finite results."""
        r16, r32 = self._make_results()
        artifact = build_casal_scaling_artifact(r16, r32)
        assert artifact["acceptance_gate_passed"] is True

    def test_verdict_has_terminal_prefix(self) -> None:
        """SCENARIO-SAMPLE-1689: honest_verdict starts with a conductor-recognised terminal prefix."""
        r16, r32 = self._make_results()
        artifact = build_casal_scaling_artifact(r16, r32)
        verdict = artifact["honest_verdict"]
        terminal_prefixes = (
            "complete:", "complete_",
            "success:", "success_",
            "passed:", "passed_",
            "shipped:", "shipped_",
        )
        assert verdict.startswith(terminal_prefixes), (
            f"honest_verdict must start with a terminal prefix; got: {verdict!r}"
        )

    def test_speedup_ratio_is_average_of_n16_n32(self) -> None:
        """REQ-SAMPLE-1689-3: Artifact speedup_ratio is the mean of n=16 and n=32 ratios."""
        r16, r32 = self._make_results()
        artifact = build_casal_scaling_artifact(r16, r32)
        expected = round((r16["speedup_ratio"] + r32["speedup_ratio"]) / 2.0, 4)
        assert artifact["speedup_ratio"] == pytest.approx(expected, abs=1e-6)

    def test_artifact_is_json_serialisable(self) -> None:
        """REQ-SAMPLE-1689-2: Artifact can be serialised to JSON without errors."""
        import json
        r16, r32 = self._make_results()
        artifact = build_casal_scaling_artifact(r16, r32)
        # Should not raise
        json.dumps(artifact)
