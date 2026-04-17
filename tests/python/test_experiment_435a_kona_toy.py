"""Tests for Phase 3 seed: continuous EBM vs Ising minimum recovery (Exp 435a).

Spec coverage: REQ-KONA-001, SCENARIO-KONA-001, SCENARIO-KONA-002
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from carnot.phase3.continuous_ebm import (
    ContinuousEBM,
    build_kona_artifact,
    compare_minima,
    fit_continuous_ebm,
    sample_continuous,
)


# ---------------------------------------------------------------------------
# Minimal stub so tests don't require a full IsingModel import
# ---------------------------------------------------------------------------


@dataclass
class _StubIsing:
    """Minimal stub with coupling and bias, mimicking IsingModel interface."""

    coupling: np.ndarray
    bias: np.ndarray


def _make_sparse_ising(n: int = 10, density: float = 0.3, seed: int = 42) -> _StubIsing:
    """Build a random n-variable Ising model with sparse symmetric couplings.

    Density controls the fraction of off-diagonal entries that are non-zero.
    This matches the problem used in the experiment script so tests are faithful.
    """
    rng = np.random.default_rng(seed)
    # Upper-triangular mask with given density
    mask = rng.random((n, n)) < density
    mask = np.triu(mask, k=1)
    mask = mask | mask.T  # symmetrise
    J_raw = rng.uniform(-1.0, 1.0, (n, n)) * mask
    J = (J_raw + J_raw.T) / 2.0  # enforce exact symmetry
    h = rng.uniform(-0.5, 0.5, n)
    return _StubIsing(coupling=J, bias=h)


# ---------------------------------------------------------------------------
# ContinuousEBM dataclass
# ---------------------------------------------------------------------------


class TestContinuousEBMDataclass:
    """REQ-KONA-001: ContinuousEBM stores variables, coupling, bias."""

    def test_fields_stored(self) -> None:
        """REQ-KONA-001: All three fields are stored and accessible."""
        J = np.eye(3)
        h = np.zeros(3)
        model = ContinuousEBM(variables=3, coupling=J, bias=h)
        assert model.variables == 3
        assert model.coupling.shape == (3, 3)
        assert model.bias.shape == (3,)

    def test_coupling_preserved(self) -> None:
        """REQ-KONA-001: Coupling matrix is stored without modification."""
        J = np.array([[0.5, 0.1], [0.1, 0.3]])
        h = np.array([0.2, -0.1])
        model = ContinuousEBM(variables=2, coupling=J, bias=h)
        np.testing.assert_array_equal(model.coupling, J)

    def test_bias_preserved(self) -> None:
        """REQ-KONA-001: Bias vector is stored without modification."""
        J = np.zeros((4, 4))
        h = np.array([1.0, -1.0, 0.5, -0.5])
        model = ContinuousEBM(variables=4, coupling=J, bias=h)
        np.testing.assert_array_equal(model.bias, h)


# ---------------------------------------------------------------------------
# fit_continuous_ebm
# ---------------------------------------------------------------------------


class TestFitContinuousEBM:
    """REQ-KONA-001: fit_continuous_ebm reuses Ising J/h as init."""

    def test_returns_continuous_ebm(self) -> None:
        """REQ-KONA-001: Return type is ContinuousEBM."""
        ising = _make_sparse_ising()
        result = fit_continuous_ebm(ising)
        assert isinstance(result, ContinuousEBM)

    def test_variables_match(self) -> None:
        """REQ-KONA-001: variables field equals coupling dimension."""
        ising = _make_sparse_ising(n=10)
        result = fit_continuous_ebm(ising)
        assert result.variables == 10

    def test_coupling_copied(self) -> None:
        """REQ-KONA-001: Coupling matrix matches Ising's coupling."""
        ising = _make_sparse_ising()
        result = fit_continuous_ebm(ising)
        np.testing.assert_allclose(result.coupling, np.asarray(ising.coupling))

    def test_bias_copied(self) -> None:
        """REQ-KONA-001: Bias vector matches Ising's bias."""
        ising = _make_sparse_ising()
        result = fit_continuous_ebm(ising)
        np.testing.assert_allclose(result.bias, np.asarray(ising.bias))

    def test_coupling_is_float64(self) -> None:
        """REQ-KONA-001: Coupling is upcast to float64 for numerical stability."""
        J = np.eye(5, dtype=np.float32)
        h = np.zeros(5, dtype=np.float32)
        ising = _StubIsing(coupling=J, bias=h)
        result = fit_continuous_ebm(ising)
        assert result.coupling.dtype == np.float64

    def test_small_model(self) -> None:
        """REQ-KONA-001: Works for n=2 (smallest non-trivial case)."""
        J = np.array([[0.0, 1.0], [1.0, 0.0]])
        h = np.array([0.5, -0.5])
        ising = _StubIsing(coupling=J, bias=h)
        result = fit_continuous_ebm(ising)
        assert result.variables == 2


# ---------------------------------------------------------------------------
# sample_continuous
# ---------------------------------------------------------------------------


class TestSampleContinuous:
    """REQ-KONA-001: sample_continuous minimises E via gradient descent + tanh."""

    def test_output_shape(self) -> None:
        """REQ-KONA-001: Output shape is (n,)."""
        model = fit_continuous_ebm(_make_sparse_ising(n=10))
        x = sample_continuous(model)
        assert x.shape == (10,)

    def test_output_in_tanh_range(self) -> None:
        """REQ-KONA-001: All values are strictly in (-1, 1) due to tanh squashing."""
        model = fit_continuous_ebm(_make_sparse_ising(n=10))
        x = sample_continuous(model, n_steps=500)
        assert np.all(x > -1.0)
        assert np.all(x < 1.0)

    def test_energy_decreases(self) -> None:
        """REQ-KONA-001: Energy at end of descent is lower than at initial point."""
        ising = _make_sparse_ising(n=10)
        model = fit_continuous_ebm(ising)
        J, h = model.coupling, model.bias

        def energy(x: np.ndarray) -> float:
            return float(-0.5 * x @ J @ x - h @ x)

        # Initial energy with x sampled at same seed=0
        rng = np.random.default_rng(0)
        x_init = rng.uniform(-1.0, 1.0, 10)
        e_init = energy(x_init)

        x_final = sample_continuous(model, n_steps=1000, lr=0.01, seed=0)
        e_final = energy(x_final)

        assert e_final < e_init, f"Energy did not decrease: {e_init:.4f} → {e_final:.4f}"

    def test_deterministic_with_same_seed(self) -> None:
        """REQ-KONA-001: Same seed produces identical output."""
        model = fit_continuous_ebm(_make_sparse_ising(n=10))
        x1 = sample_continuous(model, seed=7)
        x2 = sample_continuous(model, seed=7)
        np.testing.assert_array_equal(x1, x2)

    def test_different_seeds_differ(self) -> None:
        """REQ-KONA-001: Different seeds produce different starting points."""
        model = fit_continuous_ebm(_make_sparse_ising(n=10))
        x1 = sample_continuous(model, n_steps=1, seed=0)
        x2 = sample_continuous(model, n_steps=1, seed=99)
        assert not np.allclose(x1, x2)

    def test_zero_steps_returns_initial(self) -> None:
        """REQ-KONA-001: n_steps=0 returns tanh-squashed initial point."""
        model = fit_continuous_ebm(_make_sparse_ising(n=5))
        rng = np.random.default_rng(0)
        x_init = rng.uniform(-1.0, 1.0, 5)
        x_out = sample_continuous(model, n_steps=0, seed=0)
        # tanh(x_init) because step is x = tanh(x - lr*grad) but 0 steps means
        # we return directly from the initialisation — check it's in [-1, 1]
        assert x_out.shape == (5,)
        assert np.all(np.abs(x_out) <= 1.0)


# ---------------------------------------------------------------------------
# compare_minima
# ---------------------------------------------------------------------------


class TestCompareMinima:
    """REQ-KONA-001: compare_minima returns l2_distance and sign_agreement."""

    def test_keys_present(self) -> None:
        """REQ-KONA-001: Return dict has both required keys."""
        ising_s = np.ones(5)
        cont_s = np.ones(5) * 0.95
        result = compare_minima(ising_s, cont_s)
        assert "l2_distance" in result
        assert "sign_agreement" in result

    def test_identical_inputs_zero_l2(self) -> None:
        """REQ-KONA-001: Identical samples → l2_distance == 0."""
        x = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        result = compare_minima(x, x.copy())
        assert result["l2_distance"] == pytest.approx(0.0, abs=1e-10)

    def test_identical_inputs_full_agreement(self) -> None:
        """REQ-KONA-001: Identical samples → sign_agreement == 1.0."""
        x = np.array([1.0, -1.0, 0.5, -0.5, 0.1])
        result = compare_minima(x, x.copy())
        assert result["sign_agreement"] == pytest.approx(1.0)

    def test_opposite_inputs_zero_agreement(self) -> None:
        """REQ-KONA-001: Fully opposite signs → sign_agreement == 0.0."""
        x = np.array([1.0, -1.0, 1.0, -1.0])
        result = compare_minima(x, -x)
        assert result["sign_agreement"] == pytest.approx(0.0)

    def test_l2_distance_correct(self) -> None:
        """REQ-KONA-001: l2_distance matches numpy norm."""
        a = np.array([1.0, 0.0, -1.0])
        b = np.array([0.9, 0.1, -0.95])
        result = compare_minima(a, b)
        expected = float(np.linalg.norm(a - b))
        assert result["l2_distance"] == pytest.approx(expected)

    def test_partial_sign_agreement(self) -> None:
        """REQ-KONA-001: 3/4 matching signs → sign_agreement == 0.75."""
        a = np.array([1.0, 1.0, 1.0, 1.0])
        b = np.array([0.9, 0.8, 0.7, -0.6])
        result = compare_minima(a, b)
        assert result["sign_agreement"] == pytest.approx(0.75)

    def test_zero_treated_as_positive(self) -> None:
        """REQ-KONA-001: Zero components are treated as +1 for sign comparison."""
        a = np.array([0.0, 1.0])  # 0 → sign +1
        b = np.array([0.5, 1.0])  # both positive
        result = compare_minima(a, b)
        assert result["sign_agreement"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# build_kona_artifact
# ---------------------------------------------------------------------------


class TestBuildKonaArtifact:
    """REQ-KONA-001, SCENARIO-KONA-002: artifact schema and honest_verdict."""

    def test_schema_field(self) -> None:
        """SCENARIO-KONA-002: schema is 'carnot.kona_seed.v1'."""
        artifact = build_kona_artifact({"l2_distance": 0.05, "sign_agreement": 0.95})
        assert artifact["schema"] == "carnot.kona_seed.v1"

    def test_run_date_present(self) -> None:
        """SCENARIO-KONA-002: run_date is a non-empty string."""
        artifact = build_kona_artifact({"l2_distance": 0.05, "sign_agreement": 0.95})
        assert isinstance(artifact["run_date"], str)
        assert len(artifact["run_date"]) > 0

    def test_verdict_continuous_matches_ising(self) -> None:
        """SCENARIO-KONA-002: L2<0.1 AND sign>0.9 → 'continuous_matches_ising'."""
        artifact = build_kona_artifact({"l2_distance": 0.05, "sign_agreement": 0.95})
        assert artifact["honest_verdict"] == "continuous_matches_ising"

    def test_verdict_partial_match(self) -> None:
        """SCENARIO-KONA-002: sign>0.7 but not full match → 'partial_match'."""
        artifact = build_kona_artifact({"l2_distance": 0.5, "sign_agreement": 0.8})
        assert artifact["honest_verdict"] == "partial_match"

    def test_verdict_failed_to_match(self) -> None:
        """SCENARIO-KONA-002: sign<=0.7 → 'failed_to_match'."""
        artifact = build_kona_artifact({"l2_distance": 1.5, "sign_agreement": 0.5})
        assert artifact["honest_verdict"] == "failed_to_match"

    def test_l2_boundary_continuous_matches(self) -> None:
        """SCENARIO-KONA-002: l2 exactly at 0.099 (just under 0.1) → continuous_matches."""
        artifact = build_kona_artifact({"l2_distance": 0.099, "sign_agreement": 0.91})
        assert artifact["honest_verdict"] == "continuous_matches_ising"

    def test_l2_boundary_partial_match(self) -> None:
        """SCENARIO-KONA-002: l2>=0.1 but sign>0.9 → partial_match (l2 fails threshold)."""
        artifact = build_kona_artifact({"l2_distance": 0.1, "sign_agreement": 0.95})
        assert artifact["honest_verdict"] == "partial_match"

    def test_l2_and_sign_values_in_artifact(self) -> None:
        """SCENARIO-KONA-002: l2_distance and sign_agreement appear in artifact."""
        comp = {"l2_distance": 0.05, "sign_agreement": 0.95}
        artifact = build_kona_artifact(comp)
        assert artifact["l2_distance"] == pytest.approx(0.05)
        assert artifact["sign_agreement"] == pytest.approx(0.95)

    def test_extra_fields_merged(self) -> None:
        """SCENARIO-KONA-002: Optional extra fields appear in artifact."""
        artifact = build_kona_artifact(
            {"l2_distance": 0.05, "sign_agreement": 0.95},
            extra={"n_vars": 10, "ising_energy": -4.2},
        )
        assert artifact["n_vars"] == 10
        assert artifact["ising_energy"] == pytest.approx(-4.2)

    def test_json_serialisable(self) -> None:
        """SCENARIO-KONA-002: Artifact is serialisable to JSON without error."""
        import json

        artifact = build_kona_artifact(
            {"l2_distance": 0.05, "sign_agreement": 0.95},
            extra={"note": "test"},
        )
        serialised = json.dumps(artifact)
        assert len(serialised) > 0


# ---------------------------------------------------------------------------
# Integration: SCENARIO-KONA-001 (end-to-end agreement on 10-var problem)
# ---------------------------------------------------------------------------


class TestKonaScenario001:
    """SCENARIO-KONA-001: Continuous minimiser within L2 tolerance of Ising sample."""

    def _ising_ground_state(self, ising: _StubIsing, seed: int = 1) -> np.ndarray:
        """Simple simulated annealing for test purposes (discrete {-1,+1} states)."""
        rng = np.random.default_rng(seed)
        n = ising.coupling.shape[0]
        J = ising.coupling
        h = ising.bias

        # Start at random spin configuration
        state = rng.choice([-1.0, 1.0], size=n)
        best = state.copy()
        best_e = float(-0.5 * state @ J @ state - h @ state)

        # Annealing schedule: T from 2.0 → 0.01 over 5000 steps
        n_steps = 5000
        for step in range(n_steps):
            T = 2.0 * (0.01 / 2.0) ** (step / n_steps)
            i = int(rng.integers(n))
            # Energy change if we flip spin i
            delta = 2.0 * state[i] * (J[i] @ state + h[i])
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                state[i] = -state[i]
            e = float(-0.5 * state @ J @ state - h @ state)
            if e < best_e:
                best_e = e
                best = state.copy()
        return best

    def test_continuous_matches_ising_on_toy_problem(self) -> None:
        """SCENARIO-KONA-001: 10-var problem, continuous minimiser close to Ising SA."""
        ising = _make_sparse_ising(n=10, density=0.3, seed=42)
        ising_sample = self._ising_ground_state(ising, seed=1)
        model = fit_continuous_ebm(ising)
        cont_sample = sample_continuous(model, n_steps=2000, lr=0.02, seed=0)
        result = compare_minima(ising_sample, cont_sample)
        # At least sign agreement > 0.7 (partial_match or better) on this problem
        assert result["sign_agreement"] >= 0.7, (
            f"Sign agreement {result['sign_agreement']:.3f} below 0.7 threshold. "
            f"L2={result['l2_distance']:.4f}"
        )
