"""Tests for Exp 812: Ising Constraint Injection.

Spec: REQ-VERIFY-095, REQ-VERIFY-096, SCENARIO-VERIFY-129
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.pipeline.ising_constraint_injector import (
    ConstraintInjectionResult,
    IsingConstraintInjector,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def injector() -> IsingConstraintInjector:
    """Default injector: 384-dim embeddings, 64 spins."""
    return IsingConstraintInjector(embedding_dim=384, n_spins=64)


@pytest.fixture()
def fake_embeddings() -> list[list[float]]:
    """Two deterministic 384-dim embeddings for testing.

    We use a fixed RNG so every test run is reproducible without touching
    the sentence_transformers library (which may not be installed in CI).
    """
    rng = np.random.default_rng(0)
    return [list(rng.standard_normal(384)), list(rng.standard_normal(384))]


@pytest.fixture()
def fake_J(injector: IsingConstraintInjector) -> np.ndarray:
    """Random symmetric coupling matrix of shape (n_spins, n_spins)."""
    rng = np.random.default_rng(1)
    J_raw = rng.standard_normal((injector.n_spins, injector.n_spins)) * 0.1
    return (J_raw + J_raw.T) / 2.0


@pytest.fixture()
def fake_ising(injector: IsingConstraintInjector, fake_J: np.ndarray) -> object:
    """Minimal IsingModel stub exposing only the .coupling attribute."""

    class _FakeIsing:
        def __init__(self, J: np.ndarray) -> None:
            self.coupling = J

    return _FakeIsing(fake_J)


@pytest.fixture()
def spins(injector: IsingConstraintInjector) -> np.ndarray:
    """Fixed ±1 spin configuration for reproducible energy tests."""
    rng = np.random.default_rng(7)
    return rng.choice([-1.0, 1.0], size=injector.n_spins)


# ---------------------------------------------------------------------------
# REQ-VERIFY-095: project_to_spin_bias shape contract
# ---------------------------------------------------------------------------


class TestProjectToSpinBias:
    """REQ-VERIFY-095: bias vector must have shape (n_spins,)."""

    def test_output_shape_two_embeddings(
        self, injector: IsingConstraintInjector, fake_embeddings: list[list[float]]
    ) -> None:
        """project_to_spin_bias returns (n_spins,) for two embeddings.

        Spec: REQ-VERIFY-095
        """
        bias = injector.project_to_spin_bias(fake_embeddings)
        assert bias.shape == (injector.n_spins,)

    def test_output_shape_single_embedding(
        self, injector: IsingConstraintInjector, fake_embeddings: list[list[float]]
    ) -> None:
        """project_to_spin_bias returns (n_spins,) for a single embedding.

        Spec: REQ-VERIFY-095
        """
        bias = injector.project_to_spin_bias([fake_embeddings[0]])
        assert bias.shape == (injector.n_spins,)

    def test_empty_embeddings_returns_zeros(self, injector: IsingConstraintInjector) -> None:
        """project_to_spin_bias returns all-zero bias when embedding list is empty.

        When no constraints are retrieved, the bias must be zero so that
        inject_into_coupling_matrix produces no change to J.

        Spec: REQ-VERIFY-095-3
        """
        bias = injector.project_to_spin_bias([])
        assert bias.shape == (injector.n_spins,)
        np.testing.assert_array_equal(bias, np.zeros(injector.n_spins))


# ---------------------------------------------------------------------------
# REQ-VERIFY-095: inject_into_coupling_matrix adds bias to diagonal
# ---------------------------------------------------------------------------


class TestInjectIntoCouplingMatrix:
    """REQ-VERIFY-095: injection adds bias to diagonal; off-diagonals unchanged."""

    def test_diagonal_modified(
        self,
        injector: IsingConstraintInjector,
        fake_J: np.ndarray,
        fake_embeddings: list[list[float]],
    ) -> None:
        """inject_into_coupling_matrix adds bias exactly to the diagonal.

        Spec: REQ-VERIFY-095
        """
        bias = injector.project_to_spin_bias(fake_embeddings)
        J_injected = injector.inject_into_coupling_matrix(fake_J, bias)
        expected_diag = fake_J.diagonal() + bias
        np.testing.assert_allclose(J_injected.diagonal(), expected_diag)

    def test_off_diagonal_unchanged(
        self,
        injector: IsingConstraintInjector,
        fake_J: np.ndarray,
        fake_embeddings: list[list[float]],
    ) -> None:
        """inject_into_coupling_matrix must not touch off-diagonal entries.

        Spec: REQ-VERIFY-095
        """
        bias = injector.project_to_spin_bias(fake_embeddings)
        J_injected = injector.inject_into_coupling_matrix(fake_J, bias)
        # Zero out diagonal for comparison
        J_orig_nodiag = fake_J.copy()
        np.fill_diagonal(J_orig_nodiag, 0.0)
        J_inj_nodiag = J_injected.copy()
        np.fill_diagonal(J_inj_nodiag, 0.0)
        np.testing.assert_array_equal(J_orig_nodiag, J_inj_nodiag)

    def test_original_not_mutated(
        self,
        injector: IsingConstraintInjector,
        fake_J: np.ndarray,
        fake_embeddings: list[list[float]],
    ) -> None:
        """inject_into_coupling_matrix must not mutate the input J.

        Spec: REQ-VERIFY-095-3 (additive, not in-place)
        """
        J_copy = fake_J.copy()
        bias = injector.project_to_spin_bias(fake_embeddings)
        injector.inject_into_coupling_matrix(fake_J, bias)
        np.testing.assert_array_equal(fake_J, J_copy)

    def test_zero_bias_returns_identical_J(
        self,
        injector: IsingConstraintInjector,
        fake_J: np.ndarray,
    ) -> None:
        """inject with zero bias must return a matrix numerically equal to J.

        Spec: REQ-VERIFY-095-3
        """
        bias = np.zeros(injector.n_spins)
        J_injected = injector.inject_into_coupling_matrix(fake_J, bias)
        np.testing.assert_array_equal(J_injected, fake_J)


# ---------------------------------------------------------------------------
# REQ-VERIFY-095/096: compute_energy_with_injection returns scalar
# ---------------------------------------------------------------------------


class TestComputeEnergyWithInjection:
    """REQ-VERIFY-095: energy computation returns scalar; REQ-VERIFY-096: energy changes."""

    def test_returns_float(
        self,
        injector: IsingConstraintInjector,
        fake_ising: object,
        spins: np.ndarray,
        fake_embeddings: list[list[float]],
    ) -> None:
        """compute_energy_with_injection must return a Python float scalar.

        Spec: REQ-VERIFY-095
        """
        energy = injector.compute_energy_with_injection(fake_ising, spins, fake_embeddings)
        assert isinstance(energy, float)

    def test_energy_differs_from_no_injection(
        self,
        injector: IsingConstraintInjector,
        fake_ising: object,
        fake_J: np.ndarray,
        spins: np.ndarray,
        fake_embeddings: list[list[float]],
    ) -> None:
        """Energy WITH injection must differ from energy WITHOUT when bias is non-zero.

        This is the core RETRO-CONSTRAINT-ZERO-DELTA test: if injection has no
        effect on energy, the delta would be 0.0 — exactly the failure mode in Exp 801.

        Spec: REQ-VERIFY-096
        """
        energy_no = float(-0.5 * spins @ np.array(fake_J) @ spins)
        energy_yes = injector.compute_energy_with_injection(fake_ising, spins, fake_embeddings)
        # With non-zero embeddings, bias is non-zero; energy must differ from baseline.
        assert energy_yes != energy_no, (
            "Energy with injection must differ from energy without injection. "
            "If this fails, the bias projection is zero — check _projection init."
        )

    def test_empty_embeddings_gives_same_energy(
        self,
        injector: IsingConstraintInjector,
        fake_ising: object,
        fake_J: np.ndarray,
        spins: np.ndarray,
    ) -> None:
        """Empty embedding list must yield the same energy as the unmodified J.

        Spec: REQ-VERIFY-095-3 (additive — no change when no constraints retrieved)
        """
        energy_no = float(-0.5 * spins @ np.array(fake_J) @ spins)
        energy_empty = injector.compute_energy_with_injection(fake_ising, spins, [])
        assert pytest.approx(energy_empty, rel=1e-9) == energy_no


# ---------------------------------------------------------------------------
# REQ-VERIFY-095: verify() additive when params are None
# ---------------------------------------------------------------------------


class TestVerifyAdditiveBackcompat:
    """REQ-VERIFY-095-3: verify() must behave identically when injector/store are None."""

    def test_verify_without_injector_is_additive(self) -> None:
        """VerifyRepairPipeline.verify() with ising_constraint_injector=None must not crash.

        We do not create a full pipeline here (model loading is expensive).
        Instead, we verify that IsingConstraintInjector.compute_energy_with_injection
        with an empty embedding list returns the original energy — confirming the
        None-path does not alter behaviour.

        Spec: REQ-VERIFY-095-3
        """
        inj = IsingConstraintInjector(embedding_dim=4, n_spins=4)

        class _Tiny:
            coupling = np.eye(4)

        spins = np.array([1.0, -1.0, 1.0, -1.0])
        energy_no_store = float(-0.5 * spins @ np.eye(4) @ spins)
        energy_empty = inj.compute_energy_with_injection(_Tiny(), spins, [])
        assert pytest.approx(energy_empty, rel=1e-9) == energy_no_store
