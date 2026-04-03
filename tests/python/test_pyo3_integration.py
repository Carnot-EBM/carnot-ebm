"""Integration tests for PyO3 Rust bindings.

These tests verify that the Rust EBM implementations are correctly
exposed to Python via PyO3. They require the Rust extension to be
built first:

    PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop -p carnot-python

If the extension is not available, all tests are gracefully skipped.

Spec coverage: REQ-CORE-005, SCENARIO-CORE-005
"""

import numpy as np
import pytest

# Skip entire module if Rust extension is not built
carnot_rust = pytest.importorskip(
    "carnot._rust",
    reason="Rust extension not built. Run: maturin develop -p carnot-python",
)


class TestRustIsingModel:
    """Tests for RustIsingModel PyO3 binding — REQ-CORE-005, REQ-TIER-001."""

    def test_creation_default(self) -> None:
        """REQ-CORE-005: Rust Ising model creates with defaults."""
        model = carnot_rust.RustIsingModel()
        assert model.input_dim() == 784

    def test_creation_custom(self) -> None:
        """REQ-CORE-005: Rust Ising model creates with custom dim."""
        model = carnot_rust.RustIsingModel(input_dim=10)
        assert model.input_dim() == 10

    def test_energy_finite(self) -> None:
        """SCENARIO-CORE-005: energy returns finite scalar."""
        model = carnot_rust.RustIsingModel(input_dim=10)
        x = np.random.randn(10).astype(np.float32)
        e = model.energy(x)
        assert np.isfinite(e)

    def test_energy_batch(self) -> None:
        """SCENARIO-CORE-005: batch energy returns correct shape."""
        model = carnot_rust.RustIsingModel(input_dim=5)
        xs = np.random.randn(4, 5).astype(np.float32)
        energies = model.energy_batch(xs)
        assert energies.shape == (4,)
        assert np.all(np.isfinite(energies))

    def test_grad_energy(self) -> None:
        """SCENARIO-CORE-005: gradient returns correct shape and finite values."""
        model = carnot_rust.RustIsingModel(input_dim=5)
        x = np.random.randn(5).astype(np.float32)
        grad = model.grad_energy(x)
        assert grad.shape == (5,)
        assert np.all(np.isfinite(grad))

    def test_parameter_memory(self) -> None:
        """SCENARIO-CORE-005: parameter memory is positive."""
        model = carnot_rust.RustIsingModel(input_dim=10)
        mem = model.parameter_memory_bytes()
        assert mem > 0


class TestRustGibbsModel:
    """Tests for RustGibbsModel PyO3 binding — REQ-CORE-005, REQ-TIER-002."""

    def test_creation_default(self) -> None:
        """REQ-CORE-005: Rust Gibbs model creates with defaults."""
        model = carnot_rust.RustGibbsModel()
        assert model.input_dim() == 784

    def test_creation_custom(self) -> None:
        """REQ-CORE-005: Rust Gibbs model creates with custom config."""
        model = carnot_rust.RustGibbsModel(
            input_dim=10, hidden_dims=[8, 4], activation="relu"
        )
        assert model.input_dim() == 10

    def test_all_activations(self) -> None:
        """REQ-CORE-005: all three activations work."""
        for act in ["silu", "relu", "tanh"]:
            model = carnot_rust.RustGibbsModel(
                input_dim=5, hidden_dims=[4, 3], activation=act
            )
            x = np.random.randn(5).astype(np.float32)
            e = model.energy(x)
            assert np.isfinite(e), f"Energy not finite for activation={act}"

    def test_energy_batch(self) -> None:
        """SCENARIO-CORE-005: batch energy returns correct shape."""
        model = carnot_rust.RustGibbsModel(input_dim=5, hidden_dims=[4, 3])
        xs = np.random.randn(4, 5).astype(np.float32)
        energies = model.energy_batch(xs)
        assert energies.shape == (4,)
        assert np.all(np.isfinite(energies))

    def test_grad_energy(self) -> None:
        """SCENARIO-CORE-005: gradient returns correct shape and finite values."""
        model = carnot_rust.RustGibbsModel(input_dim=5, hidden_dims=[4, 3])
        x = np.random.randn(5).astype(np.float32)
        grad = model.grad_energy(x)
        assert grad.shape == (5,)
        assert np.all(np.isfinite(grad))

    def test_invalid_activation_raises(self) -> None:
        """SCENARIO-CORE-005: invalid activation raises Python error."""
        with pytest.raises(ValueError, match="Unknown activation"):
            carnot_rust.RustGibbsModel(input_dim=5, hidden_dims=[4], activation="gelu")


class TestRustBoltzmannModel:
    """Tests for RustBoltzmannModel PyO3 binding — REQ-CORE-005, REQ-TIER-003."""

    def test_creation_default(self) -> None:
        """REQ-CORE-005: Rust Boltzmann model creates with defaults."""
        model = carnot_rust.RustBoltzmannModel()
        assert model.input_dim() == 784

    def test_creation_custom(self) -> None:
        """REQ-CORE-005: Rust Boltzmann model creates with custom config."""
        model = carnot_rust.RustBoltzmannModel(
            input_dim=10, hidden_dims=[8, 6, 4]
        )
        assert model.input_dim() == 10

    def test_energy_finite(self) -> None:
        """SCENARIO-CORE-005: energy returns finite scalar."""
        model = carnot_rust.RustBoltzmannModel(input_dim=10, hidden_dims=[8, 6, 4])
        x = np.random.randn(10).astype(np.float32)
        e = model.energy(x)
        assert np.isfinite(e)

    def test_energy_batch(self) -> None:
        """SCENARIO-CORE-005: batch energy returns correct shape."""
        model = carnot_rust.RustBoltzmannModel(input_dim=5, hidden_dims=[4, 3])
        xs = np.random.randn(4, 5).astype(np.float32)
        energies = model.energy_batch(xs)
        assert energies.shape == (4,)
        assert np.all(np.isfinite(energies))

    def test_grad_energy(self) -> None:
        """SCENARIO-CORE-005: gradient returns correct shape and finite values."""
        model = carnot_rust.RustBoltzmannModel(input_dim=5, hidden_dims=[4, 3])
        x = np.random.randn(5).astype(np.float32)
        grad = model.grad_energy(x)
        assert grad.shape == (5,)
        assert np.all(np.isfinite(grad))

    def test_no_residual(self) -> None:
        """REQ-CORE-005: Boltzmann works without residual connections."""
        model = carnot_rust.RustBoltzmannModel(
            input_dim=5, hidden_dims=[4, 3], residual=False
        )
        x = np.random.randn(5).astype(np.float32)
        e = model.energy(x)
        assert np.isfinite(e)


class TestRustSamplers:
    """Tests for Rust sampler PyO3 bindings — REQ-CORE-005, REQ-SAMPLE-001/002."""

    def test_langevin_ising(self) -> None:
        """REQ-CORE-005: Langevin samples from Rust Ising model."""
        model = carnot_rust.RustIsingModel(input_dim=5)
        sampler = carnot_rust.RustLangevinSampler(step_size=0.01)
        init = np.zeros(5, dtype=np.float32)
        result = sampler.sample_ising(model, init, 100)
        assert result.shape == (5,)
        assert np.all(np.isfinite(result))

    def test_langevin_gibbs(self) -> None:
        """REQ-CORE-005: Langevin samples from Rust Gibbs model."""
        model = carnot_rust.RustGibbsModel(input_dim=5, hidden_dims=[4, 3])
        sampler = carnot_rust.RustLangevinSampler(step_size=0.01)
        init = np.zeros(5, dtype=np.float32)
        result = sampler.sample_gibbs(model, init, 100)
        assert result.shape == (5,)
        assert np.all(np.isfinite(result))

    def test_langevin_boltzmann(self) -> None:
        """REQ-CORE-005: Langevin samples from Rust Boltzmann model."""
        model = carnot_rust.RustBoltzmannModel(input_dim=5, hidden_dims=[4, 3])
        sampler = carnot_rust.RustLangevinSampler(step_size=0.01)
        init = np.zeros(5, dtype=np.float32)
        result = sampler.sample_boltzmann(model, init, 100)
        assert result.shape == (5,)
        assert np.all(np.isfinite(result))

    def test_hmc_ising(self) -> None:
        """REQ-CORE-005: HMC samples from Rust Ising model."""
        model = carnot_rust.RustIsingModel(input_dim=5)
        sampler = carnot_rust.RustHMCSampler(step_size=0.01, num_leapfrog_steps=5)
        init = np.zeros(5, dtype=np.float32)
        result = sampler.sample_ising(model, init, 50)
        assert result.shape == (5,)
        assert np.all(np.isfinite(result))

    def test_hmc_gibbs(self) -> None:
        """REQ-CORE-005: HMC samples from Rust Gibbs model."""
        model = carnot_rust.RustGibbsModel(input_dim=5, hidden_dims=[4, 3])
        sampler = carnot_rust.RustHMCSampler(step_size=0.01, num_leapfrog_steps=5)
        init = np.zeros(5, dtype=np.float32)
        result = sampler.sample_gibbs(model, init, 50)
        assert result.shape == (5,)
        assert np.all(np.isfinite(result))

    def test_hmc_boltzmann(self) -> None:
        """REQ-CORE-005: HMC samples from Rust Boltzmann model."""
        model = carnot_rust.RustBoltzmannModel(input_dim=5, hidden_dims=[4, 3])
        sampler = carnot_rust.RustHMCSampler(step_size=0.01, num_leapfrog_steps=5)
        init = np.zeros(5, dtype=np.float32)
        result = sampler.sample_boltzmann(model, init, 50)
        assert result.shape == (5,)
        assert np.all(np.isfinite(result))
