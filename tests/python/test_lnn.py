"""Tests for LiquidConstraintModel — MLP-driven coupling matrix ODE.

Coverage target: 100% of python/carnot/models/lnn.py.

Spec coverage: REQ-CORE-001, REQ-CORE-002, SCENARIO-CORE-001
"""

import jax.numpy as jnp
import jax.random as jrandom
import pytest

from carnot.core.energy import EnergyFunction
from carnot.models.lnn import LiquidConstraintConfig, LiquidConstraintModel


# ── Config validation ──────────────────────────────────────────────────────


class TestLiquidConstraintConfig:
    """Tests for LiquidConstraintConfig.validate().

    Spec: REQ-CORE-001, SCENARIO-CORE-001
    """

    def test_default_config_valid(self) -> None:
        """REQ-CORE-001: Default config validates without error."""
        config = LiquidConstraintConfig()
        config.validate()
        assert config.input_dim == 8
        assert config.mlp_hidden_dim == 16
        assert config.dt == 0.1
        assert config.coupling_init == "xavier_uniform"

    def test_zero_input_dim_raises(self) -> None:
        """SCENARIO-CORE-001: input_dim=0 raises ValueError."""
        config = LiquidConstraintConfig(input_dim=0)
        with pytest.raises(ValueError, match="input_dim must be > 0"):
            config.validate()

    def test_zero_mlp_hidden_dim_raises(self) -> None:
        """SCENARIO-CORE-001: mlp_hidden_dim=0 raises ValueError."""
        config = LiquidConstraintConfig(mlp_hidden_dim=0)
        with pytest.raises(ValueError, match="mlp_hidden_dim must be > 0"):
            config.validate()

    def test_zero_dt_raises(self) -> None:
        """SCENARIO-CORE-001: dt=0 raises ValueError."""
        config = LiquidConstraintConfig(dt=0.0)
        with pytest.raises(ValueError, match="dt must be > 0"):
            config.validate()

    def test_unknown_coupling_init_raises(self) -> None:
        """SCENARIO-CORE-001: Unknown coupling_init raises ValueError at construction."""
        with pytest.raises(ValueError, match="Unknown initializer"):
            LiquidConstraintModel(LiquidConstraintConfig(coupling_init="bad"))


# ── Construction ───────────────────────────────────────────────────────────


class TestLiquidConstraintModelConstruction:
    """Tests for __init__ parameter shapes and defaults.

    Spec: REQ-CORE-001, SCENARIO-CORE-001
    """

    def test_default_key_uses_seed_zero(self) -> None:
        """REQ-CORE-001: key=None produces a valid model (seed 0 path)."""
        model = LiquidConstraintModel(LiquidConstraintConfig(input_dim=4, mlp_hidden_dim=8))
        assert model.J.shape == (4, 4)
        assert model.b.shape == (4,)

    def test_explicit_key(self) -> None:
        """REQ-CORE-001: Explicit PRNG key produces valid model."""
        model = LiquidConstraintModel(
            LiquidConstraintConfig(input_dim=6, mlp_hidden_dim=8),
            key=jrandom.PRNGKey(42),
        )
        assert model.W1.shape == (8, 6)

    def test_zeros_coupling_init(self) -> None:
        """REQ-CORE-001: zeros coupling_init produces all-zero J0 and J."""
        model = LiquidConstraintModel(
            LiquidConstraintConfig(input_dim=4, mlp_hidden_dim=4, coupling_init="zeros")
        )
        assert jnp.allclose(model.J0, jnp.zeros((4, 4)))
        assert jnp.allclose(model.J, jnp.zeros((4, 4)))

    def test_xavier_coupling_is_symmetric(self) -> None:
        """REQ-CORE-001: xavier_uniform initialisation produces symmetric J0."""
        model = LiquidConstraintModel(
            LiquidConstraintConfig(input_dim=5, mlp_hidden_dim=8, coupling_init="xavier_uniform"),
            key=jrandom.PRNGKey(7),
        )
        assert jnp.allclose(model.J0, model.J0.T)

    def test_initial_bias_is_zero(self) -> None:
        """REQ-CORE-001: Initial bias b0 and b are all zeros."""
        model = LiquidConstraintModel(LiquidConstraintConfig(input_dim=4, mlp_hidden_dim=4))
        assert jnp.allclose(model.b0, jnp.zeros(4))
        assert jnp.allclose(model.b, jnp.zeros(4))

    def test_mlp_weights_shapes(self) -> None:
        """REQ-CORE-001: MLP weight matrices have correct shapes."""
        d, h = 4, 8
        model = LiquidConstraintModel(LiquidConstraintConfig(input_dim=d, mlp_hidden_dim=h))
        assert model.W1.shape == (h, d)
        assert model.b1.shape == (h,)
        assert model.W2_J.shape == (d * d, h)
        assert model.b2_J.shape == (d * d,)
        assert model.W2_b.shape == (d, h)
        assert model.b2_b.shape == (d,)


# ── Forward pass (energy, energy_batch, grad_energy) ──────────────────────


class TestLiquidConstraintModelForward:
    """Tests for energy(), energy_batch(), grad_energy().

    Spec: REQ-CORE-002, SCENARIO-CORE-001
    """

    def test_energy_is_scalar(self) -> None:
        """REQ-CORE-002: energy() returns a 0-D scalar."""
        model = LiquidConstraintModel(LiquidConstraintConfig(input_dim=4, mlp_hidden_dim=4))
        x = jnp.ones(4)
        e = model.energy(x)
        assert e.shape == ()

    def test_energy_is_finite(self) -> None:
        """REQ-CORE-002: energy() returns a finite value for typical inputs."""
        model = LiquidConstraintModel(LiquidConstraintConfig(input_dim=6, mlp_hidden_dim=8))
        x = jnp.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
        assert jnp.isfinite(model.energy(x))

    def test_energy_batch_shape(self) -> None:
        """REQ-CORE-002: energy_batch() returns shape (batch_size,) via AutoGradMixin."""
        model = LiquidConstraintModel(LiquidConstraintConfig(input_dim=4, mlp_hidden_dim=4))
        xs = jnp.ones((7, 4))
        energies = model.energy_batch(xs)
        assert energies.shape == (7,)
        assert jnp.all(jnp.isfinite(energies))

    def test_grad_energy_shape_and_finite(self) -> None:
        """REQ-CORE-002: grad_energy() returns finite gradient same shape as x."""
        model = LiquidConstraintModel(LiquidConstraintConfig(input_dim=5, mlp_hidden_dim=6))
        x = jnp.array([0.3, -0.1, 0.5, 0.0, -0.4])
        grad = model.grad_energy(x)
        assert grad.shape == x.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_energy_function_protocol(self) -> None:
        """REQ-CORE-001: LiquidConstraintModel satisfies EnergyFunction protocol."""
        model = LiquidConstraintModel(LiquidConstraintConfig(input_dim=4, mlp_hidden_dim=4))
        assert isinstance(model, EnergyFunction)

    def test_input_dim_property(self) -> None:
        """REQ-CORE-002: input_dim property returns config value."""
        model = LiquidConstraintModel(LiquidConstraintConfig(input_dim=10, mlp_hidden_dim=8))
        assert model.input_dim == 10

    def test_zeros_coupling_energy_only_bias(self) -> None:
        """REQ-CORE-001: With zeros init, energy = -b^T x (b is zero too → energy = 0)."""
        model = LiquidConstraintModel(
            LiquidConstraintConfig(input_dim=4, mlp_hidden_dim=4, coupling_init="zeros")
        )
        x = jnp.ones(4)
        # J=0, b=0 → E(x) = 0
        assert jnp.isclose(model.energy(x), jnp.array(0.0))


# ── step() ─────────────────────────────────────────────────────────────────


class TestLiquidConstraintModelStep:
    """Tests for step() method.

    Spec: REQ-CORE-001, SCENARIO-CORE-001
    """

    def test_step_changes_J(self) -> None:
        """REQ-CORE-001: step() changes the coupling matrix J."""
        model = LiquidConstraintModel(
            LiquidConstraintConfig(input_dim=4, mlp_hidden_dim=8, dt=1.0),
            key=jrandom.PRNGKey(1),
        )
        J_before = model.J.copy()
        # Use a large observation to trigger a meaningful dJ
        obs = jnp.ones(4) * 10.0
        model.step(obs)
        # J should have changed
        assert not jnp.allclose(model.J, J_before), "J was not updated by step()"

    def test_step_keeps_J_symmetric(self) -> None:
        """REQ-CORE-001: J remains symmetric after step()."""
        model = LiquidConstraintModel(
            LiquidConstraintConfig(input_dim=4, mlp_hidden_dim=8),
            key=jrandom.PRNGKey(3),
        )
        model.step(jnp.array([1.0, -1.0, 2.0, -2.0]))
        assert jnp.allclose(model.J, model.J.T, atol=1e-6)

    def test_step_changes_energy(self) -> None:
        """REQ-CORE-001: Energy at the same state differs before and after step()."""
        model = LiquidConstraintModel(
            LiquidConstraintConfig(input_dim=4, mlp_hidden_dim=8, dt=1.0),
            key=jrandom.PRNGKey(5),
        )
        x = jnp.array([1.0, -1.0, 0.5, -0.5])
        e_before = float(model.energy(x))
        model.step(x * 10.0)
        e_after = float(model.energy(x))
        assert e_before != pytest.approx(e_after, rel=1e-4), (
            f"Energy did not change after step: {e_before} → {e_after}"
        )

    def test_multiple_steps_accumulate(self) -> None:
        """REQ-CORE-001: Successive step() calls change J cumulatively."""
        model = LiquidConstraintModel(
            LiquidConstraintConfig(input_dim=4, mlp_hidden_dim=8, dt=1.0),
            key=jrandom.PRNGKey(9),
        )
        obs = jnp.ones(4) * 5.0
        J_states = [model.J]
        for _ in range(4):
            model.step(obs)
            J_states.append(model.J)

        # Not all J states should be identical
        all_equal = all(jnp.allclose(J_states[i], J_states[i + 1]) for i in range(4))
        assert not all_equal, "J did not accumulate across step() calls"


# ── reset() ────────────────────────────────────────────────────────────────


class TestLiquidConstraintModelReset:
    """Tests for reset() method.

    Spec: REQ-CORE-001, SCENARIO-CORE-001
    """

    def test_reset_restores_J_to_J0(self) -> None:
        """REQ-CORE-001: reset() sets J back to J0."""
        model = LiquidConstraintModel(
            LiquidConstraintConfig(input_dim=4, mlp_hidden_dim=6),
            key=jrandom.PRNGKey(11),
        )
        for _ in range(3):
            model.step(jnp.ones(4))
        assert not jnp.allclose(model.J, model.J0), "J should differ from J0 after step()"

        model.reset()
        assert jnp.allclose(model.J, model.J0), "reset() did not restore J to J0"

    def test_reset_restores_b_to_b0(self) -> None:
        """REQ-CORE-001: reset() sets b back to b0."""
        model = LiquidConstraintModel(
            LiquidConstraintConfig(input_dim=4, mlp_hidden_dim=6),
            key=jrandom.PRNGKey(13),
        )
        model.step(jnp.ones(4) * 10.0)
        model.reset()
        assert jnp.allclose(model.b, model.b0), "reset() did not restore b to b0"

    def test_reset_restores_energy(self) -> None:
        """REQ-CORE-001: Energy after reset() equals energy before any step()."""
        model = LiquidConstraintModel(
            LiquidConstraintConfig(input_dim=4, mlp_hidden_dim=6),
            key=jrandom.PRNGKey(17),
        )
        x = jnp.array([0.5, -0.3, 0.8, -0.1])
        e_initial = float(model.energy(x))

        for _ in range(5):
            model.step(x)

        model.reset()
        e_after_reset = float(model.energy(x))

        assert jnp.isclose(e_initial, e_after_reset, atol=1e-5), (
            f"reset() did not restore energy: initial={e_initial}, after_reset={e_after_reset}"
        )


# ── train() ────────────────────────────────────────────────────────────────


class TestLiquidConstraintModelTrain:
    """Tests for train() method.

    Spec: REQ-CORE-001, SCENARIO-CORE-001
    """

    def _make_sequences(
        self,
        n_seqs: int = 4,
        seq_len: int = 3,
        d: int = 4,
        seed: int = 0,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Create small synthetic sequences for training tests."""
        key = jrandom.PRNGKey(seed)
        k1, k2 = jrandom.split(key)
        observations = jrandom.normal(k1, (n_seqs, seq_len, d))
        # Alternate labels: +1 for even sequences, -1 for odd
        labels = jnp.where(jnp.arange(n_seqs)[:, None] % 2 == 0, 1.0, -1.0)
        labels = jnp.broadcast_to(labels, (n_seqs, seq_len))
        return observations, labels

    def test_train_returns_correct_n_losses(self) -> None:
        """REQ-CORE-001: train() returns one loss value per epoch."""
        model = LiquidConstraintModel(LiquidConstraintConfig(input_dim=4, mlp_hidden_dim=4))
        obs, lbl = self._make_sequences()
        losses = model.train(obs, lbl, n_epochs=5, lr=0.01)
        assert len(losses) == 5

    def test_train_losses_are_finite(self) -> None:
        """REQ-CORE-001: All loss values returned by train() are finite."""
        model = LiquidConstraintModel(
            LiquidConstraintConfig(input_dim=4, mlp_hidden_dim=4),
            key=jrandom.PRNGKey(21),
        )
        obs, lbl = self._make_sequences(seed=21)
        losses = model.train(obs, lbl, n_epochs=10, lr=0.005)
        assert all(float(l) == float(l) for l in losses), "Some losses are NaN"  # NaN != NaN
        assert all(jnp.isfinite(jnp.array(l)) for l in losses)

    def test_train_changes_mlp_weights(self) -> None:
        """REQ-CORE-001: train() updates W1 (and other MLP weights)."""
        model = LiquidConstraintModel(
            LiquidConstraintConfig(input_dim=4, mlp_hidden_dim=4, coupling_init="xavier_uniform"),
            key=jrandom.PRNGKey(31),
        )
        W1_before = model.W1.copy()
        obs, lbl = self._make_sequences(seed=31)
        model.train(obs, lbl, n_epochs=5, lr=0.1)
        assert not jnp.allclose(model.W1, W1_before, atol=1e-8), (
            "W1 was not updated by train()"
        )

    def test_train_does_not_affect_J0(self) -> None:
        """REQ-CORE-001: train() must not change J0 (the initial coupling prior)."""
        model = LiquidConstraintModel(
            LiquidConstraintConfig(input_dim=4, mlp_hidden_dim=4),
            key=jrandom.PRNGKey(41),
        )
        J0_before = model.J0.copy()
        obs, lbl = self._make_sequences(seed=41)
        model.train(obs, lbl, n_epochs=5, lr=0.01)
        assert jnp.allclose(model.J0, J0_before), "train() should not modify J0"

    def test_train_convergence_no_divergence(self) -> None:
        """REQ-CORE-001: Loss does not blow up over 20 epochs (no divergence)."""
        key = jrandom.PRNGKey(51)
        model = LiquidConstraintModel(
            LiquidConstraintConfig(input_dim=4, mlp_hidden_dim=8),
            key=key,
        )
        obs, lbl = self._make_sequences(n_seqs=6, seq_len=4, d=4, seed=51)
        losses = model.train(obs, lbl, n_epochs=20, lr=0.005)
        # All losses should be finite — no NaN/Inf from gradient explosion
        assert all(jnp.isfinite(jnp.array(l)) for l in losses), (
            f"Training diverged: {losses}"
        )

    def test_step_and_train_interop(self) -> None:
        """REQ-CORE-001: After train(), step() still works and reset() restores J."""
        model = LiquidConstraintModel(
            LiquidConstraintConfig(input_dim=4, mlp_hidden_dim=4),
            key=jrandom.PRNGKey(61),
        )
        obs, lbl = self._make_sequences(seed=61)
        model.train(obs, lbl, n_epochs=3, lr=0.01)

        # After training, step() should still mutate J
        J_post_train = model.J.copy()
        model.step(jnp.ones(4))
        assert not jnp.allclose(model.J, J_post_train), "step() had no effect after training"

        # And reset() should restore J to J0 (not to J_post_train)
        model.reset()
        assert jnp.allclose(model.J, model.J0)
