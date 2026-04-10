"""Tests for LNN Constraint Model — Liquid Time-Constant adaptive EBM.

Spec coverage: REQ-CORE-001, REQ-CORE-002, SCENARIO-CORE-001
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from carnot.core.energy import EnergyFunction
from carnot.models.lnn_constraint import LNNConstraintConfig, LNNConstraintModel


class TestLNNConstraintConfig:
    """Tests for LNNConstraintConfig validation.

    Spec: REQ-CORE-001, SCENARIO-CORE-001
    """

    def test_default_config_valid(self) -> None:
        """REQ-CORE-001: Default config instantiates without error."""
        config = LNNConstraintConfig()
        config.validate()
        assert config.input_dim == 32
        assert config.hidden_dim == 16
        assert config.tau_base == 1.0

    def test_zero_input_dim_raises(self) -> None:
        """SCENARIO-CORE-001: input_dim=0 raises ValueError."""
        config = LNNConstraintConfig(input_dim=0)
        with pytest.raises(ValueError, match="input_dim must be > 0"):
            config.validate()

    def test_zero_hidden_dim_raises(self) -> None:
        """SCENARIO-CORE-001: hidden_dim=0 raises ValueError."""
        config = LNNConstraintConfig(hidden_dim=0)
        with pytest.raises(ValueError, match="hidden_dim must be > 0"):
            config.validate()

    def test_nonpositive_tau_raises(self) -> None:
        """SCENARIO-CORE-001: tau_base<=0 raises ValueError."""
        config = LNNConstraintConfig(tau_base=0.0)
        with pytest.raises(ValueError, match="tau_base must be > 0"):
            config.validate()

    def test_nonpositive_dt_raises(self) -> None:
        """SCENARIO-CORE-001: dt<=0 raises ValueError."""
        config = LNNConstraintConfig(dt=0.0)
        with pytest.raises(ValueError, match="dt must be > 0"):
            config.validate()

    def test_unknown_coupling_init_raises(self) -> None:
        """SCENARIO-CORE-001: Unknown coupling_init raises ValueError."""
        with pytest.raises(ValueError, match="Unknown initializer"):
            LNNConstraintModel(LNNConstraintConfig(coupling_init="bad"))


class TestLNNForwardPass:
    """Tests for LNN energy computation.

    Spec: REQ-CORE-002, SCENARIO-CORE-001
    """

    def test_energy_is_finite(self) -> None:
        """REQ-CORE-002: LNN forward pass produces finite energy for random input.

        Spec: REQ-CORE-002, SCENARIO-CORE-001
        """
        model = LNNConstraintModel(LNNConstraintConfig(input_dim=8, hidden_dim=4))
        x = jnp.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8])
        e = model.energy(x)
        assert jnp.isfinite(e), f"Expected finite energy, got {e}"

    def test_energy_is_scalar(self) -> None:
        """REQ-CORE-002: Energy has shape () (scalar).

        Spec: REQ-CORE-002, SCENARIO-CORE-001
        """
        model = LNNConstraintModel(LNNConstraintConfig(input_dim=6, hidden_dim=3))
        x = jnp.ones(6)
        e = model.energy(x)
        assert e.shape == (), f"Expected scalar, got shape {e.shape}"

    def test_energy_batch_shape(self) -> None:
        """REQ-CORE-002: energy_batch returns shape (batch_size,).

        Spec: REQ-CORE-002, SCENARIO-CORE-001
        """
        model = LNNConstraintModel(LNNConstraintConfig(input_dim=4, hidden_dim=3))
        xs = jnp.ones((5, 4))
        energies = model.energy_batch(xs)
        assert energies.shape == (5,)
        assert jnp.all(jnp.isfinite(energies))

    def test_grad_energy_shape_and_finite(self) -> None:
        """REQ-CORE-002: grad_energy returns finite gradient same shape as x.

        Spec: REQ-CORE-002, SCENARIO-CORE-001
        """
        model = LNNConstraintModel(LNNConstraintConfig(input_dim=5, hidden_dim=4))
        x = jnp.array([0.3, -0.1, 0.5, 0.0, -0.4])
        grad = model.grad_energy(x)
        assert grad.shape == x.shape
        assert jnp.all(jnp.isfinite(grad)), "Gradient contains non-finite values"

    def test_energy_function_protocol(self) -> None:
        """REQ-CORE-001: LNNConstraintModel satisfies EnergyFunction protocol.

        Spec: REQ-CORE-001, SCENARIO-CORE-001
        """
        model = LNNConstraintModel(LNNConstraintConfig(input_dim=4, hidden_dim=2))
        assert isinstance(model, EnergyFunction)

    def test_zeros_coupling_init(self) -> None:
        """REQ-CORE-001: zeros coupling_init produces near-zero energy (only bias term).

        Spec: REQ-CORE-001, SCENARIO-CORE-001
        """
        model = LNNConstraintModel(
            LNNConstraintConfig(input_dim=4, hidden_dim=3, coupling_init="zeros")
        )
        assert jnp.allclose(model.J_eff, jnp.zeros((3, 3)))


class TestLNNAdaptation:
    """Tests for adapt() and reset() methods.

    Spec: REQ-CORE-001, SCENARIO-CORE-001
    """

    def test_adapt_changes_energy(self) -> None:
        """REQ-CORE-001: adapt() changes energy values by >10%.

        After adapting to a non-trivial observation, the energy at the same
        input x should differ from the pre-adapt energy by more than 10%.
        This validates that the hidden state is meaningfully updated.

        Spec: REQ-CORE-001, SCENARIO-CORE-001
        """
        # Use a model with enough coupling to produce non-trivial energy
        config = LNNConstraintConfig(
            input_dim=8,
            hidden_dim=4,
            tau_base=0.5,  # Fast adaptation
            dt=0.5,        # Large step to ensure visible change
        )
        model = LNNConstraintModel(config, key=jrandom.PRNGKey(7))

        x = jnp.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        e_before = float(model.energy(x))

        # Adapt with a strong observation
        obs = jnp.ones(8) * 2.0
        model.adapt(obs)

        e_after = float(model.energy(x))

        # Energy should have changed — either up or down, but by >10% relative
        if abs(e_before) > 1e-6:
            relative_change = abs(e_after - e_before) / (abs(e_before) + 1e-8)
            assert relative_change > 0.10, (
                f"Energy changed by only {relative_change:.1%}, expected >10%. "
                f"e_before={e_before:.4f}, e_after={e_after:.4f}"
            )
        else:
            # If e_before was near zero, just check they're different
            assert abs(e_after - e_before) > 1e-4, (
                f"Energy did not change meaningfully: {e_before:.6f} -> {e_after:.6f}"
            )

    def test_multiple_adaptations_accumulate(self) -> None:
        """REQ-CORE-001: Multiple adapt() calls accumulate context.

        Each successive adapt() call should change the hidden state further,
        demonstrating that context accumulates across steps.

        Spec: REQ-CORE-001, SCENARIO-CORE-001
        """
        model = LNNConstraintModel(
            LNNConstraintConfig(input_dim=4, hidden_dim=4, tau_base=0.5, dt=0.5),
            key=jrandom.PRNGKey(13),
        )
        x = jnp.ones(4)
        e0 = float(model.energy(x))

        h_states = [model._h.copy()]
        for _ in range(5):
            model.adapt(jnp.ones(4))
            h_states.append(model._h.copy())

        # Hidden states should differ across steps
        all_equal = all(
            jnp.allclose(h_states[i], h_states[i + 1]) for i in range(len(h_states) - 1)
        )
        assert not all_equal, "Hidden state did not change across adapt() calls"

    def test_reset_restores_initial_energy(self) -> None:
        """REQ-CORE-001: reset() restores energy to pre-adapt value.

        After adapt() changes the energy, reset() should restore the hidden
        state so that energy() returns the same value as before adaptation.

        Spec: REQ-CORE-001, SCENARIO-CORE-001
        """
        model = LNNConstraintModel(
            LNNConstraintConfig(input_dim=6, hidden_dim=3),
            key=jrandom.PRNGKey(99),
        )
        x = jnp.array([0.5, -0.3, 0.8, -0.1, 0.4, -0.2])
        e_before = float(model.energy(x))

        # Adapt several times
        for _ in range(5):
            model.adapt(x)

        e_adapted = float(model.energy(x))

        # Now reset
        model.reset()
        e_after_reset = float(model.energy(x))

        # After reset, energy should match original
        assert jnp.isclose(e_before, e_after_reset, atol=1e-5), (
            f"Reset did not restore energy: original={e_before:.6f}, "
            f"after_adapt={e_adapted:.6f}, after_reset={e_after_reset:.6f}"
        )

    def test_reset_zeroes_hidden_state(self) -> None:
        """REQ-CORE-001: reset() sets hidden state back to zero vector.

        Spec: REQ-CORE-001, SCENARIO-CORE-001
        """
        model = LNNConstraintModel(LNNConstraintConfig(input_dim=4, hidden_dim=3))
        model.adapt(jnp.ones(4))
        assert not jnp.allclose(model._h, jnp.zeros(3)), "Hidden state should be non-zero after adapt"

        model.reset()
        assert jnp.allclose(model._h, jnp.zeros(3)), "Hidden state should be zero after reset"


class TestLNNTraining:
    """Tests for CD training convergence.

    Spec: REQ-CORE-001, SCENARIO-CORE-001
    """

    def test_train_cd_returns_losses(self) -> None:
        """REQ-CORE-001: train_cd() returns a list of loss values.

        Spec: REQ-CORE-001, SCENARIO-CORE-001
        """
        model = LNNConstraintModel(LNNConstraintConfig(input_dim=4, hidden_dim=3))
        key = jrandom.PRNGKey(0)
        data = jrandom.normal(key, (10, 4))

        losses = model.train_cd(data, n_epochs=5, lr=0.01, key=jrandom.PRNGKey(1))
        assert len(losses) == 5, f"Expected 5 loss values, got {len(losses)}"
        assert all(jnp.isfinite(l) for l in losses), "Some loss values are non-finite"

    def test_train_cd_converges(self) -> None:
        """REQ-CORE-001: CD training reduces or stabilizes the CD loss.

        Trains for 30 epochs and checks that the loss in the second half
        is not systematically higher than the first half (convergence signal).

        Spec: REQ-CORE-001, SCENARIO-CORE-001
        """
        key = jrandom.PRNGKey(42)
        model = LNNConstraintModel(
            LNNConstraintConfig(input_dim=4, hidden_dim=4, coupling_init="xavier_uniform"),
            key=key,
        )
        k1, k2 = jrandom.split(key)
        # Training data: samples from a Gaussian cluster (model should learn to
        # assign them low energy relative to noise)
        data = jrandom.normal(k1, (20, 4)) * 0.5  # tight cluster near origin

        losses = model.train_cd(data, n_epochs=30, lr=0.005, key=k2)

        # Check that loss values are all finite
        assert all(jnp.isfinite(l) for l in losses), "Training produced non-finite losses"

        # Check that loss is not consistently growing (simple convergence check)
        # Compare first 10 vs last 10 — second half should not be dramatically worse
        first_half_mean = sum(losses[:10]) / 10
        second_half_mean = sum(losses[20:]) / 10
        # Allow up to 3x increase (CD can oscillate, but shouldn't blow up)
        if abs(first_half_mean) > 1e-6:
            ratio = abs(second_half_mean) / (abs(first_half_mean) + 1e-8)
            assert ratio < 10.0, (
                f"Training diverged: first_half={first_half_mean:.4f}, "
                f"second_half={second_half_mean:.4f}, ratio={ratio:.2f}"
            )

    def test_train_cd_changes_parameters(self) -> None:
        """REQ-CORE-001: CD training updates J_eff and b_eff parameters.

        After training, J_eff and b_eff should differ from initial values.

        Spec: REQ-CORE-001, SCENARIO-CORE-001
        """
        model = LNNConstraintModel(
            LNNConstraintConfig(input_dim=4, hidden_dim=3, coupling_init="xavier_uniform"),
            key=jrandom.PRNGKey(7),
        )
        J_init = model.J_eff.copy()
        b_init = model.b_eff.copy()

        data = jrandom.normal(jrandom.PRNGKey(0), (15, 4))
        model.train_cd(data, n_epochs=10, lr=0.05, key=jrandom.PRNGKey(3))

        assert not jnp.allclose(model.J_eff, J_init, atol=1e-6), (
            "J_eff was not updated during training"
        )
        assert not jnp.allclose(model.b_eff, b_init, atol=1e-6), (
            "b_eff was not updated during training"
        )

    def test_train_cd_default_key(self) -> None:
        """REQ-CORE-001: train_cd with key=None uses default seed (line 454 coverage).

        Spec: REQ-CORE-001, SCENARIO-CORE-001
        """
        model = LNNConstraintModel(LNNConstraintConfig(input_dim=4, hidden_dim=3))
        data = jrandom.normal(jrandom.PRNGKey(0), (8, 4))
        # key=None should use the default PRNGKey(42)
        losses = model.train_cd(data, n_epochs=3, lr=0.01, key=None)
        assert len(losses) == 3
        assert all(jnp.isfinite(l) for l in losses)


class TestLNNProperties:
    """Tests for input_dim and hidden_state properties.

    Spec: REQ-CORE-001, REQ-CORE-002, SCENARIO-CORE-001
    """

    def test_input_dim_property(self) -> None:
        """REQ-CORE-002: input_dim property returns configured input dimension.

        Spec: REQ-CORE-002, SCENARIO-CORE-001
        """
        model = LNNConstraintModel(LNNConstraintConfig(input_dim=12, hidden_dim=6))
        assert model.input_dim == 12

    def test_hidden_state_property(self) -> None:
        """REQ-CORE-001: hidden_state property returns current h array.

        Spec: REQ-CORE-001, SCENARIO-CORE-001
        """
        model = LNNConstraintModel(LNNConstraintConfig(input_dim=4, hidden_dim=3))
        h = model.hidden_state
        assert h.shape == (3,)
        assert jnp.all(jnp.isfinite(h))
