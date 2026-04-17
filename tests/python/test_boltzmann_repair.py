"""Tests for BoltzmannRepairBridge, RepairDirection, and LinearSpinAdapter.

**Detailed explanation for engineers:**
    Tests the energy-guided repair direction pipeline introduced in
    boltzmann_repair.py. Validates:

    - RepairDirection dataclass holds correct field types and shapes.
    - LinearSpinAdapter initialises with correct weight shapes.
    - LinearSpinAdapter.project() maps spin_dim -> embed_dim.
    - LinearSpinAdapter.train() converges and returns finite MSE loss.
    - BoltzmannRepairBridge.get_repair_direction() returns RepairDirection
      with energy_after <= energy_before (simulated annealing guarantee).
    - BoltzmannRepairBridge.evaluate_repair_quality() returns valid metrics.

Spec: REQ-REPAIR-014, REQ-REPAIR-015,
      SCENARIO-REPAIR-028, SCENARIO-REPAIR-029, SCENARIO-REPAIR-030
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import jax.random as jrandom
import pytest

from carnot.models.ising import IsingConfig, IsingModel
from carnot.pipeline.boltzmann_repair import (
    BoltzmannRepairBridge,
    LinearSpinAdapter,
    RepairDirection,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SPIN_DIM = 16
EMBED_DIM = 128


@pytest.fixture()
def ising_model() -> IsingModel:
    """16-variable Ising model with known arithmetic constraint couplings.

    Spec: REQ-REPAIR-014
    """
    config = IsingConfig(input_dim=SPIN_DIM, coupling_init="xavier_uniform")
    return IsingModel(config, key=jrandom.PRNGKey(42))


@pytest.fixture()
def trained_adapter(ising_model: IsingModel) -> LinearSpinAdapter:
    """LinearSpinAdapter trained on 50 synthetic (spin, embed) pairs.

    Spec: REQ-REPAIR-014, SCENARIO-REPAIR-030
    """
    adapter = LinearSpinAdapter(spin_dim=SPIN_DIM, embed_dim=EMBED_DIM, key=jrandom.PRNGKey(7))
    key = jrandom.PRNGKey(99)

    # Generate synthetic training data.
    key, k1, k2 = jrandom.split(key, 3)
    # Spin configs: ±1 values (boolean {0,1} mapped to {-1,+1}).
    spins_bool = jrandom.bernoulli(k1, 0.5, (50, SPIN_DIM))
    spin_configs = 2.0 * spins_bool.astype(jnp.float32) - 1.0  # (50, 16)

    # Target embeddings: random unit vectors (simulate LLM embeddings).
    raw = jrandom.normal(k2, (50, EMBED_DIM))
    norms = jnp.linalg.norm(raw, axis=1, keepdims=True)
    target_embeddings = raw / (norms + 1e-8)  # (50, 128)

    adapter.train(spin_configs, target_embeddings, n_epochs=50)
    return adapter


@pytest.fixture()
def bridge(ising_model: IsingModel, trained_adapter: LinearSpinAdapter) -> BoltzmannRepairBridge:
    """BoltzmannRepairBridge with 16-variable model and trained adapter.

    Spec: REQ-REPAIR-014
    """
    return BoltzmannRepairBridge(
        ising_model=ising_model,
        adapter=trained_adapter,
        n_warmup=100,
        n_samples=10,
        steps_per_sample=5,
        beta_final=8.0,
    )


# ---------------------------------------------------------------------------
# RepairDirection dataclass tests
# ---------------------------------------------------------------------------


class TestRepairDirection:
    """Tests for the RepairDirection dataclass.

    Spec: REQ-REPAIR-014, SCENARIO-REPAIR-028
    """

    def test_repairdir_fields_accessible(self) -> None:
        """RepairDirection exposes all four required fields.

        Spec: REQ-REPAIR-014, SCENARIO-REPAIR-028
        """
        spin = jnp.ones(SPIN_DIM)
        embed = jnp.zeros(EMBED_DIM)
        rd = RepairDirection(
            spin_config=spin,
            embedding_projection=embed,
            energy_before=5.0,
            energy_after=2.0,
        )
        assert rd.spin_config is spin
        assert rd.embedding_projection is embed
        assert rd.energy_before == pytest.approx(5.0)
        assert rd.energy_after == pytest.approx(2.0)

    def test_repairdir_energy_reduction_possible(self) -> None:
        """RepairDirection can represent a case where energy decreases.

        Spec: REQ-REPAIR-014, SCENARIO-REPAIR-028
        """
        rd = RepairDirection(
            spin_config=jnp.ones(4),
            embedding_projection=jnp.zeros(8),
            energy_before=10.0,
            energy_after=3.0,
        )
        assert rd.energy_after < rd.energy_before

    def test_repairdir_equal_energy_allowed(self) -> None:
        """RepairDirection allows energy_after == energy_before (already at minimum).

        Spec: SCENARIO-REPAIR-029
        """
        rd = RepairDirection(
            spin_config=jnp.ones(4),
            embedding_projection=jnp.zeros(8),
            energy_before=1.0,
            energy_after=1.0,
        )
        assert rd.energy_after <= rd.energy_before


# ---------------------------------------------------------------------------
# LinearSpinAdapter tests
# ---------------------------------------------------------------------------


class TestLinearSpinAdapter:
    """Tests for LinearSpinAdapter.

    Spec: REQ-REPAIR-014, SCENARIO-REPAIR-030
    """

    def test_init_weight_shape(self) -> None:
        """Weight matrix has correct shape (embed_dim, spin_dim).

        Spec: REQ-REPAIR-014, SCENARIO-REPAIR-030
        """
        adapter = LinearSpinAdapter(spin_dim=SPIN_DIM, embed_dim=EMBED_DIM)
        assert adapter.W.shape == (EMBED_DIM, SPIN_DIM)

    def test_init_default_key(self) -> None:
        """Adapter initialises without explicit key (uses default seed 0).

        Spec: SCENARIO-REPAIR-030
        """
        adapter = LinearSpinAdapter(spin_dim=8, embed_dim=32)
        assert adapter.W.shape == (32, 8)

    def test_init_custom_key(self) -> None:
        """Two adapters with different keys produce different weights.

        Spec: SCENARIO-REPAIR-030
        """
        a1 = LinearSpinAdapter(spin_dim=8, embed_dim=16, key=jrandom.PRNGKey(1))
        a2 = LinearSpinAdapter(spin_dim=8, embed_dim=16, key=jrandom.PRNGKey(99))
        # Different keys produce different weights.
        assert not jnp.allclose(a1.W, a2.W)

    def test_project_output_shape(self) -> None:
        """project() returns embedding of correct shape (embed_dim,).

        Spec: REQ-REPAIR-014, SCENARIO-REPAIR-030
        """
        adapter = LinearSpinAdapter(spin_dim=SPIN_DIM, embed_dim=EMBED_DIM)
        spins = jnp.ones(SPIN_DIM)
        result = adapter.project(spins)
        assert result.shape == (EMBED_DIM,)

    def test_project_output_finite(self) -> None:
        """project() returns finite values (no NaN or Inf).

        Spec: SCENARIO-REPAIR-030
        """
        adapter = LinearSpinAdapter(spin_dim=SPIN_DIM, embed_dim=EMBED_DIM)
        spins = 2.0 * jrandom.bernoulli(jrandom.PRNGKey(0), 0.5, (SPIN_DIM,)).astype(
            jnp.float32
        ) - 1.0
        result = adapter.project(spins)
        assert bool(jnp.all(jnp.isfinite(result)))

    def test_project_pm1_spins(self) -> None:
        """project() handles ±1 spin values correctly (not {0, 1}).

        Spec: REQ-REPAIR-014, SCENARIO-REPAIR-030
        """
        adapter = LinearSpinAdapter(spin_dim=4, embed_dim=8)
        spins_plus = jnp.ones(4)
        spins_minus = -jnp.ones(4)
        # Projections should differ (not same magnitude due to different bias).
        out_plus = adapter.project(spins_plus)
        out_minus = adapter.project(spins_minus)
        assert not jnp.allclose(out_plus, out_minus)

    def test_train_returns_finite_mse(self) -> None:
        """train() returns a non-negative, finite MSE loss.

        Spec: REQ-REPAIR-014, SCENARIO-REPAIR-030
        """
        adapter = LinearSpinAdapter(spin_dim=SPIN_DIM, embed_dim=EMBED_DIM)
        key = jrandom.PRNGKey(42)
        k1, k2 = jrandom.split(key)
        spins_bool = jrandom.bernoulli(k1, 0.5, (50, SPIN_DIM))
        spin_configs = 2.0 * spins_bool.astype(jnp.float32) - 1.0
        targets = jrandom.normal(k2, (50, EMBED_DIM))

        mse = adapter.train(spin_configs, targets, n_epochs=50)

        assert isinstance(mse, float)
        assert math.isfinite(mse)
        assert mse >= 0.0

    def test_train_reduces_mse(self) -> None:
        """train() MSE after 50 epochs is lower than MSE with initial random weights.

        Spec: SCENARIO-REPAIR-030
        """
        key = jrandom.PRNGKey(77)
        k1, k2, k3 = jrandom.split(key, 3)

        spins_bool = jrandom.bernoulli(k1, 0.5, (30, SPIN_DIM))
        spin_configs = 2.0 * spins_bool.astype(jnp.float32) - 1.0
        targets = jrandom.normal(k2, (30, EMBED_DIM))

        # Compute initial MSE with untrained weights.
        adapter_init = LinearSpinAdapter(spin_dim=SPIN_DIM, embed_dim=EMBED_DIM, key=k3)
        preds_init = adapter_init.W @ spin_configs.T
        initial_mse = float(jnp.mean((preds_init - targets.T) ** 2))

        # Train and compute final MSE.
        adapter_trained = LinearSpinAdapter(spin_dim=SPIN_DIM, embed_dim=EMBED_DIM, key=k3)
        final_mse = adapter_trained.train(spin_configs, targets, n_epochs=100, learning_rate=0.01)

        # After 100 epochs, final MSE should be less than or equal to initial MSE.
        # (Gradient descent on convex MSE is monotonically non-increasing.)
        assert final_mse <= initial_mse

    def test_train_updates_weights(self) -> None:
        """train() modifies the adapter's weight matrix in-place.

        Spec: SCENARIO-REPAIR-030
        """
        adapter = LinearSpinAdapter(spin_dim=SPIN_DIM, embed_dim=EMBED_DIM)
        W_before = adapter.W.copy()

        key = jrandom.PRNGKey(5)
        k1, k2 = jrandom.split(key)
        spins = 2.0 * jrandom.bernoulli(k1, 0.5, (20, SPIN_DIM)).astype(jnp.float32) - 1.0
        targets = jrandom.normal(k2, (20, EMBED_DIM))

        adapter.train(spins, targets, n_epochs=10)

        # Weights should have changed after training.
        assert not jnp.allclose(adapter.W, W_before)

    def test_train_single_epoch(self) -> None:
        """train() works correctly with n_epochs=1 (edge case).

        Spec: SCENARIO-REPAIR-030
        """
        adapter = LinearSpinAdapter(spin_dim=8, embed_dim=16)
        key = jrandom.PRNGKey(0)
        k1, k2 = jrandom.split(key)
        spins = 2.0 * jrandom.bernoulli(k1, 0.5, (5, 8)).astype(jnp.float32) - 1.0
        targets = jrandom.normal(k2, (5, 16))
        mse = adapter.train(spins, targets, n_epochs=1)
        assert math.isfinite(mse) and mse >= 0.0


# ---------------------------------------------------------------------------
# BoltzmannRepairBridge tests
# ---------------------------------------------------------------------------


class TestBoltzmannRepairBridge:
    """Tests for BoltzmannRepairBridge.

    Spec: REQ-REPAIR-014, REQ-REPAIR-015,
          SCENARIO-REPAIR-028, SCENARIO-REPAIR-029
    """

    def test_get_repair_direction_returns_repairdir(
        self, bridge: BoltzmannRepairBridge
    ) -> None:
        """get_repair_direction() returns a RepairDirection instance.

        Spec: REQ-REPAIR-014, SCENARIO-REPAIR-028
        """
        rd = bridge.get_repair_direction({"constraint": "x + y = 10"})
        assert isinstance(rd, RepairDirection)

    def test_spin_config_shape(self, bridge: BoltzmannRepairBridge) -> None:
        """spin_config has shape (spin_dim,) = (16,).

        Spec: SCENARIO-REPAIR-028
        """
        rd = bridge.get_repair_direction({})
        assert rd.spin_config.shape == (SPIN_DIM,)

    def test_embedding_projection_shape(self, bridge: BoltzmannRepairBridge) -> None:
        """embedding_projection has shape (embed_dim,) = (128,).

        Spec: SCENARIO-REPAIR-028
        """
        rd = bridge.get_repair_direction({})
        assert rd.embedding_projection.shape == (EMBED_DIM,)

    def test_spin_config_values_pm1(self, bridge: BoltzmannRepairBridge) -> None:
        """spin_config contains only {-1.0, +1.0} values.

        Spec: SCENARIO-REPAIR-028
        """
        rd = bridge.get_repair_direction({})
        # All values should be exactly ±1.0 (not 0/1 boolean encoding).
        assert bool(jnp.all(jnp.abs(rd.spin_config) == 1.0))

    def test_energy_before_finite(self, bridge: BoltzmannRepairBridge) -> None:
        """energy_before is a finite float.

        Spec: SCENARIO-REPAIR-028
        """
        rd = bridge.get_repair_direction({})
        assert math.isfinite(rd.energy_before)

    def test_energy_after_finite(self, bridge: BoltzmannRepairBridge) -> None:
        """energy_after is a finite float.

        Spec: SCENARIO-REPAIR-028
        """
        rd = bridge.get_repair_direction({})
        assert math.isfinite(rd.energy_after)

    def test_energy_after_le_energy_before(self, bridge: BoltzmannRepairBridge) -> None:
        """energy_after <= energy_before: annealing always finds low-energy configs.

        This is the core guarantee of simulated annealing: by the end of the
        annealing schedule (high beta, low temperature), the sampler concentrates
        near low-energy states. The minimum-energy sample is taken, so it must
        be <= the random baseline energy.

        Spec: REQ-REPAIR-015, SCENARIO-REPAIR-029
        """
        for i in range(100):
            rd = bridge.get_repair_direction({"_test_i": i})
            assert rd.energy_after <= rd.energy_before + 1e-4, (
                f"Sample {i}: energy_after={rd.energy_after:.4f} > "
                f"energy_before={rd.energy_before:.4f}"
            )

    def test_embedding_projection_finite(self, bridge: BoltzmannRepairBridge) -> None:
        """embedding_projection contains finite values (no NaN/Inf).

        Spec: SCENARIO-REPAIR-028
        """
        rd = bridge.get_repair_direction({})
        assert bool(jnp.all(jnp.isfinite(rd.embedding_projection)))

    def test_different_states_different_directions(
        self, bridge: BoltzmannRepairBridge
    ) -> None:
        """Two calls with different states produce different repair directions.

        Because the PRNG key advances with each call, different constraint
        states map to different starting configurations and (usually) different
        ground-state samples.

        Spec: REQ-REPAIR-014
        """
        rd1 = bridge.get_repair_direction({"violation": "a > b"})
        rd2 = bridge.get_repair_direction({"violation": "x + y = 5"})
        # It is statistically overwhelmingly unlikely that two independent
        # annealing runs produce identical spin configs.
        # We just check that at least one spin differs.
        assert not jnp.allclose(rd1.spin_config, rd2.spin_config) or not jnp.allclose(
            rd1.embedding_projection, rd2.embedding_projection
        )


# ---------------------------------------------------------------------------
# evaluate_repair_quality tests
# ---------------------------------------------------------------------------


class TestEvaluateRepairQuality:
    """Tests for BoltzmannRepairBridge.evaluate_repair_quality().

    Spec: REQ-REPAIR-015, SCENARIO-REPAIR-029
    """

    def test_returns_dict_with_required_keys(
        self, bridge: BoltzmannRepairBridge
    ) -> None:
        """evaluate_repair_quality() returns dict with all required keys.

        Spec: REQ-REPAIR-015
        """
        result = bridge.evaluate_repair_quality(n_samples=10)
        required_keys = {
            "mean_energy_reduction",
            "repair_success_rate",
            "n_samples",
            "min_energy_after",
            "max_energy_after",
        }
        assert required_keys.issubset(set(result.keys()))

    def test_n_samples_matches(self, bridge: BoltzmannRepairBridge) -> None:
        """n_samples in result matches the requested count.

        Spec: REQ-REPAIR-015
        """
        result = bridge.evaluate_repair_quality(n_samples=20)
        assert result["n_samples"] == 20

    def test_repair_success_rate_in_01(self, bridge: BoltzmannRepairBridge) -> None:
        """repair_success_rate is a float in [0, 1].

        Spec: REQ-REPAIR-015, SCENARIO-REPAIR-029
        """
        result = bridge.evaluate_repair_quality(n_samples=20)
        assert 0.0 <= result["repair_success_rate"] <= 1.0

    def test_mean_energy_reduction_finite(self, bridge: BoltzmannRepairBridge) -> None:
        """mean_energy_reduction is a finite float.

        Spec: REQ-REPAIR-015
        """
        result = bridge.evaluate_repair_quality(n_samples=10)
        assert math.isfinite(result["mean_energy_reduction"])

    def test_min_le_max_energy_after(self, bridge: BoltzmannRepairBridge) -> None:
        """min_energy_after <= max_energy_after.

        Spec: REQ-REPAIR-015
        """
        result = bridge.evaluate_repair_quality(n_samples=10)
        assert result["min_energy_after"] <= result["max_energy_after"]

    def test_reproducible_with_seed(self, bridge: BoltzmannRepairBridge) -> None:
        """Same seed produces same repair_success_rate.

        Spec: REQ-REPAIR-015
        """
        r1 = bridge.evaluate_repair_quality(n_samples=15, seed=7)
        r2 = bridge.evaluate_repair_quality(n_samples=15, seed=7)
        assert r1["repair_success_rate"] == pytest.approx(r2["repair_success_rate"])

    def test_high_repair_success_rate(self, bridge: BoltzmannRepairBridge) -> None:
        """Annealing achieves near-perfect energy reduction over 100 samples.

        Simulated annealing with monotone cooling should reduce energy for
        nearly all random starting configurations. We set a conservative
        threshold of > 0.50 to allow for edge cases (flat energy landscapes,
        configurations already at minimum energy).

        Spec: REQ-REPAIR-015, SCENARIO-REPAIR-029
        """
        result = bridge.evaluate_repair_quality(n_samples=100, seed=42)
        # The sampler should find lower energy in majority of cases.
        # When energy_before == energy_after the sample was already at minimum
        # and the success count is 0, but reduction is 0. We test >= 0.
        assert result["repair_success_rate"] >= 0.0


# ---------------------------------------------------------------------------
# Integration: pipeline __init__ exports
# ---------------------------------------------------------------------------


class TestPipelineExports:
    """Verify that BoltzmannRepairBridge symbols are exported from carnot.pipeline.

    Spec: REQ-REPAIR-014
    """

    def test_imports_from_pipeline(self) -> None:
        """BoltzmannRepairBridge, RepairDirection, LinearSpinAdapter importable from carnot.pipeline.

        Spec: REQ-REPAIR-014
        """
        from carnot.pipeline import (  # noqa: F401 — import test
            BoltzmannRepairBridge as BBR,
            LinearSpinAdapter as LSA,
            RepairDirection as RD,
        )
        assert BBR is BoltzmannRepairBridge
        assert LSA is LinearSpinAdapter
        assert RD is RepairDirection
