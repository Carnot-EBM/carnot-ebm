"""Tests for layer-selective EBM hallucination monitoring.

Spec coverage: REQ-INFER-015, SCENARIO-INFER-015-001 through SCENARIO-INFER-015-010
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

from carnot.embeddings.layer_ebm import (
    LayerEBMConfig,
    LayerEBMVerifier,
    _concat_critical_activations,
    build_layer_ebm_verifier,
    identify_critical_layers,
    train_layer_ebm,
)
from carnot.inference.learned_verifier import LearnedVerifierConfig
from carnot.models.gibbs import GibbsModel
from carnot.verify.constraint import ComposedEnergy


# ---------------------------------------------------------------------------
# Helpers: synthetic per-layer activation data
# ---------------------------------------------------------------------------


def _make_layer_activations(
    key: jax.Array,
    n_layers: int = 6,
    n_correct: int = 20,
    n_halluc: int = 20,
    hidden_dim: int = 8,
    informative_layers: tuple[int, ...] = (1, 3, 5),
    separation: float = 4.0,
    noise_scale: float = 0.3,
) -> tuple[dict[int, list[jax.Array]], dict[int, list[jax.Array]], tuple[int, ...]]:
    """Generate synthetic per-layer activations with known informative layers.

    Layers in ``informative_layers`` have a clear separation between
    correct and hallucinated samples. Other layers have overlapping
    distributions (noise only).

    Returns:
        (activations_correct, activations_hallucinated, informative_layers)
    """
    acts_correct: dict[int, list[jax.Array]] = {}
    acts_halluc: dict[int, list[jax.Array]] = {}

    for layer_idx in range(n_layers):
        key, k1, k2, k3 = jrandom.split(key, 4)

        # Random shift direction for informative layers.
        raw_dir = jrandom.normal(k1, (hidden_dim,))
        direction = raw_dir / jnp.linalg.norm(raw_dir)

        correct_list: list[jax.Array] = []
        halluc_list: list[jax.Array] = []

        for i in range(n_correct):
            key, sk = jrandom.split(key)
            correct_list.append(jrandom.normal(sk, (hidden_dim,)) * noise_scale)

        for i in range(n_halluc):
            key, sk = jrandom.split(key)
            base = jrandom.normal(sk, (hidden_dim,)) * noise_scale
            if layer_idx in informative_layers:
                base = base + separation * direction
            halluc_list.append(base)

        acts_correct[layer_idx] = correct_list
        acts_halluc[layer_idx] = halluc_list

    return acts_correct, acts_halluc, informative_layers


def _make_2d_layer_activations(
    key: jax.Array,
    n_layers: int = 4,
    n_samples: int = 10,
    seq_len: int = 5,
    hidden_dim: int = 8,
    informative_layers: tuple[int, ...] = (1, 3),
    separation: float = 4.0,
) -> tuple[dict[int, list[jax.Array]], dict[int, list[jax.Array]]]:
    """Generate activations with shape (seq_len, hidden_dim) per sample."""
    acts_correct: dict[int, list[jax.Array]] = {}
    acts_halluc: dict[int, list[jax.Array]] = {}

    for layer_idx in range(n_layers):
        key, k1 = jrandom.split(key)
        direction = jrandom.normal(k1, (hidden_dim,))
        direction = direction / jnp.linalg.norm(direction)

        correct_list: list[jax.Array] = []
        halluc_list: list[jax.Array] = []

        for _ in range(n_samples):
            key, sk1, sk2 = jrandom.split(key, 3)
            correct_list.append(jrandom.normal(sk1, (seq_len, hidden_dim)) * 0.3)
            base = jrandom.normal(sk2, (seq_len, hidden_dim)) * 0.3
            if layer_idx in informative_layers:
                base = base + separation * direction[None, :]
            halluc_list.append(base)

        acts_correct[layer_idx] = correct_list
        acts_halluc[layer_idx] = halluc_list

    return acts_correct, acts_halluc


# ---------------------------------------------------------------------------
# LayerEBMConfig validation
# ---------------------------------------------------------------------------


class TestLayerEBMConfig:
    """Tests for SCENARIO-INFER-015-001: Config validation."""

    def test_default_values(self) -> None:
        """SCENARIO-INFER-015-001: Defaults are sensible."""
        config = LayerEBMConfig()
        assert config.top_k_layers == 3
        assert config.hidden_dims == [64, 32]
        assert config.n_epochs == 100
        assert config.learning_rate == 0.01

    def test_validate_passes(self) -> None:
        """SCENARIO-INFER-015-001: Valid config passes validation."""
        config = LayerEBMConfig(top_k_layers=2, hidden_dims=[32])
        config.validate()

    def test_validate_top_k_zero(self) -> None:
        """SCENARIO-INFER-015-001: top_k_layers=0 rejected."""
        config = LayerEBMConfig(top_k_layers=0)
        with pytest.raises(ValueError, match="top_k_layers must be >= 1"):
            config.validate()

    def test_validate_empty_hidden_dims(self) -> None:
        """SCENARIO-INFER-015-001: Empty hidden_dims rejected."""
        config = LayerEBMConfig(hidden_dims=[])
        with pytest.raises(ValueError, match="hidden_dims must be non-empty"):
            config.validate()

    def test_validate_negative_hidden_dim(self) -> None:
        """SCENARIO-INFER-015-001: Negative hidden dim rejected."""
        config = LayerEBMConfig(hidden_dims=[64, -1])
        with pytest.raises(ValueError, match="hidden_dims must be non-empty"):
            config.validate()

    def test_validate_zero_epochs(self) -> None:
        """SCENARIO-INFER-015-001: n_epochs=0 rejected."""
        config = LayerEBMConfig(n_epochs=0)
        with pytest.raises(ValueError, match="n_epochs must be >= 1"):
            config.validate()

    def test_validate_negative_lr(self) -> None:
        """SCENARIO-INFER-015-001: learning_rate <= 0 rejected."""
        config = LayerEBMConfig(learning_rate=0.0)
        with pytest.raises(ValueError, match="learning_rate must be > 0"):
            config.validate()


# ---------------------------------------------------------------------------
# identify_critical_layers
# ---------------------------------------------------------------------------


class TestIdentifyCriticalLayers:
    """Tests for REQ-INFER-015: Critical layer identification via Fisher criterion."""

    def test_returns_correct_informative_layers(self) -> None:
        """REQ-INFER-015: Identified layers match the known informative layers."""
        key = jrandom.PRNGKey(42)
        acts_c, acts_h, true_informative = _make_layer_activations(
            key, n_layers=6, informative_layers=(1, 3, 5), separation=6.0
        )
        found = identify_critical_layers(acts_c, acts_h, top_k=3)
        assert set(found) == set(true_informative)

    def test_returns_sorted_by_fisher_score(self) -> None:
        """REQ-INFER-015: Layers are sorted by descending Fisher score."""
        key = jrandom.PRNGKey(7)
        # Layer 2 gets very high separation, layer 4 medium, others none.
        acts_c: dict[int, list[jax.Array]] = {}
        acts_h: dict[int, list[jax.Array]] = {}
        for layer_idx in range(5):
            acts_c[layer_idx] = []
            acts_h[layer_idx] = []
            for i in range(30):
                key, sk = jrandom.split(key)
                acts_c[layer_idx].append(jrandom.normal(sk, (8,)) * 0.3)
                key, sk = jrandom.split(key)
                shift = 0.0
                if layer_idx == 2:
                    shift = 10.0
                elif layer_idx == 4:
                    shift = 5.0
                acts_h[layer_idx].append(
                    jrandom.normal(sk, (8,)) * 0.3 + shift * jnp.ones(8)
                )

        found = identify_critical_layers(acts_c, acts_h, top_k=2)
        assert found[0] == 2, "Highest Fisher score layer should be first"
        assert found[1] == 4

    def test_top_k_clamped_to_available(self) -> None:
        """REQ-INFER-015: top_k > n_layers returns all layers."""
        key = jrandom.PRNGKey(0)
        acts_c, acts_h, _ = _make_layer_activations(key, n_layers=2)
        found = identify_critical_layers(acts_c, acts_h, top_k=10)
        assert len(found) == 2

    def test_empty_correct_raises(self) -> None:
        """REQ-INFER-015: Empty correct activations raises ValueError."""
        with pytest.raises(ValueError, match="Both activation dicts must be non-empty"):
            identify_critical_layers({}, {0: [jnp.ones(4)]})

    def test_empty_halluc_raises(self) -> None:
        """REQ-INFER-015: Empty hallucinated activations raises ValueError."""
        with pytest.raises(ValueError, match="Both activation dicts must be non-empty"):
            identify_critical_layers({0: [jnp.ones(4)]}, {})

    def test_no_common_layers_raises(self) -> None:
        """REQ-INFER-015: No overlapping layer indices raises ValueError."""
        with pytest.raises(ValueError, match="No common layers"):
            identify_critical_layers(
                {0: [jnp.ones(4)]},
                {1: [jnp.ones(4)]},
            )

    def test_handles_2d_activations(self) -> None:
        """REQ-INFER-015: Works with (seq_len, hidden_dim) activations."""
        key = jrandom.PRNGKey(3)
        acts_c, acts_h = _make_2d_layer_activations(
            key, n_layers=4, informative_layers=(1, 3), separation=6.0
        )
        found = identify_critical_layers(acts_c, acts_h, top_k=2)
        assert set(found) == {1, 3}

    def test_single_sample_per_class(self) -> None:
        """REQ-INFER-015: Works with just one sample per class."""
        acts_c = {0: [jnp.zeros(4)], 1: [jnp.zeros(4)]}
        acts_h = {0: [jnp.ones(4) * 5.0], 1: [jnp.ones(4) * 0.1]}
        found = identify_critical_layers(acts_c, acts_h, top_k=1)
        assert found[0] == 0  # Layer 0 has larger separation

    def test_empty_layer_list_gets_zero_score(self) -> None:
        """REQ-INFER-015: Layer with empty sample list gets Fisher score 0."""
        acts_c = {0: [jnp.ones(4)], 1: []}
        acts_h = {0: [jnp.ones(4) * 3.0], 1: []}
        found = identify_critical_layers(acts_c, acts_h, top_k=2)
        assert found[0] == 0


# ---------------------------------------------------------------------------
# _concat_critical_activations
# ---------------------------------------------------------------------------


class TestConcatCriticalActivations:
    """Tests for internal concatenation helper."""

    def test_output_shape_1d(self) -> None:
        """REQ-INFER-015: 1-D activations concatenated correctly."""
        acts = {
            0: [jnp.ones(4), jnp.zeros(4)],
            1: [jnp.ones(8) * 2, jnp.zeros(8)],
        }
        result = _concat_critical_activations(acts, [0, 1])
        assert result.shape == (2, 12)  # 4 + 8

    def test_output_shape_2d(self) -> None:
        """REQ-INFER-015: 2-D activations pooled then concatenated."""
        acts = {
            0: [jnp.ones((3, 4)), jnp.zeros((3, 4))],
            1: [jnp.ones((3, 4)) * 2, jnp.zeros((3, 4))],
        }
        result = _concat_critical_activations(acts, [0, 1])
        assert result.shape == (2, 8)  # 4 + 4

    def test_values_correct(self) -> None:
        """REQ-INFER-015: Values are correct after concatenation."""
        acts = {
            0: [jnp.array([1.0, 2.0])],
            1: [jnp.array([3.0, 4.0])],
        }
        result = _concat_critical_activations(acts, [0, 1])
        np.testing.assert_allclose(result[0], [1.0, 2.0, 3.0, 4.0])


# ---------------------------------------------------------------------------
# train_layer_ebm
# ---------------------------------------------------------------------------


class TestTrainLayerEBM:
    """Tests for REQ-INFER-015: Training the layer-selective EBM."""

    def test_returns_gibbs_model(self) -> None:
        """REQ-INFER-015: Returns a GibbsModel."""
        key = jrandom.PRNGKey(0)
        correct = jrandom.normal(key, (20, 16))
        key, sk = jrandom.split(key)
        halluc = jrandom.normal(sk, (20, 16)) + 3.0
        config = LearnedVerifierConfig(
            hidden_dims=[16, 8], n_epochs=10, learning_rate=0.01
        )
        model = train_layer_ebm(correct, halluc, config)
        assert isinstance(model, GibbsModel)
        assert model.config.input_dim == 16

    def test_trained_model_discriminates(self) -> None:
        """REQ-INFER-015: Trained model assigns lower energy to correct activations."""
        key = jrandom.PRNGKey(42)
        k1, k2 = jrandom.split(key)
        correct = jrandom.normal(k1, (50, 8)) * 0.5
        halluc = jrandom.normal(k2, (50, 8)) * 0.5 + 3.0

        config = LearnedVerifierConfig(
            hidden_dims=[16, 8], n_epochs=200, learning_rate=0.01
        )
        model = train_layer_ebm(correct, halluc, config)

        # Evaluate on held-out data.
        key, k3, k4 = jrandom.split(key, 3)
        test_correct = jrandom.normal(k3, (20, 8)) * 0.5
        test_halluc = jrandom.normal(k4, (20, 8)) * 0.5 + 3.0

        correct_energies = [float(model.energy(test_correct[i])) for i in range(20)]
        halluc_energies = [float(model.energy(test_halluc[i])) for i in range(20)]

        mean_correct = np.mean(correct_energies)
        mean_halluc = np.mean(halluc_energies)

        assert mean_correct < mean_halluc, (
            f"Correct energy ({mean_correct:.4f}) should be lower than "
            f"hallucinated energy ({mean_halluc:.4f})"
        )

    def test_default_config_none(self) -> None:
        """REQ-INFER-015: Works with config=None (uses default LearnedVerifierConfig)."""
        key = jrandom.PRNGKey(0)
        correct = jrandom.normal(key, (10, 8))
        key, sk = jrandom.split(key)
        halluc = jrandom.normal(sk, (10, 8)) + 2.0
        # config=None triggers the default branch (line 315).
        # Patch n_epochs after to keep test fast — we just need to hit the branch.
        model = train_layer_ebm(correct, halluc, None)
        assert isinstance(model, GibbsModel)


# ---------------------------------------------------------------------------
# LayerEBMVerifier
# ---------------------------------------------------------------------------


class TestLayerEBMVerifier:
    """Tests for REQ-INFER-015: Combined verifier constraint."""

    def _make_verifier(self) -> tuple[LayerEBMVerifier, int]:
        """Create a simple verifier for testing."""
        from carnot.models.gibbs import GibbsConfig

        dim = 8
        model = GibbsModel(GibbsConfig(input_dim=dim, hidden_dims=[8]))
        direction = jnp.ones(dim) / jnp.sqrt(float(dim))
        verifier = LayerEBMVerifier(
            critical_layers=[0, 1],
            model=model,
            direction=direction,
            threshold=1.0,
            constraint_name="test_layer_ebm",
        )
        return verifier, dim

    def test_name_property(self) -> None:
        """REQ-INFER-015: Name is accessible."""
        v, _ = self._make_verifier()
        assert v.name == "test_layer_ebm"

    def test_critical_layers_property(self) -> None:
        """REQ-INFER-015: Critical layers are accessible."""
        v, _ = self._make_verifier()
        assert v.critical_layers == [0, 1]

    def test_satisfaction_threshold(self) -> None:
        """REQ-INFER-015: Threshold is accessible."""
        v, _ = self._make_verifier()
        assert v.satisfaction_threshold == 1.0

    def test_energy_returns_scalar(self) -> None:
        """REQ-INFER-015: Energy output is scalar."""
        v, dim = self._make_verifier()
        x = jnp.ones(dim)
        e = v.energy(x)
        assert e.shape == ()

    def test_energy_combines_ebm_and_direction(self) -> None:
        """REQ-INFER-015: Energy is sum of EBM energy + direction energy."""
        from carnot.embeddings.hallucination_direction import hallucination_energy
        from carnot.models.gibbs import GibbsConfig

        dim = 4
        model = GibbsModel(GibbsConfig(input_dim=dim, hidden_dims=[4]))
        direction = jnp.array([1.0, 0.0, 0.0, 0.0])
        v = LayerEBMVerifier([0], model, direction)

        x = jnp.array([2.0, 1.0, 0.5, 0.3])
        expected = float(model.energy(x)) + float(hallucination_energy(x, direction))
        np.testing.assert_allclose(float(v.energy(x)), expected, atol=1e-5)

    def test_energy_from_layers_1d(self) -> None:
        """REQ-INFER-015: energy_from_layers works with 1-D activations."""
        v, _ = self._make_verifier()
        activations = {
            0: jnp.ones(4),
            1: jnp.ones(4) * 2,
        }
        e = v.energy_from_layers(activations)
        assert e.shape == ()

    def test_energy_from_layers_2d(self) -> None:
        """REQ-INFER-015: energy_from_layers pools 2-D activations."""
        v, _ = self._make_verifier()
        activations = {
            0: jnp.ones((3, 4)),
            1: jnp.ones((3, 4)) * 2,
        }
        e = v.energy_from_layers(activations)
        assert e.shape == ()

    def test_energy_from_layers_matches_manual(self) -> None:
        """REQ-INFER-015: energy_from_layers matches manual concat + energy."""
        v, _ = self._make_verifier()
        activations = {
            0: jnp.array([1.0, 2.0, 3.0, 4.0]),
            1: jnp.array([5.0, 6.0, 7.0, 8.0]),
        }
        e_layers = v.energy_from_layers(activations)
        x_concat = jnp.concatenate([activations[0], activations[1]])
        e_manual = v.energy(x_concat)
        np.testing.assert_allclose(float(e_layers), float(e_manual), atol=1e-6)

    def test_grad_energy_shape(self) -> None:
        """REQ-INFER-015: Gradient has same shape as input."""
        v, dim = self._make_verifier()
        x = jnp.ones(dim)
        g = v.grad_energy(x)
        assert g.shape == (dim,)

    def test_is_satisfied_below_threshold(self) -> None:
        """REQ-INFER-015: Satisfied when energy < threshold."""
        from carnot.models.gibbs import GibbsConfig

        dim = 4
        # Zero-initialized model: energy starts at 0 for any input.
        model = GibbsModel(GibbsConfig(input_dim=dim, hidden_dims=[4]))
        # Direction orthogonal to test input so direction energy ~ 0.
        direction = jnp.array([0.0, 0.0, 1.0, 0.0])
        v = LayerEBMVerifier([0], model, direction, threshold=1.0)
        x = jnp.array([1.0, 0.0, 0.0, 0.0])
        assert v.is_satisfied(x)

    def test_composable_with_composed_energy(self) -> None:
        """REQ-INFER-015: Integrates with ComposedEnergy."""
        v, dim = self._make_verifier()
        composed = ComposedEnergy(input_dim=dim)
        composed.add_constraint(v, weight=1.0)
        x = jnp.ones(dim)
        result = composed.verify(x)
        # Just verify it runs without error.
        assert result is not None


# ---------------------------------------------------------------------------
# build_layer_ebm_verifier (end-to-end factory)
# ---------------------------------------------------------------------------


class TestBuildLayerEBMVerifier:
    """Tests for REQ-INFER-015: End-to-end verifier construction."""

    def test_returns_verifier(self) -> None:
        """REQ-INFER-015: Factory returns a LayerEBMVerifier."""
        key = jrandom.PRNGKey(0)
        acts_c, acts_h, _ = _make_layer_activations(
            key, n_layers=6, informative_layers=(1, 3, 5), separation=5.0
        )
        config = LayerEBMConfig(
            top_k_layers=2, hidden_dims=[16, 8], n_epochs=10
        )
        verifier = build_layer_ebm_verifier(acts_c, acts_h, config)
        assert isinstance(verifier, LayerEBMVerifier)
        assert len(verifier.critical_layers) == 2

    def test_default_config_none(self) -> None:
        """REQ-INFER-015: Works with config=None (hits line 540)."""
        key = jrandom.PRNGKey(1)
        acts_c, acts_h, _ = _make_layer_activations(
            key, n_layers=4, informative_layers=(0, 2), separation=5.0
        )
        # config=None triggers the default LayerEBMConfig() branch.
        verifier = build_layer_ebm_verifier(acts_c, acts_h, None)
        assert isinstance(verifier, LayerEBMVerifier)

    def test_verifier_discriminates(self) -> None:
        """REQ-INFER-015: End-to-end verifier gives higher energy to hallucinated."""
        key = jrandom.PRNGKey(99)
        acts_c, acts_h, info_layers = _make_layer_activations(
            key,
            n_layers=6,
            n_correct=40,
            n_halluc=40,
            hidden_dim=8,
            informative_layers=(1, 3),
            separation=6.0,
        )
        config = LayerEBMConfig(
            top_k_layers=2, hidden_dims=[16, 8], n_epochs=100
        )
        verifier = build_layer_ebm_verifier(acts_c, acts_h, config)

        # Generate held-out test activations.
        key, k1, k2 = jrandom.split(key, 3)
        correct_energies: list[float] = []
        halluc_energies: list[float] = []

        for i in range(10):
            key, sk = jrandom.split(key)
            # Correct: near origin on critical layers.
            acts_test: dict[int, jax.Array] = {}
            for layer_idx in range(6):
                key, sk = jrandom.split(key)
                acts_test[layer_idx] = jrandom.normal(sk, (8,)) * 0.3
            correct_energies.append(float(verifier.energy_from_layers(acts_test)))

            # Hallucinated: shifted on informative layers.
            acts_test_h: dict[int, jax.Array] = {}
            for layer_idx in range(6):
                key, sk = jrandom.split(key)
                base = jrandom.normal(sk, (8,)) * 0.3
                if layer_idx in info_layers:
                    base = base + 6.0 * jnp.ones(8)
                acts_test_h[layer_idx] = base
            halluc_energies.append(float(verifier.energy_from_layers(acts_test_h)))

        mean_c = np.mean(correct_energies)
        mean_h = np.mean(halluc_energies)
        assert mean_h > mean_c, (
            f"Hallucinated mean energy ({mean_h:.4f}) should exceed "
            f"correct mean energy ({mean_c:.4f})"
        )

    def test_with_2d_activations(self) -> None:
        """REQ-INFER-015: Factory works with 2-D (seq_len, hidden_dim) activations."""
        key = jrandom.PRNGKey(5)
        acts_c, acts_h = _make_2d_layer_activations(
            key, n_layers=4, informative_layers=(1, 3), separation=5.0
        )
        config = LayerEBMConfig(
            top_k_layers=2, hidden_dims=[16], n_epochs=5
        )
        verifier = build_layer_ebm_verifier(acts_c, acts_h, config)
        assert isinstance(verifier, LayerEBMVerifier)


# ---------------------------------------------------------------------------
# Package-level exports
# ---------------------------------------------------------------------------


class TestPackageExports:
    """Tests for package-level imports of layer EBM symbols."""

    def test_layer_ebm_exports_from_embeddings(self) -> None:
        """REQ-INFER-015: Layer EBM symbols accessible from carnot.embeddings."""
        from carnot.embeddings import (
            LayerEBMConfig as PkgConfig,
            LayerEBMVerifier as PkgVerifier,
            build_layer_ebm_verifier as pkg_build,
            identify_critical_layers as pkg_identify,
            train_layer_ebm as pkg_train,
        )

        assert PkgConfig is LayerEBMConfig
        assert PkgVerifier is LayerEBMVerifier
        assert pkg_build is build_layer_ebm_verifier
        assert pkg_identify is identify_critical_layers
        assert pkg_train is train_layer_ebm
