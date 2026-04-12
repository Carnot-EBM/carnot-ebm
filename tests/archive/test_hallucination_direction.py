"""Tests for hallucination direction detection in activation space.

Spec coverage: REQ-INFER-014, SCENARIO-INFER-014-001 through SCENARIO-INFER-014-006
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

from carnot.embeddings.hallucination_direction import (
    HallucinationDirectionConfig,
    HallucinationDirectionConstraint,
    find_hallucination_direction,
    hallucination_energy,
)
from carnot.verify.constraint import ComposedEnergy


# ---------------------------------------------------------------------------
# Helpers: synthetic activation data
# ---------------------------------------------------------------------------


def _make_synthetic_activations(
    key: jax.Array,
    n_correct: int = 20,
    n_hallucinated: int = 20,
    hidden_dim: int = 16,
    separation: float = 3.0,
) -> tuple[list[jax.Array], list[jax.Array], jax.Array]:
    """Generate synthetic activation clusters separated along a known direction.

    Returns:
        (correct_activations, hallucinated_activations, true_direction)
        where true_direction is the unit vector along which hallucinated
        samples are shifted.
    """
    k1, k2, k3 = jrandom.split(key, 3)

    # Random unit direction for the hallucination shift.
    raw_dir = jrandom.normal(k1, (hidden_dim,))
    true_direction = raw_dir / jnp.linalg.norm(raw_dir)

    # Correct activations: centered at origin with small noise.
    correct_noise = jrandom.normal(k2, (n_correct, hidden_dim)) * 0.5
    correct = [correct_noise[i] for i in range(n_correct)]

    # Hallucinated activations: shifted along true_direction.
    halluc_noise = jrandom.normal(k3, (n_hallucinated, hidden_dim)) * 0.5
    halluc = [halluc_noise[i] + separation * true_direction for i in range(n_hallucinated)]

    return correct, halluc, true_direction


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestHallucinationDirectionConfig:
    """Tests for SCENARIO-INFER-014-001: Configuration validation."""

    def test_default_values(self) -> None:
        """SCENARIO-INFER-014-001: Default config has sensible values."""
        config = HallucinationDirectionConfig()
        assert config.top_k == 1
        assert config.normalize is True

    def test_custom_values(self) -> None:
        """SCENARIO-INFER-014-001: Custom config values are respected."""
        config = HallucinationDirectionConfig(top_k=3, normalize=False)
        assert config.top_k == 3
        assert config.normalize is False

    def test_validate_top_k_zero(self) -> None:
        """SCENARIO-INFER-014-001: top_k=0 is rejected."""
        config = HallucinationDirectionConfig(top_k=0)
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            config.validate()

    def test_validate_top_k_negative(self) -> None:
        """SCENARIO-INFER-014-001: Negative top_k is rejected."""
        config = HallucinationDirectionConfig(top_k=-1)
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            config.validate()

    def test_validate_passes_for_valid(self) -> None:
        """SCENARIO-INFER-014-001: Valid config passes validation."""
        config = HallucinationDirectionConfig(top_k=5)
        config.validate()  # should not raise


# ---------------------------------------------------------------------------
# find_hallucination_direction
# ---------------------------------------------------------------------------


class TestFindHallucinationDirection:
    """Tests for REQ-INFER-014: Finding the hallucination direction."""

    def test_output_shape_single_direction(self) -> None:
        """REQ-INFER-014: Single direction has shape (hidden_dim,)."""
        key = jrandom.PRNGKey(0)
        correct, halluc, _ = _make_synthetic_activations(key, hidden_dim=16)
        direction = find_hallucination_direction(correct, halluc)
        assert direction.shape == (16,)

    def test_output_shape_multi_direction(self) -> None:
        """REQ-INFER-014: Multi-direction has shape (top_k, hidden_dim)."""
        key = jrandom.PRNGKey(0)
        correct, halluc, _ = _make_synthetic_activations(key, hidden_dim=16)
        config = HallucinationDirectionConfig(top_k=3)
        directions = find_hallucination_direction(correct, halluc, config)
        assert directions.shape == (3, 16)

    def test_direction_aligns_with_true_separation(self) -> None:
        """REQ-INFER-014: Discovered direction aligns with true separation axis."""
        key = jrandom.PRNGKey(42)
        correct, halluc, true_dir = _make_synthetic_activations(
            key, n_correct=50, n_hallucinated=50, hidden_dim=8, separation=5.0
        )
        found_dir = find_hallucination_direction(correct, halluc)

        # Cosine similarity between found and true directions should be high.
        cos_sim = float(jnp.abs(jnp.dot(found_dir, true_dir)))
        assert cos_sim > 0.8, f"Cosine similarity {cos_sim:.3f} too low"

    def test_normalized_output_is_unit_vector(self) -> None:
        """REQ-INFER-014: With normalize=True, direction has unit norm."""
        key = jrandom.PRNGKey(1)
        correct, halluc, _ = _make_synthetic_activations(key)
        direction = find_hallucination_direction(
            correct, halluc, HallucinationDirectionConfig(normalize=True)
        )
        norm = float(jnp.linalg.norm(direction))
        np.testing.assert_allclose(norm, 1.0, atol=1e-6)

    def test_unnormalized_output_preserves_magnitude(self) -> None:
        """REQ-INFER-014: With normalize=False, direction is raw difference."""
        key = jrandom.PRNGKey(2)
        correct, halluc, _ = _make_synthetic_activations(key, separation=5.0)
        direction = find_hallucination_direction(
            correct, halluc, HallucinationDirectionConfig(normalize=False)
        )
        norm = float(jnp.linalg.norm(direction))
        # Should be roughly equal to separation (5.0), not 1.0.
        assert norm > 2.0, f"Expected large norm, got {norm:.3f}"

    def test_empty_correct_raises(self) -> None:
        """REQ-INFER-014: Empty correct_activations raises ValueError."""
        with pytest.raises(ValueError, match="correct_activations must not be empty"):
            find_hallucination_direction([], [jnp.ones(4)])

    def test_empty_hallucinated_raises(self) -> None:
        """REQ-INFER-014: Empty hallucinated_activations raises ValueError."""
        with pytest.raises(ValueError, match="hallucinated_activations must not be empty"):
            find_hallucination_direction([jnp.ones(4)], [])

    def test_mismatched_dimensions_raises(self) -> None:
        """REQ-INFER-014: Mismatched activation dimensions raise ValueError."""
        with pytest.raises(ValueError, match="Activation dimensions must match"):
            find_hallucination_direction(
                [jnp.ones(4)],
                [jnp.ones(8)],
            )

    def test_single_sample_each(self) -> None:
        """REQ-INFER-014: Works with just one sample per class."""
        correct = [jnp.array([1.0, 0.0, 0.0])]
        halluc = [jnp.array([0.0, 1.0, 0.0])]
        direction = find_hallucination_direction(correct, halluc)
        assert direction.shape == (3,)

    def test_svd_directions_orthogonal(self) -> None:
        """REQ-INFER-014: SVD directions are approximately orthogonal."""
        key = jrandom.PRNGKey(3)
        correct, halluc, _ = _make_synthetic_activations(key, hidden_dim=16)
        config = HallucinationDirectionConfig(top_k=3)
        directions = find_hallucination_direction(correct, halluc, config)

        # Check pairwise orthogonality.
        for i in range(3):
            for j in range(i + 1, 3):
                dot = float(jnp.abs(jnp.dot(directions[i], directions[j])))
                assert dot < 0.1, f"Directions {i},{j} not orthogonal: dot={dot:.3f}"


# ---------------------------------------------------------------------------
# hallucination_energy
# ---------------------------------------------------------------------------


class TestHallucinationEnergy:
    """Tests for REQ-INFER-014: Hallucination energy computation."""

    def test_energy_is_scalar(self) -> None:
        """REQ-INFER-014: Energy output is a scalar."""
        direction = jnp.array([1.0, 0.0, 0.0])
        activation = jnp.array([0.5, 0.3, 0.1])
        e = hallucination_energy(activation, direction)
        assert e.shape == ()

    def test_positive_projection_gives_positive_energy(self) -> None:
        """REQ-INFER-014: Activation aligned with direction gives positive energy."""
        direction = jnp.array([1.0, 0.0, 0.0])
        activation = jnp.array([3.0, 0.0, 0.0])
        e = hallucination_energy(activation, direction)
        assert float(e) > 0.0

    def test_negative_projection_gives_negative_energy(self) -> None:
        """REQ-INFER-014: Activation opposite to direction gives negative energy."""
        direction = jnp.array([1.0, 0.0, 0.0])
        activation = jnp.array([-3.0, 0.0, 0.0])
        e = hallucination_energy(activation, direction)
        assert float(e) < 0.0

    def test_orthogonal_gives_zero_energy(self) -> None:
        """REQ-INFER-014: Activation orthogonal to direction gives zero energy."""
        direction = jnp.array([1.0, 0.0, 0.0])
        activation = jnp.array([0.0, 5.0, 0.0])
        e = hallucination_energy(activation, direction)
        np.testing.assert_allclose(float(e), 0.0, atol=1e-6)

    def test_hallucinated_higher_than_correct(self) -> None:
        """REQ-INFER-014: Hallucinated activations get higher energy than correct."""
        key = jrandom.PRNGKey(10)
        correct, halluc, _ = _make_synthetic_activations(
            key, n_correct=30, n_hallucinated=30, separation=5.0
        )
        direction = find_hallucination_direction(correct, halluc)

        correct_energies = [float(hallucination_energy(a, direction)) for a in correct]
        halluc_energies = [float(hallucination_energy(a, direction)) for a in halluc]

        mean_correct = np.mean(correct_energies)
        mean_halluc = np.mean(halluc_energies)

        assert mean_halluc > mean_correct, (
            f"Hallucinated energy ({mean_halluc:.4f}) should exceed "
            f"correct energy ({mean_correct:.4f})"
        )

    def test_multi_direction_energy_is_scalar(self) -> None:
        """REQ-INFER-014: Multi-direction energy is also a scalar."""
        directions = jnp.eye(3)[:2]  # 2 directions in 3D
        activation = jnp.array([1.0, 2.0, 3.0])
        e = hallucination_energy(activation, directions)
        assert e.shape == ()

    def test_multi_direction_energy_sum_of_squares(self) -> None:
        """REQ-INFER-014: Multi-direction energy = sum of squared projections."""
        directions = jnp.eye(3)[:2]  # first two standard basis vectors
        activation = jnp.array([1.0, 2.0, 3.0])
        e = hallucination_energy(activation, directions)
        # Projections: [1.0, 2.0], sum of squares: 1 + 4 = 5
        np.testing.assert_allclose(float(e), 5.0, atol=1e-5)

    def test_zero_direction_safe(self) -> None:
        """REQ-INFER-014: Zero direction doesn't cause division by zero."""
        direction = jnp.zeros(4)
        activation = jnp.ones(4)
        e = hallucination_energy(activation, direction)
        assert np.isfinite(float(e))


# ---------------------------------------------------------------------------
# HallucinationDirectionConstraint
# ---------------------------------------------------------------------------


class TestHallucinationDirectionConstraint:
    """Tests for REQ-INFER-014: Constraint wrapper for ComposedEnergy integration."""

    def test_name_property(self) -> None:
        """REQ-INFER-014: Constraint has the expected name."""
        direction = jnp.array([1.0, 0.0])
        c = HallucinationDirectionConstraint(direction, constraint_name="test_halluc")
        assert c.name == "test_halluc"

    def test_default_name(self) -> None:
        """REQ-INFER-014: Default constraint name is 'hallucination_direction'."""
        direction = jnp.array([1.0, 0.0])
        c = HallucinationDirectionConstraint(direction)
        assert c.name == "hallucination_direction"

    def test_satisfaction_threshold(self) -> None:
        """REQ-INFER-014: Custom threshold is respected."""
        direction = jnp.array([1.0, 0.0])
        c = HallucinationDirectionConstraint(direction, threshold=0.5)
        assert c.satisfaction_threshold == 0.5

    def test_energy_relu_positive_projection(self) -> None:
        """REQ-INFER-014: Positive projection gives positive energy."""
        direction = jnp.array([1.0, 0.0])
        c = HallucinationDirectionConstraint(direction)
        x = jnp.array([2.0, 0.0])
        e = c.energy(x)
        assert float(e) > 0.0

    def test_energy_relu_negative_projection_is_zero(self) -> None:
        """REQ-INFER-014: Negative projection (away from hallucination) gives zero energy."""
        direction = jnp.array([1.0, 0.0])
        c = HallucinationDirectionConstraint(direction)
        x = jnp.array([-2.0, 0.0])
        e = c.energy(x)
        np.testing.assert_allclose(float(e), 0.0, atol=1e-6)

    def test_is_satisfied_below_threshold(self) -> None:
        """REQ-INFER-014: Constraint is satisfied when energy < threshold."""
        direction = jnp.array([1.0, 0.0])
        c = HallucinationDirectionConstraint(direction, threshold=0.5)
        # Small positive projection, energy < 0.5.
        x = jnp.array([0.3, 0.0])
        assert c.is_satisfied(x)

    def test_is_not_satisfied_above_threshold(self) -> None:
        """REQ-INFER-014: Constraint is violated when energy > threshold."""
        direction = jnp.array([1.0, 0.0])
        c = HallucinationDirectionConstraint(direction, threshold=0.5)
        x = jnp.array([5.0, 0.0])
        assert not c.is_satisfied(x)

    def test_grad_energy_shape(self) -> None:
        """REQ-INFER-014: Gradient has same shape as input."""
        direction = jnp.array([1.0, 0.0, 0.0])
        c = HallucinationDirectionConstraint(direction)
        x = jnp.array([2.0, 1.0, 0.5])
        g = c.grad_energy(x)
        assert g.shape == x.shape

    def test_grad_energy_points_toward_hallucination(self) -> None:
        """REQ-INFER-014: Gradient points in the hallucination direction (to increase energy)."""
        direction = jnp.array([1.0, 0.0, 0.0])
        c = HallucinationDirectionConstraint(direction)
        # x with positive projection — gradient should align with direction.
        x = jnp.array([2.0, 0.0, 0.0])
        g = c.grad_energy(x)
        # Gradient of max(dot(x, d)/||d||, 0) w.r.t. x is d/||d|| when dot > 0.
        np.testing.assert_allclose(np.array(g), np.array([1.0, 0.0, 0.0]), atol=1e-5)

    def test_composable_with_composed_energy(self) -> None:
        """REQ-INFER-014: Constraint integrates with ComposedEnergy."""
        direction = jnp.array([1.0, 0.0, 0.0])
        c = HallucinationDirectionConstraint(direction, threshold=0.5)

        composed = ComposedEnergy(input_dim=3)
        composed.add_constraint(c, weight=2.0)

        # Hallucinating activation.
        x_halluc = jnp.array([5.0, 0.0, 0.0])
        result = composed.verify(x_halluc)
        assert not result.is_verified()
        assert "hallucination_direction" in result.failing_constraints()

        # Non-hallucinating activation.
        x_good = jnp.array([-1.0, 0.0, 0.0])
        result_good = composed.verify(x_good)
        assert result_good.is_verified()

    def test_energy_batch_via_vmap(self) -> None:
        """REQ-INFER-014: Constraint energy works with jax.vmap for batching."""
        direction = jnp.array([1.0, 0.0])
        c = HallucinationDirectionConstraint(direction)
        xs = jnp.array([[2.0, 0.0], [-1.0, 0.0], [0.5, 0.0]])
        energies = jax.vmap(c.energy)(xs)
        assert energies.shape == (3,)
        # First should be 2.0, second should be 0.0 (ReLU), third 0.5.
        np.testing.assert_allclose(
            np.array(energies), np.array([2.0, 0.0, 0.5]), atol=1e-5
        )


# ---------------------------------------------------------------------------
# Package-level exports
# ---------------------------------------------------------------------------


class TestPackageExports:
    """Tests for package-level imports of hallucination direction symbols."""

    def test_hallucination_exports_from_embeddings(self) -> None:
        """REQ-INFER-014: Hallucination direction symbols accessible from carnot.embeddings."""
        from carnot.embeddings import (
            HallucinationDirectionConfig as PkgConfig,
            HallucinationDirectionConstraint as PkgConstraint,
            find_hallucination_direction as pkg_find,
            hallucination_energy as pkg_energy,
        )

        assert PkgConfig is HallucinationDirectionConfig
        assert PkgConstraint is HallucinationDirectionConstraint
        assert pkg_find is find_hallucination_direction
        assert pkg_energy is hallucination_energy
