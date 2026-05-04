"""Tests for Exp 1275 FSNet-style feasibility repair on ContinuousEBM latents.

Spec refs: REQ-KONA-027, SCENARIO-KONA-027.
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.phase3.continuous_ebm import ContinuousEBM, feasibility_step


def _task_energy(model: ContinuousEBM, state: np.ndarray) -> float:
    return float(-0.5 * state @ model.coupling @ state - model.bias @ state)


class TestFeasibilityStep:
    """REQ-KONA-027: feasibility_step repairs latent violation energy."""

    def test_reduces_violation_energy_without_task_gradient(self) -> None:
        """REQ-KONA-027: violation energy falls even when task energy prefers violation."""
        state = np.array([0.8, 0.65, -0.25], dtype=np.float64)
        constraints = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        bias = np.array([-0.2, -0.2], dtype=np.float64)
        model = ContinuousEBM(
            variables=3,
            coupling=np.zeros((3, 3), dtype=np.float64),
            bias=np.array([1.0, 1.0, 0.0], dtype=np.float64),
        )

        result = feasibility_step(
            state,
            constraints,
            bias,
            n_steps=48,
            lr=0.65,
            anchor_weight=0.0,
            tolerance=1e-8,
        )

        assert result.violation_energy < result.initial_violation_energy
        assert result.violation_count == 0
        assert result.convergence_steps > 0
        assert result.converged is True
        assert _task_energy(model, result.state) > _task_energy(model, state)

    def test_preserves_shape_and_tanh_bounds(self) -> None:
        """SCENARIO-KONA-027: repaired latents keep shape and bounded range."""
        state = np.array([0.95, -0.95, 0.55, -0.4], dtype=np.float64)
        constraints = np.array([[1.0, 0.0, 1.0, 0.0]], dtype=np.float64)
        bias = np.array([-0.1], dtype=np.float64)

        result = feasibility_step(state, constraints, bias, n_steps=16, lr=0.4)

        assert result.state.shape == state.shape
        assert np.all(result.state > -1.0)
        assert np.all(result.state < 1.0)
        assert result.distortion_l2 >= 0.0

    def test_already_feasible_state_has_zero_distortion(self) -> None:
        """REQ-KONA-027: feasible inputs are not changed by the repair step."""
        state = np.array([-0.4, 0.1, 0.2], dtype=np.float64)
        constraints = np.eye(3, dtype=np.float64)
        bias = np.array([0.0, -0.2, -0.3], dtype=np.float64)

        result = feasibility_step(state, constraints, bias)

        np.testing.assert_allclose(result.state, state)
        assert result.initial_violation_energy == pytest.approx(0.0)
        assert result.violation_energy == pytest.approx(0.0)
        assert result.violation_count == 0
        assert result.convergence_steps == 0
        assert result.distortion_l2 == pytest.approx(0.0)
        assert result.converged is True

    def test_default_zero_bias_constraints(self) -> None:
        """REQ-KONA-027: omitted constraint_bias means A @ z <= 0."""
        state = np.array([0.7, -0.2], dtype=np.float64)
        constraints = np.array([[1.0, 0.0]], dtype=np.float64)

        result = feasibility_step(
            state,
            constraints,
            n_steps=24,
            lr=0.7,
            anchor_weight=0.0,
        )

        assert result.initial_violation_energy > 0.0
        assert result.violation_energy == pytest.approx(0.0, abs=1e-8)
        assert result.violation_count == 0

    def test_invalid_shapes_raise(self) -> None:
        """REQ-KONA-027: malformed latent or constraint shapes are rejected."""
        with pytest.raises(ValueError, match="state must be one-dimensional"):
            feasibility_step(np.zeros((1, 2)), np.eye(2))

        with pytest.raises(ValueError, match="constraint_matrix"):
            feasibility_step(np.zeros(2), np.ones(2))

        with pytest.raises(ValueError, match="constraint_bias"):
            feasibility_step(np.zeros(2), np.eye(2), np.zeros(3))

    def test_invalid_hyperparameters_raise(self) -> None:
        """REQ-KONA-027: repair hyperparameters reject impossible values."""
        state = np.array([0.2, 0.3], dtype=np.float64)
        constraints = np.eye(2, dtype=np.float64)

        with pytest.raises(ValueError, match="n_steps"):
            feasibility_step(state, constraints, n_steps=-1)

        with pytest.raises(ValueError, match="lr"):
            feasibility_step(state, constraints, lr=-0.1)

        with pytest.raises(ValueError, match="anchor_weight"):
            feasibility_step(state, constraints, anchor_weight=-0.1)

        with pytest.raises(ValueError, match="tolerance"):
            feasibility_step(state, constraints, tolerance=-1e-6)
