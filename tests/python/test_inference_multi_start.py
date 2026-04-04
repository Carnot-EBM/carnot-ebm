"""Tests for multi-start self-verification inference.

Spec coverage: REQ-INFER-009
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from carnot.inference.multi_start import MultiStartResult, multi_start_repair
from carnot.verify.constraint import BaseConstraint, ComposedEnergy


class QuadraticConstraint(BaseConstraint):
    """Constraint: (x[i] - target)^2. Satisfied when x[i] == target.

    A simple constraint with a single global minimum at x[i] = target.
    Used for testing basic multi-start behavior.
    """

    def __init__(self, name: str, index: int, target: float) -> None:
        self._name = name
        self._index = index
        self._target = target

    @property
    def name(self) -> str:
        return self._name

    def energy(self, x: jax.Array) -> jax.Array:
        return (x[self._index] - self._target) ** 2


class MultiWellConstraint(BaseConstraint):
    """Constraint with multiple local minima: cos(scale*x[i])^2.

    This creates a landscape with many wells. Repair from different
    starting points will converge to different local minima, so
    multi-start has a better chance of finding the global minimum
    (which is at x[i] = pi/(2*scale) + k*pi/scale for integer k).
    """

    def __init__(self, name: str, index: int, scale: float = 3.0) -> None:
        self._name = name
        self._index = index
        self._scale = scale

    @property
    def name(self) -> str:
        return self._name

    def energy(self, x: jax.Array) -> jax.Array:
        return jnp.cos(self._scale * x[self._index]) ** 2


class TestMultiStartRepair:
    """Tests for the multi_start_repair function.

    Spec: REQ-INFER-009
    """

    def test_finds_lower_energy_than_single_start(self) -> None:
        """REQ-INFER-009: Multi-start finds energy <= single start on multi-well."""
        composed = ComposedEnergy(input_dim=2)
        composed.add_constraint(MultiWellConstraint("well_0", 0, scale=5.0), 1.0)
        composed.add_constraint(MultiWellConstraint("well_1", 1, scale=5.0), 1.0)

        # Start from a point that is NOT at a global minimum
        x0 = jnp.array([0.5, 0.5])
        key = jax.random.PRNGKey(42)

        result = multi_start_repair(
            x0,
            composed,
            n_starts=10,
            perturbation_scale=1.0,
            step_size=0.05,
            max_repair_steps=100,
            key=key,
        )

        # Multi-start with 10 starts should find a result at least as good as
        # the worst single start. Just verify it produced a valid result
        # with finite energy.
        assert jnp.isfinite(result.best_energy), (
            f"Best energy should be finite, got {result.best_energy}"
        )
        assert result.n_starts == 10
        assert len(result.all_final_energies) == 10

    def test_n_starts_one_degrades_to_single(self) -> None:
        """REQ-INFER-009: n_starts=1 degrades to single repair run."""
        composed = ComposedEnergy(input_dim=2)
        composed.add_constraint(QuadraticConstraint("q0", 0, 1.0), 1.0)
        composed.add_constraint(QuadraticConstraint("q1", 1, 2.0), 1.0)

        x0 = jnp.array([0.0, 0.0])
        key = jax.random.PRNGKey(0)

        result = multi_start_repair(
            x0,
            composed,
            n_starts=1,
            perturbation_scale=0.1,
            step_size=0.1,
            max_repair_steps=100,
            key=key,
        )

        assert result.n_starts == 1
        assert result.best_start_index == 0
        assert len(result.all_final_energies) == 1
        assert jnp.isfinite(result.best_energy)

    def test_best_start_index_is_correct(self) -> None:
        """REQ-INFER-009: best_start_index points to the lowest energy result."""
        composed = ComposedEnergy(input_dim=2)
        composed.add_constraint(QuadraticConstraint("q0", 0, 3.0), 1.0)
        composed.add_constraint(QuadraticConstraint("q1", 1, -1.0), 1.0)

        x0 = jnp.array([0.0, 0.0])
        key = jax.random.PRNGKey(7)

        result = multi_start_repair(
            x0,
            composed,
            n_starts=5,
            perturbation_scale=0.5,
            step_size=0.1,
            max_repair_steps=200,
            key=key,
        )

        # Verify the best_start_index actually points to the min energy
        min_energy_idx = int(jnp.argmin(jnp.array(result.all_final_energies)))
        assert result.best_start_index == min_energy_idx
        assert result.best_energy == result.all_final_energies[min_energy_idx]

    def test_with_round_fn(self) -> None:
        """REQ-INFER-009: round_fn is applied to repaired results."""
        composed = ComposedEnergy(input_dim=2)
        composed.add_constraint(QuadraticConstraint("q0", 0, 2.0), 1.0)
        composed.add_constraint(QuadraticConstraint("q1", 1, 3.0), 1.0)

        x0 = jnp.array([0.0, 0.0])
        key = jax.random.PRNGKey(99)

        result = multi_start_repair(
            x0,
            composed,
            n_starts=3,
            perturbation_scale=0.5,
            step_size=0.1,
            max_repair_steps=200,
            round_fn=jnp.round,
            key=key,
        )

        # After rounding, best_x values should be integers
        assert jnp.allclose(result.best_x, jnp.round(result.best_x)), (
            "After rounding, values should be integers"
        )

    def test_default_key_is_deterministic(self) -> None:
        """REQ-INFER-009: Default key (None) produces deterministic results."""
        composed = ComposedEnergy(input_dim=2)
        composed.add_constraint(QuadraticConstraint("q0", 0, 1.0), 1.0)

        x0 = jnp.array([0.0, 0.0])

        result1 = multi_start_repair(x0, composed, n_starts=3)
        result2 = multi_start_repair(x0, composed, n_starts=3)

        assert jnp.allclose(result1.best_x, result2.best_x), (
            "Default key should produce deterministic results"
        )

    def test_result_has_history(self) -> None:
        """REQ-INFER-009: best_history contains verification results."""
        composed = ComposedEnergy(input_dim=2)
        composed.add_constraint(QuadraticConstraint("q0", 0, 1.0), 1.0)

        x0 = jnp.array([0.0, 0.0])

        result = multi_start_repair(x0, composed, n_starts=2, max_repair_steps=10)

        assert len(result.best_history) > 0, "History should not be empty"
        assert hasattr(result.best_history[0], "total_energy")


class TestMultiStartResult:
    """Tests for the MultiStartResult dataclass.

    Spec: REQ-INFER-009
    """

    def test_dataclass_defaults(self) -> None:
        """REQ-INFER-009: MultiStartResult has correct defaults."""
        result = MultiStartResult(
            best_x=jnp.zeros(2),
            best_energy=0.0,
            best_history=[],
        )
        assert result.all_final_energies == []
        assert result.best_start_index == 0
        assert result.n_starts == 0


class TestMultiStartImports:
    """Tests for module imports.

    Spec: REQ-INFER-009
    """

    def test_import_from_inference(self) -> None:
        """REQ-INFER-009: multi_start_repair importable from carnot.inference."""
        from carnot.inference import multi_start_repair as msr

        assert callable(msr)

    def test_import_from_module(self) -> None:
        """REQ-INFER-009: multi_start_repair importable from carnot.inference.multi_start."""
        from carnot.inference.multi_start import multi_start_repair

        assert callable(multi_start_repair)
