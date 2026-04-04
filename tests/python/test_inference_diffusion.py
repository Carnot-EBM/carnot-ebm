"""Tests for energy-based diffusion generation.

Spec coverage: REQ-INFER-012, SCENARIO-INFER-013
"""

from __future__ import annotations

import jax.numpy as jnp

from carnot.inference.diffusion import (
    DiffusionConfig,
    DiffusionResult,
    _schedule_value,
    diffusion_generate,
    diffusion_generate_coloring,
    diffusion_generate_sat,
)
from carnot.verify.sat import SATClause, build_sat_energy


class TestScheduleValue:
    """Tests for step/noise scheduling."""

    def test_constant(self) -> None:
        """REQ-INFER-012: constant schedule returns max_val."""
        assert _schedule_value(5, 10, 1.0, "constant") == 1.0

    def test_linear_start(self) -> None:
        """REQ-INFER-012: linear starts at max_val."""
        val = _schedule_value(0, 10, 1.0, "linear")
        assert abs(val - 1.0) < 0.01

    def test_linear_end(self) -> None:
        """REQ-INFER-012: linear ends near 0."""
        val = _schedule_value(9, 10, 1.0, "linear")
        assert val < 0.15

    def test_cosine(self) -> None:
        """REQ-INFER-012: cosine schedule is bounded."""
        val = _schedule_value(5, 10, 1.0, "cosine")
        assert 0.0 <= val <= 1.0

    def test_single_step(self) -> None:
        """REQ-INFER-012: handles total=1."""
        assert _schedule_value(0, 1, 1.0, "linear") == 1.0


class TestDiffusionGenerate:
    """Tests for the main diffusion generator."""

    def test_returns_result(self) -> None:
        """REQ-INFER-012: returns DiffusionResult."""
        clauses = [SATClause([(0, False), (1, False)])]
        energy = build_sat_energy(clauses, n_vars=2)
        config = DiffusionConfig(n_diffusion_steps=10, seed=42)
        result = diffusion_generate(energy, config)
        assert isinstance(result, DiffusionResult)
        assert result.final_state is not None
        assert jnp.isfinite(result.final_energy)

    def test_energy_trajectory(self) -> None:
        """REQ-INFER-012: trajectory records energy at each step."""
        clauses = [SATClause([(0, False)])]
        energy = build_sat_energy(clauses, n_vars=1)
        config = DiffusionConfig(n_diffusion_steps=10)
        result = diffusion_generate(energy, config)
        assert len(result.trajectory_energies) == 11  # steps + final

    def test_multiple_candidates(self) -> None:
        """REQ-INFER-012: n_candidates > 1 selects best."""
        clauses = [SATClause([(0, False), (1, False)])]
        energy = build_sat_energy(clauses, n_vars=2)
        config = DiffusionConfig(n_diffusion_steps=20, n_candidates=3)
        result = diffusion_generate(energy, config)
        assert result.candidates_generated == 3

    def test_with_rounding(self) -> None:
        """REQ-INFER-012: round_fn applied to final state."""
        clauses = [SATClause([(0, False)])]
        energy = build_sat_energy(clauses, n_vars=1)
        config = DiffusionConfig(n_diffusion_steps=10)

        def round_fn(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.where(x >= 0.5, 1.0, 0.0)

        result = diffusion_generate(energy, config, round_fn=round_fn)
        val = float(result.final_state[0])
        assert val == 0.0 or val == 1.0

    def test_verification_populated(self) -> None:
        """REQ-INFER-012: verification is computed."""
        clauses = [SATClause([(0, False)])]
        energy = build_sat_energy(clauses, n_vars=1)
        config = DiffusionConfig(n_diffusion_steps=5)
        result = diffusion_generate(energy, config)
        assert result.verification is not None

    def test_default_config(self) -> None:
        """REQ-INFER-012: works with default config."""
        clauses = [SATClause([(0, False)])]
        energy = build_sat_energy(clauses, n_vars=1)
        result = diffusion_generate(energy)
        assert result.n_steps == 100


class TestDiffusionConvenience:
    """Tests for SAT and coloring convenience functions."""

    def test_sat(self) -> None:
        """SCENARIO-INFER-013: diffusion generates SAT assignment."""
        clauses = [
            SATClause([(0, False), (1, False)]),
            SATClause([(0, True), (1, False)]),
        ]
        config = DiffusionConfig(n_diffusion_steps=30, n_candidates=2)
        result = diffusion_generate_sat(clauses, n_vars=2, config=config)
        assert result.final_state is not None
        assert result.verification is not None

    def test_coloring(self) -> None:
        """SCENARIO-INFER-013: diffusion generates coloring."""
        edges = [(0, 1), (1, 2)]
        config = DiffusionConfig(n_diffusion_steps=30, n_candidates=2)
        result = diffusion_generate_coloring(edges, n_nodes=3, n_colors=3, config=config)
        assert result.final_state is not None


class TestDiffusionDataclasses:
    """Tests for dataclass defaults."""

    def test_config_defaults(self) -> None:
        """REQ-INFER-012: DiffusionConfig defaults."""
        c = DiffusionConfig()
        assert c.n_diffusion_steps == 100
        assert c.n_candidates == 1

    def test_result_defaults(self) -> None:
        """REQ-INFER-012: DiffusionResult defaults."""
        r = DiffusionResult()
        assert r.final_state is None
        assert r.final_energy == 0.0
        assert r.trajectory_energies == []
