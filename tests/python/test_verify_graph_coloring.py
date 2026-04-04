"""Tests for graph coloring constraint satisfaction.

Spec coverage: REQ-INFER-002, SCENARIO-INFER-003, SCENARIO-INFER-004
"""

from __future__ import annotations

import jax.numpy as jnp
from carnot.verify.constraint import repair
from carnot.verify.graph_coloring import (
    ColorDifferenceConstraint,
    ColorRangeConstraint,
    array_to_coloring,
    build_coloring_energy,
    coloring_to_array,
)

# ---------------------------------------------------------------------------
# Tests: ColorDifferenceConstraint
# ---------------------------------------------------------------------------


class TestColorDifferenceConstraint:
    """Tests for edge constraint."""

    def test_different_colors_zero_energy(self) -> None:
        """SCENARIO-INFER-003: different colors -> zero energy."""
        constraint = ColorDifferenceConstraint("edge_0_1", 0, 1)
        x = jnp.array([0.0, 1.0, 2.0])
        assert float(constraint.energy(x)) < 0.01

    def test_same_color_positive_energy(self) -> None:
        """SCENARIO-INFER-003: same color -> positive energy."""
        constraint = ColorDifferenceConstraint("edge_0_1", 0, 1)
        x = jnp.array([1.0, 1.0, 2.0])
        assert float(constraint.energy(x)) > 0.5

    def test_close_colors_penalty(self) -> None:
        """REQ-INFER-002: colors closer than 1 get penalty."""
        constraint = ColorDifferenceConstraint("edge_0_1", 0, 1)
        x = jnp.array([0.0, 0.5])
        energy = float(constraint.energy(x))
        assert energy > 0.0
        assert energy < 1.0  # Not maximum

    def test_name_and_threshold(self) -> None:
        """REQ-INFER-002: properties work."""
        constraint = ColorDifferenceConstraint("e01", 0, 1)
        assert constraint.name == "e01"
        assert constraint.satisfaction_threshold == 0.01


# ---------------------------------------------------------------------------
# Tests: ColorRangeConstraint
# ---------------------------------------------------------------------------


class TestColorRangeConstraint:
    """Tests for range penalty."""

    def test_in_range_zero_energy(self) -> None:
        """REQ-INFER-002: colors in [0, n_colors-1] have zero penalty."""
        constraint = ColorRangeConstraint("range", [0, 1, 2], n_colors=3)
        x = jnp.array([0.0, 1.0, 2.0])
        assert float(constraint.energy(x)) < 0.01

    def test_below_range_penalty(self) -> None:
        """REQ-INFER-002: negative color values penalized."""
        constraint = ColorRangeConstraint("range", [0], n_colors=3)
        x = jnp.array([-1.0])
        assert float(constraint.energy(x)) > 0.5

    def test_above_range_penalty(self) -> None:
        """REQ-INFER-002: above max color penalized."""
        constraint = ColorRangeConstraint("range", [0], n_colors=3)
        x = jnp.array([5.0])  # max_color = 2
        assert float(constraint.energy(x)) > 0.5

    def test_name_and_threshold(self) -> None:
        """REQ-INFER-002: properties work."""
        constraint = ColorRangeConstraint("r", [0], 3)
        assert constraint.name == "r"
        assert constraint.satisfaction_threshold == 0.01


# ---------------------------------------------------------------------------
# Tests: build_coloring_energy
# ---------------------------------------------------------------------------


class TestBuildColoringEnergy:
    """Tests for energy builder."""

    def test_valid_coloring_verified(self) -> None:
        """SCENARIO-INFER-003: valid coloring passes verification."""
        # Triangle: 3 nodes, 3 edges, 3 colors
        edges = [(0, 1), (1, 2), (0, 2)]
        energy = build_coloring_energy(edges, n_nodes=3, n_colors=3)
        x = jnp.array([0.0, 1.0, 2.0])
        result = energy.verify(x)
        assert result.verdict.verified

    def test_invalid_coloring_not_verified(self) -> None:
        """SCENARIO-INFER-003: invalid coloring fails."""
        edges = [(0, 1), (1, 2), (0, 2)]
        energy = build_coloring_energy(edges, n_nodes=3, n_colors=3)
        x = jnp.array([0.0, 0.0, 0.0])  # All same color
        result = energy.verify(x)
        assert not result.verdict.verified

    def test_constraint_count(self) -> None:
        """REQ-INFER-002: correct number of constraints."""
        edges = [(0, 1), (1, 2)]
        energy = build_coloring_energy(edges, n_nodes=3, n_colors=3)
        # 2 edge constraints + 1 range penalty = 3
        assert energy.num_constraints == 3

    def test_repair_reduces_coloring_energy(self) -> None:
        """SCENARIO-INFER-004: repair reduces coloring violations."""
        edges = [(0, 1), (1, 2), (0, 2)]
        energy = build_coloring_energy(edges, n_nodes=3, n_colors=3)
        x0 = jnp.array([0.0, 0.0, 0.0])  # All same color
        initial_e = float(energy.energy(x0))

        x_repaired, _history = repair(energy, x0, step_size=0.1, max_steps=20)
        repaired_e = float(energy.energy(x_repaired))

        assert repaired_e < initial_e


# ---------------------------------------------------------------------------
# Tests: conversion helpers
# ---------------------------------------------------------------------------


class TestColoringConversion:
    """Tests for coloring conversion."""

    def test_coloring_to_array(self) -> None:
        """REQ-INFER-002: int list to float array."""
        arr = coloring_to_array([0, 1, 2])
        assert arr.shape == (3,)
        assert float(arr[1]) == 1.0

    def test_array_to_coloring(self) -> None:
        """REQ-INFER-002: float array to int list."""
        result = array_to_coloring(jnp.array([0.1, 1.8, 2.3]))
        assert result == [0, 2, 2]

    def test_roundtrip(self) -> None:
        """REQ-INFER-002: coloring -> array -> coloring roundtrip."""
        original = [0, 1, 2, 1, 0]
        restored = array_to_coloring(coloring_to_array(original))
        assert restored == original
