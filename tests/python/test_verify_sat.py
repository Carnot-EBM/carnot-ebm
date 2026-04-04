"""Tests for SAT constraint satisfaction.

Spec coverage: REQ-INFER-001, SCENARIO-INFER-001, SCENARIO-INFER-002,
SCENARIO-INFER-004, SCENARIO-INFER-006
"""

from __future__ import annotations

import jax.numpy as jnp
from carnot.verify.constraint import repair
from carnot.verify.sat import (
    SATBinaryConstraint,
    SATClause,
    SATClauseConstraint,
    array_to_assignment,
    assignment_to_array,
    build_sat_energy,
    parse_dimacs,
)

# ---------------------------------------------------------------------------
# Tests: SATClauseConstraint
# ---------------------------------------------------------------------------


class TestSATClauseConstraint:
    """Tests for single clause energy."""

    def test_satisfied_clause_zero_energy(self) -> None:
        """SCENARIO-INFER-001: satisfied clause has energy ~0."""
        # Clause (x0 OR x1), assignment x0=True
        clause = SATClause([(0, False), (1, False)])
        constraint = SATClauseConstraint("c0", clause)
        x = jnp.array([1.0, 0.0])
        energy = float(constraint.energy(x))
        assert energy < 0.01

    def test_violated_clause_positive_energy(self) -> None:
        """SCENARIO-INFER-001: violated clause has energy > 0."""
        # Clause (x0 OR x1), assignment x0=False, x1=False
        clause = SATClause([(0, False), (1, False)])
        constraint = SATClauseConstraint("c0", clause)
        x = jnp.array([0.0, 0.0])
        energy = float(constraint.energy(x))
        assert energy > 0.5  # Product = (1-0)*(1-0) = 1.0

    def test_negated_literal_satisfied(self) -> None:
        """REQ-INFER-001: negated literal handles correctly."""
        # Clause (NOT x0), assignment x0=False -> NOT x0=True -> satisfied
        clause = SATClause([(0, True)])
        constraint = SATClauseConstraint("c0", clause)
        x = jnp.array([0.0])
        energy = float(constraint.energy(x))
        assert energy < 0.01

    def test_negated_literal_violated(self) -> None:
        """REQ-INFER-001: negated literal violated correctly."""
        # Clause (NOT x0), assignment x0=True -> NOT x0=False -> violated
        clause = SATClause([(0, True)])
        constraint = SATClauseConstraint("c0", clause)
        x = jnp.array([1.0])
        energy = float(constraint.energy(x))
        assert energy > 0.5

    def test_three_sat_clause(self) -> None:
        """REQ-INFER-001: 3-SAT clause with mixed literals."""
        # Clause (x0 OR NOT x1 OR x2)
        clause = SATClause([(0, False), (1, True), (2, False)])
        constraint = SATClauseConstraint("c0", clause)

        # x0=T -> satisfied
        assert float(constraint.energy(jnp.array([1.0, 1.0, 0.0]))) < 0.01

        # All false: x0=F, NOT x1=F (x1=T), x2=F -> violated
        assert float(constraint.energy(jnp.array([0.0, 1.0, 0.0]))) > 0.5

    def test_satisfaction_threshold(self) -> None:
        """REQ-INFER-001: threshold matches Sudoku convention."""
        clause = SATClause([(0, False)])
        constraint = SATClauseConstraint("c0", clause)
        assert constraint.satisfaction_threshold == 0.01

    def test_is_satisfied(self) -> None:
        """REQ-INFER-001: is_satisfied delegates to energy + threshold."""
        clause = SATClause([(0, False)])
        constraint = SATClauseConstraint("c0", clause)
        assert constraint.is_satisfied(jnp.array([1.0]))
        assert not constraint.is_satisfied(jnp.array([0.0]))

    def test_name_property(self) -> None:
        """REQ-INFER-001: name is accessible."""
        constraint = SATClauseConstraint("test_name", SATClause([]))
        assert constraint.name == "test_name"


# ---------------------------------------------------------------------------
# Tests: SATBinaryConstraint
# ---------------------------------------------------------------------------


class TestSATBinaryConstraint:
    """Tests for binary penalty."""

    def test_binary_values_zero_energy(self) -> None:
        """SCENARIO-INFER-006: binary values have zero penalty."""
        constraint = SATBinaryConstraint("bin", [0, 1, 2])
        x = jnp.array([0.0, 1.0, 0.0])
        assert float(constraint.energy(x)) < 0.01

    def test_midpoint_maximum_energy(self) -> None:
        """SCENARIO-INFER-006: x=0.5 has maximum penalty."""
        constraint = SATBinaryConstraint("bin", [0, 1])
        x_mid = jnp.array([0.5, 0.5])
        x_bin = jnp.array([0.0, 1.0])
        assert float(constraint.energy(x_mid)) > float(constraint.energy(x_bin))

    def test_name_and_threshold(self) -> None:
        """SCENARIO-INFER-006: properties are set."""
        constraint = SATBinaryConstraint("bin", [0])
        assert constraint.name == "bin"
        assert constraint.satisfaction_threshold == 0.01


# ---------------------------------------------------------------------------
# Tests: build_sat_energy
# ---------------------------------------------------------------------------


class TestBuildSATEnergy:
    """Tests for energy builder."""

    def test_satisfying_assignment_verified(self) -> None:
        """SCENARIO-INFER-001: known-satisfying assignment verifies."""
        # (x0 OR x1) AND (NOT x0 OR x1) — satisfied by x0=T, x1=T
        clauses = [
            SATClause([(0, False), (1, False)]),
            SATClause([(0, True), (1, False)]),
        ]
        energy = build_sat_energy(clauses, n_vars=2)
        x = jnp.array([1.0, 1.0])
        result = energy.verify(x)
        assert result.verdict.verified

    def test_violating_assignment_not_verified(self) -> None:
        """SCENARIO-INFER-001: violating assignment fails."""
        # (x0) AND (NOT x0) — unsatisfiable
        clauses = [
            SATClause([(0, False)]),
            SATClause([(0, True)]),
        ]
        energy = build_sat_energy(clauses, n_vars=1)
        x = jnp.array([0.0])
        result = energy.verify(x)
        assert not result.verdict.verified

    def test_constraint_count(self) -> None:
        """REQ-INFER-001: correct number of constraints."""
        clauses = [SATClause([(0, False)]) for _ in range(5)]
        energy = build_sat_energy(clauses, n_vars=3)
        # 5 clause constraints + 1 binary penalty = 6
        assert energy.num_constraints == 6


# ---------------------------------------------------------------------------
# Tests: parse_dimacs
# ---------------------------------------------------------------------------


class TestParseDIMACS:
    """Tests for DIMACS parser."""

    def test_basic_parsing(self) -> None:
        """SCENARIO-INFER-002: basic DIMACS format."""
        text = """\
c Example
p cnf 3 2
1 -2 3 0
-1 2 0
"""
        clauses, n_vars = parse_dimacs(text)
        assert n_vars == 3
        assert len(clauses) == 2
        assert clauses[0].literals == [(0, False), (1, True), (2, False)]
        assert clauses[1].literals == [(0, True), (1, False)]

    def test_comments_ignored(self) -> None:
        """SCENARIO-INFER-002: comment lines ignored."""
        text = """\
c This is a comment
c Another comment
p cnf 2 1
1 2 0
"""
        clauses, n_vars = parse_dimacs(text)
        assert n_vars == 2
        assert len(clauses) == 1

    def test_negated_variables(self) -> None:
        """SCENARIO-INFER-002: negative literals parsed as negated."""
        text = "p cnf 3 1\n-1 -2 -3 0\n"
        clauses, _ = parse_dimacs(text)
        assert all(neg for _, neg in clauses[0].literals)

    def test_empty_input(self) -> None:
        """SCENARIO-INFER-002: empty input returns empty."""
        clauses, n_vars = parse_dimacs("")
        assert n_vars == 0
        assert clauses == []

    def test_missing_terminator(self) -> None:
        """SCENARIO-INFER-002: clause without trailing 0 still parsed."""
        text = "p cnf 2 1\n1 2"
        clauses, n_vars = parse_dimacs(text)
        assert n_vars == 2
        assert len(clauses) == 1


# ---------------------------------------------------------------------------
# Tests: conversion helpers
# ---------------------------------------------------------------------------


class TestConversionHelpers:
    """Tests for assignment/array conversion."""

    def test_assignment_to_array(self) -> None:
        """REQ-INFER-001: bool list to float array."""
        arr = assignment_to_array([True, False, True])
        assert arr.shape == (3,)
        assert float(arr[0]) == 1.0
        assert float(arr[1]) == 0.0

    def test_array_to_assignment(self) -> None:
        """REQ-INFER-001: float array to bool list."""
        result = array_to_assignment(jnp.array([0.9, 0.1, 0.6]))
        assert result == [True, False, True]

    def test_roundtrip(self) -> None:
        """REQ-INFER-001: assignment -> array -> assignment roundtrip."""
        original = [True, False, True, False]
        restored = array_to_assignment(assignment_to_array(original))
        assert restored == original


# ---------------------------------------------------------------------------
# Tests: repair
# ---------------------------------------------------------------------------


class TestSATRepair:
    """Tests for gradient repair on SAT."""

    def test_repair_reduces_energy(self) -> None:
        """SCENARIO-INFER-004: repair reduces SAT energy."""
        # Small instance: (x0 OR x1) AND (NOT x0 OR x1) AND (x0 OR NOT x1)
        clauses = [
            SATClause([(0, False), (1, False)]),
            SATClause([(0, True), (1, False)]),
            SATClause([(0, False), (1, True)]),
        ]
        energy = build_sat_energy(clauses, n_vars=2)

        # Start from a bad assignment
        x0 = jnp.array([0.0, 0.0])
        initial_e = float(energy.energy(x0))

        # Repair
        x_repaired, _history = repair(energy, x0, step_size=0.1, max_steps=20)
        repaired_e = float(energy.energy(x_repaired))

        assert repaired_e <= initial_e
