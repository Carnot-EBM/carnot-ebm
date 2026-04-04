"""Tests for LLM-EBM verify-and-repair pipeline.

Spec coverage: REQ-INFER-003, REQ-INFER-004, REQ-INFER-005,
SCENARIO-INFER-005
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from carnot.inference.benchmark import (
    BenchmarkSummary,
    _count_violations,
    _summarize,
    generate_random_assignment,
    generate_random_graph,
    generate_random_sat,
    run_coloring_benchmark,
    run_sat_benchmark,
)
from carnot.inference.verify_and_repair import (
    VerifyRepairResult,
    parse_llm_coloring,
    parse_llm_sat_assignment,
    verify_and_repair,
)
from carnot.verify.sat import SATClause, build_sat_energy

# ---------------------------------------------------------------------------
# Tests: parse_llm_sat_assignment
# ---------------------------------------------------------------------------


class TestParseLLMSATAssignment:
    """Tests for SAT output parsing."""

    def test_parse_space_separated_01(self) -> None:
        """REQ-INFER-003: parse '1 0 1 0' format."""
        result = parse_llm_sat_assignment("1 0 1 0", 4)
        assert result.shape == (4,)
        assert float(result[0]) == 1.0
        assert float(result[1]) == 0.0

    def test_parse_true_false(self) -> None:
        """REQ-INFER-003: parse 'True False True' format."""
        result = parse_llm_sat_assignment("True False True", 3)
        assert float(result[0]) == 1.0
        assert float(result[1]) == 0.0

    def test_parse_t_f(self) -> None:
        """REQ-INFER-003: parse 'T F T' format."""
        result = parse_llm_sat_assignment("T F T", 3)
        assert float(result[0]) == 1.0
        assert float(result[1]) == 0.0

    def test_parse_named_format(self) -> None:
        """REQ-INFER-003: parse 'x1=True, x2=False' format."""
        result = parse_llm_sat_assignment("x1=True, x2=False, x3=1", 3)
        assert float(result[0]) == 1.0
        assert float(result[1]) == 0.0
        assert float(result[2]) == 1.0

    def test_parse_wrong_count_raises(self) -> None:
        """REQ-INFER-003: wrong variable count raises ValueError."""
        with pytest.raises(ValueError, match="Could not parse"):
            parse_llm_sat_assignment("1 0 1", 5)

    def test_parse_garbage_raises(self) -> None:
        """REQ-INFER-003: unparseable input raises ValueError."""
        with pytest.raises(ValueError, match="Could not parse"):
            parse_llm_sat_assignment("hello world", 2)

    def test_parse_comma_separated(self) -> None:
        """REQ-INFER-003: comma-separated values handled."""
        result = parse_llm_sat_assignment("1,0,1", 3)
        assert float(result[0]) == 1.0


# ---------------------------------------------------------------------------
# Tests: parse_llm_coloring
# ---------------------------------------------------------------------------


class TestParseLLMColoring:
    """Tests for coloring output parsing."""

    def test_parse_space_separated(self) -> None:
        """REQ-INFER-003: parse '0 1 2' format."""
        result = parse_llm_coloring("0 1 2", 3)
        assert result.shape == (3,)
        assert float(result[1]) == 1.0

    def test_parse_named_format(self) -> None:
        """REQ-INFER-003: parse 'node 0: 1, node 1: 2' format."""
        result = parse_llm_coloring("node 0: 1, node 1: 2, node 2: 0", 3)
        assert float(result[0]) == 1.0
        assert float(result[2]) == 0.0

    def test_parse_wrong_count_raises(self) -> None:
        """REQ-INFER-003: wrong node count raises ValueError."""
        with pytest.raises(ValueError, match="Could not parse"):
            parse_llm_coloring("0 1", 5)

    def test_parse_garbage_raises(self) -> None:
        """REQ-INFER-003: unparseable input raises ValueError."""
        with pytest.raises(ValueError, match="Could not parse"):
            parse_llm_coloring("red blue green", 3)


# ---------------------------------------------------------------------------
# Tests: verify_and_repair
# ---------------------------------------------------------------------------


class TestVerifyAndRepair:
    """Tests for the full pipeline."""

    def test_already_satisfied_no_repair(self) -> None:
        """SCENARIO-INFER-005: satisfied assignment needs no repair."""
        clauses = [SATClause([(0, False), (1, False)])]
        energy = build_sat_energy(clauses, n_vars=2)
        x = jnp.array([1.0, 1.0])

        result = verify_and_repair(x, energy)
        assert result.initial_verification is not None
        assert result.initial_verification.verdict.verified
        assert result.n_repair_steps == 0

    def test_violated_gets_repaired(self) -> None:
        """SCENARIO-INFER-005: violated assignment improved by repair."""
        clauses = [
            SATClause([(0, False), (1, False)]),
            SATClause([(0, True), (1, False)]),
        ]
        energy = build_sat_energy(clauses, n_vars=2)
        x = jnp.array([0.0, 0.0])  # Violates clause 0

        result = verify_and_repair(x, energy, max_repair_steps=30)
        assert result.initial_verification is not None
        assert result.repaired_verification is not None
        assert result.repaired_verification.total_energy <= result.initial_verification.total_energy

    def test_result_contains_trajectory(self) -> None:
        """SCENARIO-INFER-005: result includes repair trajectory."""
        clauses = [SATClause([(0, False)])]
        energy = build_sat_energy(clauses, n_vars=1)
        x = jnp.array([0.0])

        result = verify_and_repair(x, energy, max_repair_steps=10)
        assert len(result.repair_trajectory) >= 1

    def test_rounding_applied(self) -> None:
        """SCENARIO-INFER-005: round_fn applied to repaired result."""
        clauses = [SATClause([(0, False)])]
        energy = build_sat_energy(clauses, n_vars=1)
        x = jnp.array([0.7])

        def round_fn(arr: jnp.ndarray) -> jnp.ndarray:
            return jnp.where(arr >= 0.5, 1.0, 0.0)

        result = verify_and_repair(x, energy, round_fn=round_fn)
        assert result.rounded_assignment is not None
        assert float(result.rounded_assignment[0]) == 1.0

    def test_no_round_fn(self) -> None:
        """REQ-INFER-004: without round_fn, rounded = repaired."""
        clauses = [SATClause([(0, False)])]
        energy = build_sat_energy(clauses, n_vars=1)
        x = jnp.array([1.0])

        result = verify_and_repair(x, energy)
        assert result.rounded_verification is not None

    def test_result_dataclass(self) -> None:
        """REQ-INFER-004: VerifyRepairResult has all fields."""
        result = VerifyRepairResult()
        assert result.initial_assignment is None
        assert result.n_repair_steps == 0
        assert result.repair_trajectory == []


# ---------------------------------------------------------------------------
# Tests: benchmark harness
# ---------------------------------------------------------------------------


class TestBenchmarkHarness:
    """Tests for benchmark utilities."""

    def test_generate_random_sat(self) -> None:
        """REQ-INFER-005: generates correct number of clauses."""
        clauses = generate_random_sat(n_vars=5, n_clauses=10, clause_size=3)
        assert len(clauses) == 10
        for clause in clauses:
            assert len(clause.literals) == 3

    def test_generate_random_sat_deterministic(self) -> None:
        """REQ-INFER-005: same seed produces same result."""
        c1 = generate_random_sat(5, 10, seed=42)
        c2 = generate_random_sat(5, 10, seed=42)
        assert c1[0].literals == c2[0].literals

    def test_generate_random_graph(self) -> None:
        """REQ-INFER-005: generates edges between valid nodes."""
        edges = generate_random_graph(n_nodes=5, edge_probability=0.5, seed=42)
        for a, b in edges:
            assert 0 <= a < 5
            assert 0 <= b < 5
            assert a < b

    def test_generate_random_assignment(self) -> None:
        """REQ-INFER-005: generates correct length."""
        assignment = generate_random_assignment(10)
        assert len(assignment) == 10
        assert all(isinstance(v, bool) for v in assignment)

    def test_run_sat_benchmark(self) -> None:
        """REQ-INFER-005: SAT benchmark runs and produces summary."""
        summary = run_sat_benchmark(
            n_instances=2,
            n_vars=5,
            n_clauses=10,
            max_repair_steps=5,
        )
        assert isinstance(summary, BenchmarkSummary)
        assert summary.n_instances == 2
        assert len(summary.results) == 2

    def test_run_coloring_benchmark(self) -> None:
        """REQ-INFER-005: coloring benchmark runs and produces summary."""
        summary = run_coloring_benchmark(
            n_instances=2,
            n_nodes=4,
            n_colors=3,
            max_repair_steps=5,
        )
        assert isinstance(summary, BenchmarkSummary)
        assert summary.n_instances == 2
        assert len(summary.results) == 2

    def test_count_violations_none(self) -> None:
        """REQ-INFER-005: _count_violations handles None."""
        assert _count_violations(None) == 0

    def test_count_violations_no_verdict(self) -> None:
        """REQ-INFER-005: _count_violations handles missing verdict."""
        assert _count_violations(object()) == 0

    def test_summarize_empty(self) -> None:
        """REQ-INFER-005: _summarize handles empty results."""
        summary = _summarize([], 0)
        assert summary.n_instances == 0
        assert summary.repair_success_rate == 0.0
