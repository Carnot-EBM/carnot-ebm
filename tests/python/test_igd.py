"""Tests for Interleaved Gibbs Diffusion (IGD) (REQ-IGD-001, REQ-IGD-002, REQ-IGD-003, SCENARIO-IGD-001)."""

import numpy as np

from carnot.models.igd import IGDSmokeTest


def test_igd_smoke_test_initialization():
    """Test REQ-IGD-001 initialization."""
    np.random.seed(42)
    model = IGDSmokeTest(num_variables=5, num_clauses=3)
    assert model.num_variables == 5
    assert model.num_clauses == 3
    assert len(model.clauses) == 3


def test_igd_denoising():
    """Test REQ-IGD-002 and SCENARIO-IGD-001 denoising process."""
    np.random.seed(42)
    model = IGDSmokeTest(num_variables=10, num_clauses=5)
    result = model.run_denoising(num_steps=5)
    assert "satisfied_clauses" in result
    assert "total_clauses" in result
    assert "success" in result
    assert result["total_clauses"] == 5


def test_igd_execution_on_cpu():
    """Test REQ-IGD-003 execution on CPU implicitly via numpy."""
    np.random.seed(42)
    model = IGDSmokeTest(num_variables=3, num_clauses=2)
    result = model.run_denoising(num_steps=2)
    assert isinstance(result["final_state"], list)


def test_igd_check_clause():
    """Test check_clause to improve coverage."""
    model = IGDSmokeTest(num_variables=3, num_clauses=0)
    state = np.array([1, -1, 1])
    assert model.check_clause([1, 2, 3], state) == True  # var 1 is positive
    assert model.check_clause([-1, 2, -3], state) == False # None match
    assert model.check_clause([-1, -2, -3], state) == True # var 2 is negative
