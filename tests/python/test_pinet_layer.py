"""Tests for the Pi-Net Douglas-Rachford projection layer.

Spec refs: REQ-KONA-039, SCENARIO-KONA-039.
"""

import jax.numpy as jnp
import pytest

from carnot.models.pinet_layer import (
    DouglasRachfordPiNetLayer,
    LinearConstraintSet,
    _as_matrix,
    _as_vector,
    _as_float,
    build_toy_projection_cases,
    evaluate_toy_projection_cases,
    build_experiment_1662_artifact,
    write_experiment_1662_artifact,
    _repo_root,
)


def test_linear_constraint_set_invalid_dim():
    with pytest.raises(ValueError, match="state_dim must be positive"):
        LinearConstraintSet.from_arrays(state_dim=0)


def test_linear_constraint_set_no_constraints():
    with pytest.raises(ValueError, match="at least one hard constraint is required"):
        LinearConstraintSet.from_arrays(state_dim=2)


def test_linear_constraint_set_invalid_shapes():
    with pytest.raises(ValueError, match="equality_matrix must have shape \\(n_constraints, 2\\)"):
        LinearConstraintSet.from_arrays(state_dim=2, equality_matrix=[[1.0, 2.0, 3.0]])

    with pytest.raises(ValueError, match="equality_target must have shape \\(1,\\)"):
        LinearConstraintSet.from_arrays(state_dim=2, equality_matrix=[[1.0, 2.0]], equality_target=[1.0, 2.0])


def test_douglas_rachford_layer_validation():
    constraints = LinearConstraintSet.from_arrays(
        state_dim=2,
        equality_matrix=[[1.0, 1.0]],
        equality_target=[1.0]
    )

    with pytest.raises(ValueError, match="max_steps must be non-negative"):
        DouglasRachfordPiNetLayer(constraints, max_steps=-1)

    with pytest.raises(ValueError, match="tolerance must be non-negative"):
        DouglasRachfordPiNetLayer(constraints, tolerance=-1.0)

    with pytest.raises(ValueError, match="relaxation must be in \\(0, 2\\]"):
        DouglasRachfordPiNetLayer(constraints, relaxation=0.0)

    with pytest.raises(ValueError, match="relaxation must be in \\(0, 2\\]"):
        DouglasRachfordPiNetLayer(constraints, relaxation=2.5)

    layer = DouglasRachfordPiNetLayer(constraints)
    with pytest.raises(ValueError, match="state must have shape \\(2,\\)"):
        layer._validate_state([1.0, 2.0, 3.0])


def test_projection_satisfies_convex_inequalities():
    """Verify projection satisfies convex inequalities."""
    constraints = LinearConstraintSet.from_arrays(
        state_dim=2,
        inequality_matrix=[[-1.0, 0.0], [0.0, -1.0]],
        inequality_bound=[0.0, 0.0]
    )
    layer = DouglasRachfordPiNetLayer(constraints)
    
    # Test point that violates both constraints
    result = layer.project([-1.0, -2.0])
    
    assert result.converged
    assert result.projection_error <= layer.tolerance
    assert result.projected_state[0] >= -layer.tolerance
    assert result.projected_state[1] >= -layer.tolerance


def test_evaluate_toy_projection_cases():
    """Verify SCENARIO-KONA-039."""
    summary = evaluate_toy_projection_cases()
    assert summary["cases_evaluated"] == 3
    assert summary["differentiable_projection"] is True
    assert summary["convergence_steps"] > 0
    assert summary["projection_error"] < 1e-4

def test_build_experiment_1662_artifact():
    artifact = build_experiment_1662_artifact()
    assert artifact["status"] == "complete"
    assert artifact["schema"] == "carnot.models.pinet_layer.v1"
    assert artifact["honest_verdict"] == "pinet_layer_projection_complete"

def test_write_experiment_1662_artifact(tmp_path):
    out_path = tmp_path / "art.json"
    artifact = write_experiment_1662_artifact(output_path=out_path)
    assert out_path.exists()
    assert artifact["status"] == "complete"

def test_as_vector_none():
    vec = _as_vector(None, 2, "test")
    assert vec.shape == (2,)
    assert jnp.all(vec == 0.0)

def test_as_matrix_none():
    mat = _as_matrix(None, 2, "test")
    assert mat.shape == (0, 2)

def test_as_float():
    val = _as_float(jnp.array(3.14))
    assert isinstance(val, float)
    assert abs(val - 3.14) < 1e-6

def test_repo_root():
    root = _repo_root()
    assert root.name == "carnot"

def test_project_vector():
    constraints = LinearConstraintSet.from_arrays(
        state_dim=2,
        equality_matrix=[[1.0, 1.0]],
        equality_target=[1.0]
    )
    layer = DouglasRachfordPiNetLayer(constraints)
    vec = layer.project_vector([0.0, 0.0])
    assert vec.shape == (2,)
    assert abs(vec[0] + vec[1] - 1.0) < 1e-4

def test_projection_result_to_json():
    cases = build_toy_projection_cases()
    layer = DouglasRachfordPiNetLayer(cases[0].constraints)
    result = layer.project(cases[0].start)
    json_data = result.to_json()
    assert json_data["case_name"] == cases[0].constraints.name
    assert isinstance(json_data["projected_state"], list)
    assert isinstance(json_data["converged"], bool)

def test_project_equalities_only():
    constraints = LinearConstraintSet.from_arrays(
        state_dim=2,
        equality_matrix=[[1.0, -1.0]],
        equality_target=[0.0]
    )
    layer = DouglasRachfordPiNetLayer(constraints)
    result = layer.project([1.0, 0.0])
    assert result.converged
    assert abs(result.projected_state[0] - result.projected_state[1]) < 1e-4

def test_project_zero_error():
    constraints = LinearConstraintSet.from_arrays(
        state_dim=2,
        equality_matrix=[[1.0, 1.0]],
        equality_target=[1.0]
    )
    layer = DouglasRachfordPiNetLayer(constraints)
    result = layer.project([0.5, 0.5])
    assert result.convergence_steps == 0
    assert result.converged

def test_inequality_matrix_zero_rows():
    constraints = LinearConstraintSet.from_arrays(
        state_dim=2,
        equality_matrix=[[1.0, 1.0]],
        equality_target=[1.0]
    )
    assert constraints.inequality_matrix.shape[0] == 0
    error = constraints.projection_error([0.5, 0.5])
    assert _as_float(error) < 1e-6
