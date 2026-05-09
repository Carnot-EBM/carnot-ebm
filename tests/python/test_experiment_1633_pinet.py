"""Tests for Exp 1633 Pi-Net-style continuous projection.

Spec refs: REQ-KONA-037, SCENARIO-KONA-037.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts import experiment_1633_pinet as mod


def test_req_kona_037_projects_infeasible_simplex_state() -> None:
    """REQ-KONA-037: infeasible continuous states are hard-projected."""

    case = mod.build_toy_cases()[0]
    layer = mod.PiNetProjectionLayer(case.system, max_steps=64)

    result = layer.project(case.start)
    projected = np.asarray(result.projected_state, dtype=np.float32)

    assert result.initial_projection_error > result.projection_error
    assert result.projection_error <= mod.DEFAULT_TOLERANCE
    assert result.converged is True
    assert 0 < result.convergence_steps <= 64
    assert np.sum(projected) == pytest.approx(1.0, abs=mod.DEFAULT_TOLERANCE)
    assert np.all(projected >= -mod.DEFAULT_TOLERANCE)

    feasible = layer.project((0.6, 0.25, 0.15))
    assert feasible.convergence_steps == 0
    assert feasible.projection_error <= mod.DEFAULT_TOLERANCE


def test_req_kona_037_projection_is_jax_differentiable() -> None:
    """REQ-KONA-037: JAX autodiff sees a finite projected-state gradient."""

    case = mod.build_toy_cases()[1]
    layer = mod.PiNetProjectionLayer(case.system, max_steps=8)

    def loss(state: jax.Array) -> jax.Array:
        projected = layer.project_vector(state)
        return jnp.sum(projected * projected)

    grad = jax.grad(loss)(jnp.asarray(case.start, dtype=jnp.float32))

    assert grad.shape == (case.system.state_dim,)
    assert jnp.all(jnp.isfinite(grad))


def test_req_kona_037_rejects_malformed_constraints() -> None:
    """REQ-KONA-037: malformed linear constraint arrays fail before projection."""

    with pytest.raises(ValueError, match="equality_target"):
        mod.ContinuousConstraintSystem.from_arrays(
            state_dim=2,
            equality_matrix=[[1.0, 1.0]],
            equality_target=[1.0, 2.0],
            name="bad_eq_target",
        )

    with pytest.raises(ValueError, match="inequality_matrix"):
        mod.ContinuousConstraintSystem.from_arrays(
            state_dim=2,
            inequality_matrix=[[1.0, 0.0, 0.0]],
            inequality_bound=[0.0],
            name="bad_ineq_matrix",
        )

    system = mod.ContinuousConstraintSystem.from_arrays(
        state_dim=2,
        inequality_matrix=[[1.0, 0.0]],
        inequality_bound=[0.0],
        name="valid",
    )
    layer = mod.PiNetProjectionLayer(system)
    with pytest.raises(ValueError, match="state"):
        layer.project(jnp.zeros((1, 2), dtype=jnp.float32))


def test_scenario_kona_037_evaluates_all_toy_cases() -> None:
    """SCENARIO-KONA-037: toy projection summary reports convergence metrics."""

    summary = mod.evaluate_projection_cases(max_steps=96)

    assert summary["cases_evaluated"] == len(mod.build_toy_cases())
    assert summary["projection_error"] <= mod.DEFAULT_TOLERANCE
    assert summary["convergence_steps"] > 0
    assert summary["differentiable_projection"] is True
    assert all(row["projection_error"] <= mod.DEFAULT_TOLERANCE for row in summary["case_results"])


def test_req_kona_037_run_writes_required_artifact(tmp_path: Path) -> None:
    """REQ-KONA-037: run writes projection_error and convergence_steps to JSON."""

    output_path = tmp_path / "experiment_1633_pinet.json"
    artifact = mod.run_experiment(
        output_path=output_path,
        tests_run=[".venv/bin/pytest tests/python/test_experiment_1633_pinet.py -q"],
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["schema"] == mod.SCHEMA
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["spec_refs"] == mod.SPEC_REFS
    assert artifact["projection_error"] <= mod.DEFAULT_TOLERANCE
    assert artifact["convergence_steps"] >= 1
    assert artifact["honest_verdict"] == "pinet_projection_satisfies_hard_constraints"


def test_req_kona_037_artifact_validation_rejects_inconsistent_payload() -> None:
    """REQ-KONA-037: artifact validation enforces measured projection fields."""

    valid = mod.build_artifact(tests_run=())
    mod.validate_artifact(valid)

    missing = dict(valid)
    del missing["projection_error"]
    with pytest.raises(AssertionError, match="missing"):
        mod.validate_artifact(missing)

    with pytest.raises(AssertionError, match="projection_error"):
        mod.validate_artifact(dict(valid, projection_error=1.0, status="complete"))

    with pytest.raises(AssertionError, match="differentiable_projection"):
        mod.validate_artifact(dict(valid, differentiable_projection=False, status="complete"))
