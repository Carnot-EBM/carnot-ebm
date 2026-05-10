"""Tests for the reusable Pi-Net Douglas-Rachford projection layer.

Spec refs: REQ-KONA-039, SCENARIO-KONA-039.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from carnot.models.pinet_layer import (
    DEFAULT_TOLERANCE,
    REQUIRED_ARTIFACT_FIELDS,
    DouglasRachfordPiNetLayer,
    LinearConstraintSet,
    build_experiment_1662_artifact,
    build_toy_projection_cases,
    write_experiment_1662_artifact,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_req_kona_039_spec_anchor_exists() -> None:
    """REQ-KONA-039, SCENARIO-KONA-039: Pi-Net layer work is spec-anchored."""

    spec = (REPO_ROOT / "openspec/capabilities/phase3-kona/spec.md").read_text(encoding="utf-8")

    assert "REQ-KONA-039" in spec
    assert "SCENARIO-KONA-039" in spec
    assert "python/carnot/models/pinet_layer.py" in spec


def test_req_kona_039_projects_infeasible_simplex_state() -> None:
    """REQ-KONA-039: infeasible continuous states are hard-projected."""

    case = build_toy_projection_cases()[0]
    layer = DouglasRachfordPiNetLayer(case.constraints, max_steps=32)

    result = layer.project(case.start)
    projected = jnp.asarray(result.projected_state, dtype=jnp.float32)

    assert result.initial_projection_error > result.projection_error
    assert result.projection_error <= DEFAULT_TOLERANCE
    assert result.converged is True
    assert 0 < result.convergence_steps <= 32
    assert jnp.sum(projected) == pytest.approx(1.0, abs=DEFAULT_TOLERANCE)
    assert jnp.all(projected >= -DEFAULT_TOLERANCE)


def test_req_kona_039_leaves_feasible_state_stable() -> None:
    """REQ-KONA-039: already feasible states do not move beyond tolerance."""

    case = build_toy_projection_cases()[0]
    layer = DouglasRachfordPiNetLayer(case.constraints, max_steps=32)
    feasible = jnp.array([0.6, 0.25, 0.15], dtype=jnp.float32)

    result = layer.project(feasible)

    assert result.convergence_steps == 0
    assert result.projection_error <= DEFAULT_TOLERANCE
    assert result.projected_state == pytest.approx(tuple(feasible.tolist()), abs=1e-6)


def test_req_kona_039_projection_is_jax_differentiable() -> None:
    """REQ-KONA-039: JAX autodiff sees a finite projected-state gradient."""

    case = build_toy_projection_cases()[1]
    layer = DouglasRachfordPiNetLayer(case.constraints, max_steps=8)

    def loss(state: jax.Array) -> jax.Array:
        projected = layer.project_vector(state)
        return jnp.sum(projected * projected)

    grad = jax.grad(loss)(jnp.asarray(case.start, dtype=jnp.float32))

    assert grad.shape == (case.constraints.state_dim,)
    assert jnp.all(jnp.isfinite(grad))


def test_req_kona_039_rejects_malformed_constraints_and_state() -> None:
    """REQ-KONA-039: malformed arrays fail before projection."""

    with pytest.raises(ValueError, match="state_dim"):
        LinearConstraintSet.from_arrays(
            state_dim=0,
            inequality_matrix=[[1.0]],
            inequality_bound=[0.0],
        )

    with pytest.raises(ValueError, match="equality_target"):
        LinearConstraintSet.from_arrays(
            state_dim=2,
            equality_matrix=[[1.0, 1.0]],
            equality_target=[1.0, 2.0],
        )

    with pytest.raises(ValueError, match="inequality_matrix"):
        LinearConstraintSet.from_arrays(
            state_dim=2,
            inequality_matrix=[[1.0, 0.0, 0.0]],
            inequality_bound=[0.0],
        )

    constraints = LinearConstraintSet.from_arrays(
        state_dim=2,
        inequality_matrix=[[1.0, 0.0]],
        inequality_bound=[0.0],
    )
    layer = DouglasRachfordPiNetLayer(constraints)

    with pytest.raises(ValueError, match="state"):
        layer.project(jnp.zeros((1, 2), dtype=jnp.float32))

    with pytest.raises(ValueError, match="at least one hard constraint"):
        LinearConstraintSet.from_arrays(state_dim=2)

    with pytest.raises(ValueError, match="max_steps"):
        DouglasRachfordPiNetLayer(constraints, max_steps=-1)

    with pytest.raises(ValueError, match="tolerance"):
        DouglasRachfordPiNetLayer(constraints, tolerance=-1.0)

    with pytest.raises(ValueError, match="relaxation"):
        DouglasRachfordPiNetLayer(constraints, relaxation=0.0)


def test_scenario_kona_039_artifact_has_required_schema_fields(tmp_path: Path) -> None:
    """SCENARIO-KONA-039: Exp 1662 artifact records projection diagnostics."""

    artifact = build_experiment_1662_artifact(
        tests_run=[".venv/bin/pytest tests/python/test_models_pinet_layer.py -q"]
    )

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["schema"] == "carnot.models.pinet_layer.v1"
    assert artifact["experiment_id"] == 1662
    assert artifact["spec_refs"] == ["REQ-KONA-039", "SCENARIO-KONA-039"]
    assert artifact["module_path"] == "python/carnot/models/pinet_layer.py"
    assert artifact["projection_error"] <= DEFAULT_TOLERANCE
    assert artifact["differentiable_projection"] is True
    assert artifact["honest_verdict"] == "pinet_layer_projection_complete"

    output_path = tmp_path / "experiment_1662_pinet_layer.json"
    written = write_experiment_1662_artifact(output_path)

    assert written["status"] == "complete"
    assert json.loads(output_path.read_text(encoding="utf-8")) == written
