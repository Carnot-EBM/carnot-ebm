"""Tests for Exp 2005 adaptive KAEM/KAN energy landscape topology updates.

Spec refs: REQ-KAN-2005, SCENARIO-KAN-2005.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

from carnot.models.kaem_adaptive_topology import (
    REQUIRED_ARTIFACT_FIELDS,
    build_adaptive_energy_landscape_kan_artifact,
    validate_adaptive_energy_landscape_kan_artifact,
    write_adaptive_energy_landscape_kan_artifact,
)
from carnot.models.kaem_energy import UnivariateKAEMLayer


def _layer_with_control_points(values: list[float]) -> UnivariateKAEMLayer:
    layer = UnivariateKAEMLayer(n_vars=1, n_knots=len(values), key=jrandom.PRNGKey(0))
    layer.control_points = jnp.array([values], dtype=jnp.float32)
    layer._knots = jnp.linspace(-1.0, 1.0, len(values))
    layer.n_knots = len(values)
    return layer


def test_req_kan_2005_complex_landscape_adds_knot_and_stays_evaluable() -> None:
    """REQ-KAN-2005: complex spline regions gain a knot and remain finite."""

    layer = _layer_with_control_points([0.0, 0.05, 0.95, 0.98, 1.0])
    before = np.array(layer._knots)

    metrics = layer.adaptive_mesh_refine(
        high_complexity_threshold=1.0,
        low_complexity_threshold=-1.0,
        min_knots=3,
        max_knots=8,
        max_additions=1,
        max_removals=0,
    )

    after = np.array(layer._knots)
    assert metrics.knots_added == 1
    assert metrics.knots_removed == 0
    assert layer.n_knots == len(before) + 1
    assert after[0] == pytest.approx(-1.0)
    assert after[-1] == pytest.approx(1.0)
    assert metrics.added_positions[0] not in before.tolist()
    assert np.isfinite(float(layer.energy(jnp.array([-0.25], dtype=jnp.float32))))


def test_req_kan_2005_smooth_landscape_removes_interior_knot() -> None:
    """REQ-KAN-2005: smooth spline regions lose removable interior knots."""

    layer = _layer_with_control_points([0.0, 0.25, 0.5, 0.75, 1.0])

    metrics = layer.adaptive_mesh_refine(
        high_complexity_threshold=99.0,
        low_complexity_threshold=0.01,
        min_knots=3,
        max_knots=8,
        max_additions=0,
        max_removals=1,
    )

    assert metrics.knots_added == 0
    assert metrics.knots_removed == 1
    assert layer.n_knots == 4
    assert all(-1.0 < pos < 1.0 for pos in metrics.removed_positions)
    np.testing.assert_allclose(np.array(layer._knots)[[0, -1]], [-1.0, 1.0])
    assert np.isfinite(float(layer.energy(jnp.array([0.2], dtype=jnp.float32))))


def test_req_kan_2005_metrics_are_json_safe_and_capture_mixed_change() -> None:
    """REQ-KAN-2005: structural change metrics carry serializable topology evidence."""

    layer = _layer_with_control_points([0.0, 0.05, 0.95, 0.98, 1.0])

    metrics = layer.adaptive_mesh_refine(
        high_complexity_threshold=1.0,
        low_complexity_threshold=0.05,
        min_knots=3,
        max_knots=8,
        max_additions=1,
        max_removals=1,
    )
    payload = metrics.to_dict()

    assert metrics.changed is True
    assert payload["spec_traces"] == ["REQ-KAN-2005", "SCENARIO-KAN-2005"]
    assert payload["knots_added"] == 1
    assert payload["knots_removed"] == 1
    assert payload["n_knots_after"] == payload["n_knots_before"]
    assert payload["high_complexity_threshold"] == pytest.approx(1.0)
    assert payload["low_complexity_threshold"] == pytest.approx(0.05)
    json.dumps(payload)


def test_req_kan_2005_noop_and_input_guards_fail_closed() -> None:
    """REQ-KAN-2005: AMR validates topology inputs and can return a no-op result."""

    two_knot_layer = _layer_with_control_points([0.0, 1.0])
    noop = two_knot_layer.adaptive_mesh_refine(
        high_complexity_threshold=1.0,
        low_complexity_threshold=0.1,
        min_knots=2,
        max_knots=4,
    )
    assert noop.changed is False
    assert noop.complexity_scores == []

    threshold_noop_layer = _layer_with_control_points([0.0, 0.05, 0.95, 0.98, 1.0])
    threshold_noop = threshold_noop_layer.adaptive_mesh_refine(
        high_complexity_threshold=99.0,
        low_complexity_threshold=-1.0,
        min_knots=3,
        max_knots=8,
        max_additions=1,
        max_removals=1,
    )
    assert threshold_noop.changed is False

    with pytest.raises(ValueError, match="min_knots"):
        threshold_noop_layer.adaptive_mesh_refine(min_knots=1)
    with pytest.raises(ValueError, match="max_knots"):
        threshold_noop_layer.adaptive_mesh_refine(min_knots=5, max_knots=4)
    with pytest.raises(ValueError, match="non-negative"):
        threshold_noop_layer.adaptive_mesh_refine(max_additions=-1)

    mismatched = _layer_with_control_points([0.0, 0.5, 1.0])
    mismatched._knots = jnp.linspace(-1.0, 1.0, 4)
    with pytest.raises(ValueError, match="matching layer._knots"):
        mismatched.adaptive_mesh_refine()

    non_increasing = _layer_with_control_points([0.0, 0.25, 0.5, 0.75])
    non_increasing._knots = jnp.array([-1.0, -0.25, -0.25, 1.0], dtype=jnp.float32)
    with pytest.raises(ValueError, match="strictly increasing"):
        non_increasing.adaptive_mesh_refine()


def test_req_kan_2005_artifact_validation_rejects_schema_drift() -> None:
    """REQ-KAN-2005: artifact validation rejects incomplete or inconsistent evidence."""

    artifact = build_adaptive_energy_landscape_kan_artifact(run_date="20260513")

    missing = dict(artifact)
    del missing["energy_probe"]
    with pytest.raises(AssertionError, match="missing required fields"):
        validate_adaptive_energy_landscape_kan_artifact(missing)

    with pytest.raises(AssertionError, match="schema"):
        validate_adaptive_energy_landscape_kan_artifact(dict(artifact, schema="wrong"))
    with pytest.raises(AssertionError, match="status"):
        validate_adaptive_energy_landscape_kan_artifact(dict(artifact, status="draft"))
    with pytest.raises(AssertionError, match="experiment_id"):
        validate_adaptive_energy_landscape_kan_artifact(dict(artifact, experiment_id=999))
    with pytest.raises(AssertionError, match="spec_traces"):
        validate_adaptive_energy_landscape_kan_artifact(dict(artifact, spec_traces=[]))

    bad_metrics = dict(artifact["structural_change_metrics"], knots_removed=0)
    with pytest.raises(AssertionError, match="added and removed"):
        validate_adaptive_energy_landscape_kan_artifact(
            dict(artifact, structural_change_metrics=bad_metrics)
        )

    bad_counts = dict(artifact["structural_change_metrics"], n_knots_after=99)
    with pytest.raises(AssertionError, match="add and remove one knot"):
        validate_adaptive_energy_landscape_kan_artifact(
            dict(artifact, structural_change_metrics=bad_counts)
        )

    with pytest.raises(AssertionError, match="adaptive_mesh_refinement_ready"):
        validate_adaptive_energy_landscape_kan_artifact(
            dict(artifact, adaptive_mesh_refinement_ready=False)
        )

    bad_probe = dict(artifact["energy_probe"], finite_after_refinement=False)
    with pytest.raises(AssertionError, match="finite"):
        validate_adaptive_energy_landscape_kan_artifact(dict(artifact, energy_probe=bad_probe))


def test_scenario_kan_2005_artifact_writer_persists_required_json(tmp_path: Path) -> None:
    """SCENARIO-KAN-2005: Exp 2005 artifact records completed AMR evidence."""

    output_path = tmp_path / "experiment_2005_adaptive_energy_landscapes_kan.json"
    artifact = write_adaptive_energy_landscape_kan_artifact(
        output_path=output_path,
        run_date="20260513",
        tests_run=["tests/python/test_experiment_2005_adaptive_energy_landscapes_kan.py"],
    )
    persisted = json.loads(output_path.read_text(encoding="utf-8"))

    assert persisted == artifact
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["schema"] == "carnot.adaptive_energy_landscapes_kan.v1"
    assert artifact["status"] == "complete"
    assert artifact["experiment_id"] == 2005
    assert artifact["spec_traces"] == ["REQ-KAN-2005", "SCENARIO-KAN-2005"]
    assert artifact["run_date"] == "20260513"
    assert artifact["adaptive_mesh_refinement_ready"] is True
    assert artifact["structural_change_metrics"]["knots_added"] >= 1
    assert artifact["structural_change_metrics"]["knots_removed"] >= 1
    assert artifact["energy_probe"]["finite_after_refinement"] is True
    assert artifact["honest_verdict"].startswith("complete:")
