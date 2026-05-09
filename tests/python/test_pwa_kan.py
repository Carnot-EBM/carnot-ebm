"""Tests for model-level PWA KAN spline activation bounds.

Spec traces: REQ-KAN-1618, SCENARIO-KAN-1618
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from carnot.models.kan import BSpline, BSplineParams
from carnot.models.pwa_kan import (
    PWAKANUnit,
    PWASplineSegment,
    build_experiment_1618_artifact,
    build_pwa_for_bspline,
    write_experiment_1618_artifact,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_req_kan_1618_spec_anchor_exists() -> None:
    """REQ-KAN-1618, SCENARIO-KAN-1618: PWA KAN wrapper is spec-anchored."""

    spec = (REPO_ROOT / "openspec/capabilities/kan/spec.md").read_text(encoding="utf-8")

    assert "REQ-KAN-1618" in spec
    assert "SCENARIO-KAN-1618" in spec
    assert "results/experiment_1618_pwa_kan.json" in spec


def test_req_kan_1618_wraps_existing_bspline_with_exact_linear_bounds() -> None:
    """REQ-KAN-1618: BSpline units get exact knot-aligned affine boundaries."""

    spline = BSpline(num_knots=3, degree=3)
    params = BSplineParams(control_points=jnp.array([0.0, 1.0, 2.0, 9.0, 9.0, 9.0]))

    unit = build_pwa_for_bspline(
        spline,
        name="exact_linear_bspline",
        params=params,
        samples_per_segment=17,
    )

    assert unit.name == "exact_linear_bspline"
    assert len(unit.segments) == 2
    assert unit.max_abs_error < 1e-12

    for x in np.linspace(-1.0, 1.0, 17):
        original = float(spline.evaluate(jnp.asarray(x), params))
        assert unit.evaluate(float(x)) == pytest.approx(original, abs=1e-12)

    bounds = unit.activation_bounds(-0.25, 0.75)
    assert bounds.lower == pytest.approx(0.75, abs=1e-12)
    assert bounds.upper == pytest.approx(1.75, abs=1e-12)
    assert bounds.lower_witness_x == pytest.approx(-0.25, abs=1e-12)
    assert bounds.upper_witness_x == pytest.approx(0.75, abs=1e-12)

    weighted = unit.weighted_activation_bounds(-0.25, 0.75, weight=-2.0)
    assert weighted.lower == pytest.approx(-3.5, abs=1e-12)
    assert weighted.upper == pytest.approx(-1.5, abs=1e-12)

    positive_weighted = unit.weighted_activation_bounds(-0.25, 0.75, weight=2.0)
    assert positive_weighted.lower == pytest.approx(1.5, abs=1e-12)
    assert positive_weighted.upper == pytest.approx(3.5, abs=1e-12)

    constraints = unit.logical_constraints(input_name="x", output_name="y")
    assert len(constraints) == 2
    assert constraints[0]["condition"] == "-1.0 <= x <= 0.0"
    assert constraints[0]["lower"] == "1.0 * x + 1.0 <= y"
    assert constraints[0]["upper"] == "y <= 1.0 * x + 1.0"

    random_bspline_unit = build_pwa_for_bspline(BSpline(num_knots=2, degree=3), name="random")
    assert len(random_bspline_unit.segments) == 1


def test_req_kan_1618_arbitrary_nonlinear_callable_gets_sampled_envelopes() -> None:
    """REQ-KAN-1618: arbitrary 1D splines get conservative affine envelopes."""

    unit = PWAKANUnit.from_callable(
        name="quadratic_callable",
        evaluator=lambda x: x * x,
        breakpoints=(-1.0, 0.0, 1.0),
        samples_per_segment=65,
    )

    assert len(unit.segments) == 2
    assert unit.max_abs_error == pytest.approx(0.25, abs=1e-12)

    for segment in unit.segments:
        assert segment.residual_min == pytest.approx(-0.25, abs=1e-12)
        assert segment.residual_max == pytest.approx(0.0, abs=1e-12)
        for x in np.linspace(segment.lower_x, segment.upper_x, 65):
            y = float(x * x)
            assert segment.lower_affine(float(x)) <= y + 1e-12
            assert y <= segment.upper_affine(float(x)) + 1e-12

    bounds = unit.activation_bounds(-0.75, 0.75)
    sampled_values = [float(x * x) for x in np.linspace(-0.75, 0.75, 151)]
    assert bounds.lower <= min(sampled_values) + 1e-12
    assert bounds.upper >= max(sampled_values) - 1e-12
    assert bounds.segment_count == 2


def test_req_kan_1618_validation_rejects_ambiguous_domains() -> None:
    """REQ-KAN-1618: malformed PWA domains and interval queries fail loudly."""

    with pytest.raises(ValueError, match="at least two breakpoints"):
        PWAKANUnit.from_callable("bad", lambda x: x, breakpoints=(0.0,))

    with pytest.raises(ValueError, match="strictly increasing"):
        PWAKANUnit.from_callable("bad", lambda x: x, breakpoints=(0.0, 0.0, 1.0))

    with pytest.raises(ValueError, match="samples_per_segment"):
        PWAKANUnit.from_callable("bad", lambda x: x, breakpoints=(0.0, 1.0), samples_per_segment=1)

    with pytest.raises(ValueError, match="residual_padding"):
        PWAKANUnit.from_callable(
            "bad",
            lambda x: x,
            breakpoints=(0.0, 1.0),
            residual_padding=-1.0,
        )

    with pytest.raises(ValueError, match="non-finite"):
        PWAKANUnit.from_callable("bad", lambda x: float("nan"), breakpoints=(0.0, 1.0))

    with pytest.raises(ValueError, match="at least one segment"):
        PWAKANUnit("empty", ())

    segment = PWASplineSegment(
        lower_x=0.0,
        upper_x=1.0,
        slope=1.0,
        intercept=0.0,
        residual_min=0.0,
        residual_max=0.0,
        max_abs_error=0.0,
    )
    with pytest.raises(ValueError, match="does not overlap"):
        segment.bound_on_interval(2.0, 3.0)

    unit = PWAKANUnit.from_callable("linear", lambda x: x, breakpoints=(-1.0, 1.0))

    with pytest.raises(ValueError, match="lower_x"):
        unit.activation_bounds(0.5, -0.5)

    with pytest.raises(ValueError, match="outside"):
        unit.evaluate(2.0)

    with pytest.raises(ValueError, match="does not overlap"):
        unit.activation_bounds(2.0, 3.0)

    assert unit.count_sample_envelope_violations(lambda x: 10.0, samples_per_segment=3) == 3


def test_scenario_kan_1618_builds_and_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-KAN-1618: completed PWA KAN artifact has required schema fields."""

    artifact = build_experiment_1618_artifact()

    required_fields = {
        "schema",
        "status",
        "experiment",
        "experiment_id",
        "run_date",
        "spec",
        "module",
        "artifact_path",
        "pwa_kan_ready",
        "arbitrary_1d_spline_supported",
        "exact_linear_max_abs_error",
        "nonlinear_sample_violations",
        "logical_constraint_count",
        "sample_interval_bound",
        "honest_verdict",
    }
    assert required_fields <= set(artifact)
    assert artifact["schema"] == "carnot.pwa_kan_logical_bounds.v1"
    assert artifact["status"] == "complete"
    assert artifact["experiment_id"] == 1618
    assert artifact["spec"] == ["REQ-KAN-1618", "SCENARIO-KAN-1618"]
    assert artifact["pwa_kan_ready"] is True
    assert artifact["arbitrary_1d_spline_supported"] is True
    assert artifact["exact_linear_max_abs_error"] < 1e-12
    assert artifact["nonlinear_sample_violations"] == 0
    assert artifact["logical_constraint_count"] >= 4
    assert artifact["honest_verdict"].startswith("complete:")

    output_path = tmp_path / "experiment_1618_pwa_kan.json"
    written = write_experiment_1618_artifact(output_path)

    assert written == artifact
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
