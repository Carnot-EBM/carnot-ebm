"""Tests for the exact-rational KAN forward pass.

Spec traces: REQ-KAN-1602, SCENARIO-KAN-1602
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from carnot.models.rkan import (
    RationalKANEnergyFunction,
    RationalLinearSpline,
    build_experiment_1602_artifact,
    serialize_fraction,
    to_fraction,
    write_experiment_1602_artifact,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_req_kan_1602_spec_anchor_exists() -> None:
    """REQ-KAN-1602, SCENARIO-KAN-1602: RKAN work is spec-anchored."""

    spec = (REPO_ROOT / "openspec/capabilities/kan/spec.md").read_text(encoding="utf-8")

    assert "REQ-KAN-1602" in spec
    assert "SCENARIO-KAN-1602" in spec
    assert "results/experiment_1602_rkan.json" in spec


def test_req_kan_1602_fraction_conversion_rejects_implicit_float() -> None:
    """REQ-KAN-1602: exact mode accepts rational encodings but rejects floats."""

    assert to_fraction(Fraction(2, 5)) == Fraction(2, 5)
    assert to_fraction(3) == Fraction(3, 1)
    assert to_fraction("-7/9") == Fraction(-7, 9)

    with pytest.raises(TypeError, match="bool"):
        to_fraction(True)
    with pytest.raises(TypeError, match="exact rational"):
        to_fraction(0.5)


def test_req_kan_1602_spline_interpolates_as_fraction() -> None:
    """REQ-KAN-1602: spline interpolation stays in the rational field."""

    spline = RationalLinearSpline([0, "1", 0])

    assert spline.evaluate(Fraction(-3, 1)) == Fraction(0, 1)
    assert spline.evaluate(Fraction(-1, 1)) == Fraction(0, 1)
    assert spline.evaluate(Fraction(0, 1)) == Fraction(1, 1)
    assert spline.evaluate(Fraction(1, 2)) == Fraction(1, 2)
    assert spline.evaluate(Fraction(3, 1)) == Fraction(0, 1)
    assert isinstance(spline.evaluate(Fraction(1, 3)), Fraction)
    assert spline.as_serializable() == {
        "domain": ["-1", "1"],
        "control_points": ["0", "1", "0"],
    }


def test_req_kan_1602_spline_validation() -> None:
    """REQ-KAN-1602: exact splines validate their rational domain and knots."""

    with pytest.raises(ValueError, match="at least two"):
        RationalLinearSpline([1])
    with pytest.raises(ValueError, match="domain"):
        RationalLinearSpline([0, 1], domain=(1, -1))


def test_req_kan_1602_forward_is_exact_and_repeatable() -> None:
    """REQ-KAN-1602: RKAN energy is a deterministic Fraction."""

    model = RationalKANEnergyFunction(
        input_dim=3,
        edge_control_points={
            (0, 1): [0, 1, 2],
            (1, 2): ["1/3", "2/3", "1"],
        },
        bias_control_points=[
            [0, 0, 0],
            ["1/2", "1/2", "1/2"],
            [-1, 0, 1],
        ],
    )
    x = [Fraction(1, 2), Fraction(-1, 2), Fraction(1, 3)]

    first = model.forward(x)
    second = model.forward(tuple(Fraction(v.numerator, v.denominator) for v in x))

    assert first == Fraction(79, 36)
    assert second == first
    assert model.energy(x) == first
    assert model(x) == first
    assert model.energy_batch([x, x]) == (first, first)
    assert model.n_params == 15
    assert isinstance(first, Fraction)


def test_req_kan_1602_model_validation_and_serialization() -> None:
    """REQ-KAN-1602: RKAN rejects malformed shapes and serializes exact params."""

    with pytest.raises(ValueError, match="input_dim"):
        RationalKANEnergyFunction(input_dim=0)
    with pytest.raises(ValueError, match="edge index"):
        RationalKANEnergyFunction(input_dim=2, edge_control_points={(0, 2): [0, 1]})
    with pytest.raises(ValueError, match="bias_control_points"):
        RationalKANEnergyFunction(input_dim=2, bias_control_points=[[0, 1]])

    model = RationalKANEnergyFunction(
        input_dim=2,
        edge_control_points={(0, 1): [0, "1/2"]},
    )

    with pytest.raises(ValueError, match="expected 2 inputs"):
        model.forward([1])

    assert model.as_serializable() == {
        "input_dim": 2,
        "edges": {
            "0,1": {
                "domain": ["-1", "1"],
                "control_points": ["0", "1/2"],
            }
        },
        "biases": [
            {"domain": ["-1", "1"], "control_points": ["0", "0"]},
            {"domain": ["-1", "1"], "control_points": ["0", "0"]},
        ],
    }


def test_req_kan_1602_fraction_serialization() -> None:
    """REQ-KAN-1602: artifact fields use stable numerator/denominator strings."""

    assert serialize_fraction(Fraction(3, 1)) == "3"
    assert serialize_fraction(Fraction(-2, 7)) == "-2/7"


def test_scenario_kan_1602_builds_and_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-KAN-1602: the completed RKAN artifact has required fields."""

    artifact = build_experiment_1602_artifact()

    required_fields = {
        "schema",
        "status",
        "experiment",
        "experiment_id",
        "run_date",
        "spec",
        "module",
        "artifact_path",
        "exact_rational_forward_pass_ready",
        "float_operations_used",
        "repeated_forward_outputs_identical",
        "sample_outputs",
        "reference_energy",
        "honest_verdict",
    }
    assert required_fields <= set(artifact)
    assert artifact["schema"] == "carnot.rkan_exact_fraction.v1"
    assert artifact["status"] == "complete"
    assert artifact["experiment_id"] == 1602
    assert artifact["spec"] == ["REQ-KAN-1602", "SCENARIO-KAN-1602"]
    assert artifact["exact_rational_forward_pass_ready"] is True
    assert artifact["float_operations_used"] is False
    assert artifact["repeated_forward_outputs_identical"] is True
    assert artifact["sample_outputs"][0]["energy"] == "79/36"
    assert artifact["reference_energy"] == "79/36"
    assert artifact["honest_verdict"].startswith("complete:")

    output_path = tmp_path / "experiment_1602_rkan.json"
    written = write_experiment_1602_artifact(output_path)

    assert written == artifact
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
