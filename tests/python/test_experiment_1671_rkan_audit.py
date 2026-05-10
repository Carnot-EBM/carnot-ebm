"""Tests for Exp 1671 Hybrid Zeckendorf RKAN audit.

Spec: REQ-KAN-1671, SCENARIO-KAN-1671.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from scripts import experiment_1671_rkan_audit as exp


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_req_kan_1671_spec_anchor_exists() -> None:
    """REQ-KAN-1671, SCENARIO-KAN-1671: the audit is spec-anchored."""

    spec = (REPO_ROOT / "openspec/capabilities/kan/spec.md").read_text(encoding="utf-8")

    assert "REQ-KAN-1671" in spec
    assert "SCENARIO-KAN-1671" in spec
    assert "results/experiment_1671_rkan.json" in spec


def test_req_kan_1671_hybrid_zeckendorf_reconstructs_and_rejects_float() -> None:
    """REQ-KAN-1671: Hybrid Zeckendorf rationals witness exact integers only."""

    witness = exp.zeckendorf_witness(-42)

    assert witness["sign"] == -1
    assert exp.reconstruct_zeckendorf(witness) == -42
    assert exp.zeckendorf_terms_are_nonconsecutive(witness)

    rational = exp.HybridZeckendorfRational.from_value("79/36")

    assert rational.fraction == Fraction(79, 36)
    assert exp.reconstruct_zeckendorf(rational.numerator_witness) == 79
    assert exp.reconstruct_zeckendorf(rational.denominator_witness) == 36
    assert rational.to_json()["fraction"] == "79/36"

    with pytest.raises(TypeError, match="bool"):
        exp.HybridZeckendorfRational.from_value(True)
    with pytest.raises(TypeError, match="exact rational"):
        exp.HybridZeckendorfRational.from_value(0.25)
    with pytest.raises(TypeError, match="integer"):
        exp.zeckendorf_witness("3")  # type: ignore[arg-type]


def test_req_kan_1671_cpu_simulation_uses_exact_rational_math() -> None:
    """REQ-KAN-1671: the mock KAN simulation uses exact rational arithmetic."""

    model = exp.build_mock_rkan()
    simulation = exp.simulate_cpu(model, exp.DEFAULT_SAMPLE_INPUTS)

    assert simulation.energies == (Fraction(41, 48), Fraction(-7, 12), Fraction(17, 12))
    assert simulation.float_operations_used is False
    assert simulation.operation_counts["edge_products"] == 3
    assert simulation.operation_counts["spline_evaluations"] == 9
    assert simulation.operation_counts["total_fraction_ops_upper_bound"] == 39
    assert all(isinstance(energy, Fraction) for energy in simulation.energies)

    serialized = simulation.to_json()
    assert [row["energy"]["fraction"] for row in serialized["samples"]] == [
        "41/48",
        "-7/12",
        "17/12",
    ]

    with pytest.raises(ValueError, match="at least two"):
        exp.MockRationalSpline.from_points("bad", ["0"])
    with pytest.raises(ValueError, match="domain"):
        exp.MockRationalSpline.from_points("bad", ["0", "1"], domain=("1", "-1"))
    with pytest.raises(ValueError, match="expected 2 inputs"):
        model.forward(["1"])


def test_req_kan_1671_bounding_certificates_contain_simulated_energies() -> None:
    """REQ-KAN-1671: complexity and bounds are certificate-backed."""

    model = exp.build_mock_rkan()
    simulation = exp.simulate_cpu(model, exp.DEFAULT_SAMPLE_INPUTS)
    certificates = exp.build_bounding_certificates(model, simulation)

    assert certificates["spline_output_bounds"]["edge_0_1"] == {"lower": "-1/4", "upper": "3/4"}
    assert certificates["model_output_bound"]["lower"] == "-7/12"
    assert certificates["model_output_bound"]["upper"] == "25/12"
    assert certificates["sample_energy_bounds"]["lower"] == "-7/12"
    assert certificates["sample_energy_bounds"]["upper"] == "17/12"
    assert certificates["sample_energy_bounds"]["contains_all_simulated_energies"] is True
    assert certificates["zeckendorf_witnesses_valid"] is True
    assert exp.validate_bounding_certificates(certificates, simulation) is True

    invalid = dict(certificates)
    invalid["sample_energy_bounds"] = dict(certificates["sample_energy_bounds"], upper="0")
    assert exp.validate_bounding_certificates(invalid, simulation) is False
    assert exp.validate_bounding_certificates({}, simulation) is False


def test_scenario_kan_1671_artifact_and_cli_write_required_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-KAN-1671: the runner writes the completed audit artifact."""

    output_path = tmp_path / "experiment_1671_rkan.json"
    artifact = exp.run_experiment(
        output_path=output_path,
        run_date="20260510",
        tests_run=["test_scenario_kan_1671_artifact_and_cli_write_required_json"],
    )
    persisted = json.loads(output_path.read_text(encoding="utf-8"))

    assert persisted == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["schema"] == "carnot.rkan.hybrid_zeckendorf_audit.v1"
    assert artifact["status"] == "complete"
    assert artifact["experiment_id"] == 1671
    assert artifact["spec_traces"] == ["REQ-KAN-1671", "SCENARIO-KAN-1671"]
    assert artifact["float_operations_used"] is False
    assert artifact["hardware_synthesis_claimed"] is False
    assert artifact["no_synthesis_accounting_only"] is True
    assert artifact["hybrid_zeckendorf_rational_math"] is True
    assert artifact["exact_rational_cpu_simulation"] is True
    assert (
        artifact["bounding_certificates"]["sample_energy_bounds"]["contains_all_simulated_energies"]
        is True
    )
    assert artifact["honest_verdict"].startswith("complete:")

    with pytest.raises(AssertionError, match="missing required fields"):
        exp.validate_artifact({"status": "complete"})
    with pytest.raises(AssertionError, match="hardware_synthesis_claimed"):
        exp.validate_artifact(dict(artifact, hardware_synthesis_claimed=True))
    with pytest.raises(AssertionError, match="float_operations_used"):
        exp.validate_artifact(dict(artifact, float_operations_used=True))
    with pytest.raises(AssertionError, match="spec_traces"):
        exp.validate_artifact(dict(artifact, spec_traces=[]))

    cli_path = tmp_path / "cli_experiment_1671_rkan.json"
    rc = exp.main(["--output", str(cli_path), "--run-date", "20260510"])
    assert rc == 0
    assert json.loads(cli_path.read_text(encoding="utf-8"))["status"] == "complete"
    assert "wrote=" in capsys.readouterr().out
