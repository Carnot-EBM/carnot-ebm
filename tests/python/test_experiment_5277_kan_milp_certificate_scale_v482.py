"""Tests for Exp 5277 KAN PWA/MILP certificate scale.

Spec refs: REQ-KAN-5277, SCENARIO-KAN-5277.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_5277_kan_milp_certificate_scale_v482 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/kan/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _value(artifact: dict[str, object], field: str) -> object:
    wrapped = artifact[field]
    assert isinstance(wrapped, dict)
    return wrapped["value"]


def test_req_kan_5277_spec_declares_multi_component_certificate_contract() -> None:
    """REQ-KAN-5277: OpenSpec anchors the V482 scaled certificate contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-KAN-5277") : spec.index("## Implementation Status")]

    for marker in (
        "REQ-KAN-5277",
        "SCENARIO-KAN-5277",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "multi-component PWA/MILP",
        "false-property",
        "dynamic spot-check",
        "broad KAN verification",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_kan_5277_builds_three_component_pwa_envelope() -> None:
    """SCENARIO-KAN-5277: three convex components become six PWA pieces."""

    source = mod.load_v481_source()
    abstraction = mod.build_multi_component_abstraction()

    assert source.ready is True
    assert abstraction.component_count == 3
    assert abstraction.piece_count == 6
    assert abstraction.input_box == mod.INPUT_BOX
    assert abstraction.local_error_bounds == pytest.approx((0.00625, 0.005, 0.00375))
    assert abstraction.local_error_bound_max == pytest.approx(0.00625)
    assert abstraction.global_error_bound == pytest.approx(0.015)
    assert abstraction.evaluate_actual((0.6, 0.6, 0.6)) == pytest.approx(0.4984)
    assert abstraction.evaluate_upper_envelope((0.6, 0.6, 0.6)) == pytest.approx(0.4984)


def test_req_kan_5277_accepts_true_property_and_rejects_false_property() -> None:
    """REQ-KAN-5277: true bound certifies while nearby false bound is rejected."""

    if mod.detect_solver() != "z3":
        pytest.skip("Z3 is not available in this environment")

    result = mod.solve_scaled_certificate(mod.build_multi_component_abstraction(), solver_name="z3")
    by_id = {row.property_id: row for row in result.property_results}

    true_property = by_id["v482_true_multi_component_upper_property"]
    false_property = by_id["v482_nearby_false_multi_component_upper_property"]

    assert result.solver_status == "optimal"
    assert result.certified_upper_bound == pytest.approx(0.4984)
    assert result.piece_count == 6
    assert result.solve_time_s >= 0.0
    assert result.witness_inputs == pytest.approx((0.6, 0.6, 0.6))
    assert true_property.certified is True
    assert true_property.rejected is False
    assert true_property.threshold == pytest.approx(mod.TRUE_PROPERTY_THRESHOLD)
    assert true_property.certificate_slack == pytest.approx(0.0166)
    assert false_property.certified is False
    assert false_property.rejected is True
    assert false_property.threshold == pytest.approx(mod.FALSE_PROPERTY_THRESHOLD)
    assert false_property.certificate_slack == pytest.approx(-0.0004)
    assert false_property.actual_witness_value == pytest.approx(0.4984)


def test_req_kan_5277_dynamic_spot_check_covers_certificate_region() -> None:
    """REQ-KAN-5277: sampled falsification pass checks the certified region."""

    abstraction = mod.build_multi_component_abstraction()
    result = mod.solve_scaled_certificate(abstraction, solver_name="z3")
    spot_check = mod.run_dynamic_spot_check(abstraction, result)

    assert spot_check.passed is True
    assert spot_check.sample_count == 125
    assert spot_check.max_actual_value <= mod.TRUE_PROPERTY_THRESHOLD
    assert spot_check.max_actual_value == pytest.approx(0.4984)
    assert spot_check.false_property_witness_seen is True
    assert spot_check.envelope_violation_count == 0


def test_req_kan_5277_artifact_fields_are_principle_wrapped(tmp_path: Path) -> None:
    """REQ-KAN-5277: deliverable fields expose principles and slack accounting."""

    if mod.detect_solver() != "z3":
        pytest.skip("Z3 is not available in this environment")

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.write_outputs(
        artifact_path=artifact_path,
        duration_s=0.4,
        tests_run=[{"command": "unit fixture", "outcome": "passed"}],
        solver_name="z3",
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload == artifact
    mod.validate_artifact(artifact)
    assert _value(artifact, "honest_verdict").startswith("complete:")
    assert "positive" in _value(artifact, "honest_verdict")
    assert _value(artifact, "inference_substrate") == mod.INFERENCE_SUBSTRATE
    assert _value(artifact, "certificate_scaled") is True
    assert _value(artifact, "false_property_rejected") is True
    assert _value(artifact, "approximation_slack") == pytest.approx(0.0166)
    assert _value(artifact, "piece_count") == 6
    assert _value(artifact, "solve_time_s") >= 0.0
    assert _value(artifact, "dynamic_spot_check_passed") is True
    assert artifact["tests_run"] == [{"command": "unit fixture", "outcome": "passed"}]
    assert artifact["slack_accounting"]["true_property_slack"] == pytest.approx(0.0166)
    assert artifact["slack_accounting"]["global_error_bound"] == pytest.approx(0.015)
    assert artifact["spot_check"]["sample_count"] == 125
    assert len(artifact["reproducibility_checksum"]) == 64


def test_req_kan_5277_validation_rejects_slack_and_false_property_drift() -> None:
    """REQ-KAN-5277: schema validation fails closed on unsound drift."""

    artifact = mod.build_artifact(
        duration_s=0.1,
        tests_run=[{"command": "unit fixture", "outcome": "passed"}],
        solver_name="z3",
    )

    broken = copy.deepcopy(artifact)
    broken["false_property_rejected"] = mod.wrap_field("false_property_rejected", False)
    with pytest.raises(AssertionError, match="false property"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["approximation_slack"] = mod.wrap_field("approximation_slack", -0.1)
    with pytest.raises(AssertionError, match="approximation_slack"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["dynamic_spot_check_passed"] = mod.wrap_field("dynamic_spot_check_passed", False)
    with pytest.raises(AssertionError, match="spot check"):
        mod.validate_artifact(broken)


def test_deliverable_file_validates_for_scenario_kan_5277() -> None:
    """SCENARIO-KAN-5277: committed deliverable satisfies the V482 contract."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert _value(artifact, "certificate_scaled") is True
    assert _value(artifact, "false_property_rejected") is True
    assert _value(artifact, "piece_count") == 6
    assert _value(artifact, "dynamic_spot_check_passed") is True
    assert "REQ-KAN-5277" in artifact["spec_refs"]
