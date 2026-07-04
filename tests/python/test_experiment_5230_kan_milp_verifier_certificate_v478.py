"""Tests for Exp 5230 KAEM PWA/MILP verifier certificate.

Spec refs: REQ-KAN-5230, SCENARIO-KAN-5230.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_5230_kan_milp_verifier_certificate_v478 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/kan/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _value(artifact: dict[str, object], field: str) -> object:
    wrapped = artifact[field]
    assert isinstance(wrapped, dict)
    return wrapped["value"]


def test_req_kan_5230_spec_declares_certificate_contract() -> None:
    """REQ-KAN-5230: OpenSpec anchors the tiny certificate before implementation."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-KAN-5230") : spec.index("## Implementation Status")]

    for marker in (
        "REQ-KAN-5230",
        "SCENARIO-KAN-5230",
        str(mod.RESULT_RELATIVE_PATH),
        mod.TARGET_MODULE,
        mod.INFERENCE_SUBSTRATE,
        "monotonicity",
        "no-unsafe-decision",
    ):
        assert marker in section
    for field in mod.REQUIRED_WRAPPED_FIELDS:
        assert f"`{field}`" in section


def test_scenario_kan_5230_exports_tiny_kaem_pwa_fixture() -> None:
    """SCENARIO-KAN-5230: fixture uses the real KAEM layer and existing PWA helpers."""

    layer = mod.build_tiny_certificate_layer()
    abstraction = mod.build_certificate_abstraction(layer)

    assert abstraction.component_path == mod.TARGET_MODULE
    assert abstraction.input_dimension == 2
    assert abstraction.binary_variable_count == 6
    assert abstraction.pwa_piece_count == 6
    assert abstraction.global_error_bound == pytest.approx(0.0)
    assert abstraction.evaluate((0.5, 0.5)) == pytest.approx(0.625)
    assert mod.monotonicity_certificate(abstraction).verified is True
    assert mod.monotonicity_certificate(abstraction).min_slope == pytest.approx(0.225)


def test_req_kan_5230_solver_checks_monotonicity_and_unsafe_bound() -> None:
    """REQ-KAN-5230: solver certificate proves only the tiny bounded properties."""

    if mod.detect_solver() != "z3":
        pytest.skip("Z3 is not available in this environment")

    certificate = mod.run_certificate_checks(solver_name="z3")
    by_id = {row.property_id: row for row in certificate.property_results}

    monotonicity = by_id["bounded_monotonicity"]
    unsafe_bound = by_id["no_unsafe_decision"]

    assert certificate.produced is True
    assert monotonicity.verified is True
    assert monotonicity.method == "pwa_slope_inspection"
    assert unsafe_bound.verified is True
    assert unsafe_bound.method == "z3_mixed_integer_pwa_box_bound"
    assert unsafe_bound.certified_upper_bound == pytest.approx(0.625)
    assert unsafe_bound.threshold == pytest.approx(mod.UNSAFE_DECISION_THRESHOLD)
    assert unsafe_bound.bound_tightness == pytest.approx(0.075)
    assert unsafe_bound.witness_inputs == pytest.approx([0.5, 0.5])


def test_req_kan_5230_artifact_fields_are_principle_wrapped(tmp_path: Path) -> None:
    """REQ-KAN-5230: artifact emits the required principle-wrapped schema."""

    if mod.detect_solver() != "z3":
        pytest.skip("Z3 is not available in this environment")

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.write_outputs(
        artifact_path=artifact_path,
        run_date="20260704",
        duration_s=0.25,
        tests_run=[{"command": "unit fixture", "passed": True}],
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload == artifact
    mod.validate_artifact(artifact)
    assert _value(artifact, "kan_certificate_produced") is True
    assert _value(artifact, "certificate_path") == str(mod.RESULT_RELATIVE_PATH)
    assert _value(artifact, "target_module") == mod.TARGET_MODULE
    assert _value(artifact, "reused_existing_helpers") is True
    assert _value(artifact, "inference_substrate") == mod.INFERENCE_SUBSTRATE
    assert _value(artifact, "bound_tightness") == pytest.approx(0.075)
    assert _value(artifact, "honest_verdict").startswith("success:")
    assert "broad KAN verification" in artifact["methodology_note"]
    assert len(artifact["reproducibility_checksum"]) == 64


def test_req_kan_5230_blocks_without_solver_and_validates_edges() -> None:
    """REQ-KAN-5230: missing solver produces a precise blocked prerequisite."""

    blocked = mod.build_artifact(
        solver_name="",
        run_date="20260704",
        duration_s=0.1,
        tests_run=[{"command": "solver forced absent", "passed": True}],
    )

    mod.validate_artifact(blocked)
    assert _value(blocked, "kan_certificate_produced") is False
    assert _value(blocked, "certificate_path") is None
    assert _value(blocked, "bound_tightness") is None
    assert _value(blocked, "honest_verdict").startswith("complete:")
    assert blocked["blocked_reason"] == "blocked_kan_pwa_milp_solver_unavailable"
    assert blocked["solver_status"] == "blocked_solver_dependency"

    broken = copy.deepcopy(blocked)
    broken["inference_substrate"] = mod.wrap_field(
        "inference_substrate",
        "deterministic_enumeration",
    )
    with pytest.raises(AssertionError, match="inference_substrate"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(blocked)
    broken["honest_verdict"] = mod.wrap_field("honest_verdict", "ready_without_prefix")
    with pytest.raises(AssertionError, match="honest_verdict"):
        mod.validate_artifact(broken)


def test_deliverable_file_validates_for_req_kan_5230() -> None:
    """SCENARIO-KAN-5230: committed deliverable JSON is the tiny certificate."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert _value(artifact, "kan_certificate_produced") is True
    assert _value(artifact, "inference_substrate") == mod.INFERENCE_SUBSTRATE
    assert _value(artifact, "target_module") == mod.TARGET_MODULE
