"""Tests for Exp 5080 tiny KAEM/PWA/MILP bridge.

Spec refs: REQ-KAN-5080, SCENARIO-KAN-5080.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5080_kan_pwa_milp_bridge_v466 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/kan/spec.md"
ARTIFACT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_kan_5080_spec_declares_bridge_contract() -> None:
    """REQ-KAN-5080: OpenSpec anchors the bridge before implementation."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-KAN-5080" in spec
    assert "SCENARIO-KAN-5080" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert mod.INFERENCE_SUBSTRATE in spec
    assert mod.SUCCESS_VERDICT in spec
    assert mod.BLOCKED_VERDICT in spec


def test_scenario_kan_5080_builds_exact_kaem_pwa_abstraction() -> None:
    """SCENARIO-KAN-5080: KAEM knot/control points become PWA segments."""

    layer = mod.build_tiny_kaem_layer()
    abstraction = mod.build_pwa_abstraction(layer)

    assert abstraction.component_path == mod.KAN_COMPONENT_PATH
    assert abstraction.n_segments == 3
    assert abstraction.binary_variable_count == 3
    assert abstraction.error_bound == pytest.approx(0.0)
    assert abstraction.segments[0].x_min == pytest.approx(-1.0)
    assert abstraction.segments[-1].x_max == pytest.approx(1.0)
    assert all(segment.slope >= 0.0 for segment in abstraction.segments)
    assert abstraction.evaluate(0.0) == pytest.approx(0.5)
    assert abstraction.as_serializable()["exact_for_linear_kaem_spline"] is True


def test_req_kan_5080_solver_proves_bound_when_available() -> None:
    """REQ-KAN-5080: an available solver checks the tiny bound property."""

    if mod.detect_milp_solver() != "z3":
        pytest.skip("Z3 is not available in this environment")

    result = mod.solve_bound_property(mod.build_pwa_abstraction(mod.build_tiny_kaem_layer()))

    assert result.milp_solver_available is True
    assert result.property_checked is True
    assert result.property_holds is True
    assert result.solver_status == "optimal"
    assert result.certified_upper_bound == pytest.approx(1.0)
    assert result.witness_x == pytest.approx(1.0)
    assert result.certificate["method"] == "z3_mixed_integer_linear_pwa_bound"


def test_req_kan_5080_blocked_artifact_when_solver_forced_absent() -> None:
    """REQ-KAN-5080: absent solver dependencies produce blocked artifacts."""

    artifact = mod.build_artifact(solver_name="")

    assert artifact["honest_verdict"] == mod.BLOCKED_VERDICT
    assert artifact["milp_solver_available"] is False
    assert artifact["property_checked"] is False
    assert artifact["property_holds"] is None
    assert artifact["blocked_reason"] == "blocked_kan_pwa_milp_solver_unavailable"
    assert artifact["pwa_abstraction_built"] is True
    assert artifact["flagged_adversarial"] is False
    mod.validate_artifact(artifact)


def test_req_kan_5080_artifact_fields_and_principles(tmp_path: Path) -> None:
    """REQ-KAN-5080: artifact emits required fields and principle notes."""

    if mod.detect_milp_solver() != "z3":
        pytest.skip("Z3 is not available in this environment")

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.write_outputs(artifact_path=artifact_path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload == artifact
    assert set(mod.REQUIRED_USER_ARTIFACT_FIELDS).issubset(payload)
    assert artifact["honest_verdict"] == mod.SUCCESS_VERDICT
    assert artifact["duration_s"] >= 0.0
    assert artifact["inference_substrate"] == "deterministic_formal_check"
    assert artifact["kan_component_path"] == mod.KAN_COMPONENT_PATH
    assert artifact["pwa_abstraction_built"] is True
    assert artifact["milp_solver_available"] is True
    assert artifact["property_checked"] is True
    assert artifact["property_holds"] is True
    assert artifact["error_bound"] == pytest.approx(0.0)
    assert artifact["binary_variable_count"] == 3
    assert artifact["blocked_reason"] is None
    assert artifact["flagged_adversarial"] is False
    assert set(mod.REQUIRED_USER_ARTIFACT_FIELDS).issubset(artifact["field_principles"])
    assert "exact piecewise-affine" in artifact["methodology_note"]
    assert len(artifact["reproducibility_checksum"]) == 64
    assert "REQ-KAN-5080" in artifact["spec_refs"]
    mod.validate_artifact(artifact)


def test_deliverable_file_validates_for_req_kan_5080() -> None:
    """SCENARIO-KAN-5080: committed deliverable JSON satisfies the schema."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.SUCCESS_VERDICT
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["property_holds"] is True
