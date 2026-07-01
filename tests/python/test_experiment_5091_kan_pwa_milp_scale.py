"""Tests for Exp 5091 small KAEM/PWA/MILP scale telemetry.

Spec refs: REQ-KAN-5091, SCENARIO-KAN-5091.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5091_kan_pwa_milp_scale_v467 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/kan/spec.md"
ARTIFACT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_kan_5091_spec_declares_scale_contract() -> None:
    """REQ-KAN-5091: OpenSpec anchors the scale telemetry before code."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-KAN-5091" in spec
    assert "SCENARIO-KAN-5091" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert mod.INFERENCE_SUBSTRATE in spec
    assert mod.SUCCESS_VERDICT in spec
    assert mod.BLOCKED_VERDICT in spec


def test_scenario_kan_5091_builds_two_input_exact_abstraction() -> None:
    """SCENARIO-KAN-5091: two KAEM variables become explicit PWA units."""

    abstraction = mod.build_scaled_abstraction(mod.build_two_unit_kaem_layer())

    assert abstraction.input_dimension == 2
    assert abstraction.pwa_piece_count == 6
    assert abstraction.binary_variable_count == 6
    assert abstraction.local_error_bound == pytest.approx(0.0)
    assert abstraction.global_error_bound == pytest.approx(0.0)
    assert abstraction.units[0].n_segments == 3
    assert abstraction.units[1].n_segments == 3
    assert abstraction.evaluate((0.0, 0.0)) == pytest.approx(0.8)
    assert abstraction.evaluate((1.0, 1.0)) == pytest.approx(mod.PROPERTY_THRESHOLD)
    assert "two-variable" in abstraction.as_serializable()["scale_up_note"]


def test_req_kan_5091_solver_reports_complexity_and_tight_bound() -> None:
    """REQ-KAN-5091: deterministic solver reports counts, time, and tightness."""

    if mod.detect_solver() != "z3":
        pytest.skip("Z3 is not available in this environment")

    result = mod.solve_scale_property(
        mod.build_scaled_abstraction(mod.build_two_unit_kaem_layer())
    )

    assert result.solver_available is True
    assert result.property_status == "verified"
    assert result.property_holds is True
    assert result.binary_variable_count == 6
    assert result.constraint_count == 43
    assert result.pwa_piece_count == 6
    assert result.certified_upper_bound == pytest.approx(mod.PROPERTY_THRESHOLD)
    assert result.bound_tightness == pytest.approx(0.0)
    assert result.solve_time_s >= 0.0
    assert result.witness_inputs == pytest.approx((1.0, 1.0))
    assert result.certificate["method"] == "z3_mixed_integer_linear_pwa_scale_bound"


def test_req_kan_5091_blocked_artifact_when_solver_forced_absent() -> None:
    """REQ-KAN-5091: absent solver dependencies fail closed with a blocker."""

    artifact = mod.build_artifact(solver_name="")

    assert artifact["honest_verdict"] == mod.BLOCKED_VERDICT
    assert artifact["abstraction_built"] is True
    assert artifact["solver_available"] is False
    assert artifact["property_status"] == "blocked_solver_dependency"
    assert artifact["property_holds"] is None
    assert artifact["binary_variable_count"] == 6
    assert artifact["constraint_count"] == 43
    assert artifact["pwa_piece_count"] == 6
    assert artifact["scale_blocker"] == "blocked_kan_pwa_milp_solver_unavailable"
    assert artifact["flagged_adversarial"] is False
    mod.validate_artifact(artifact)


def test_req_kan_5091_artifact_fields_and_principles(tmp_path: Path) -> None:
    """REQ-KAN-5091: artifact emits required schema fields and principles."""

    if mod.detect_solver() != "z3":
        pytest.skip("Z3 is not available in this environment")

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.write_outputs(artifact_path=artifact_path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(payload)
    assert artifact["honest_verdict"] == mod.SUCCESS_VERDICT
    assert artifact["duration_s"] >= 0.0
    assert artifact["inference_substrate"] == "deterministic_formal_solver"
    assert artifact["abstraction_built"] is True
    assert artifact["solver_available"] is True
    assert artifact["property_status"] == "verified"
    assert artifact["property_holds"] is True
    assert artifact["binary_variable_count"] == 6
    assert artifact["constraint_count"] == 43
    assert artifact["pwa_piece_count"] == 6
    assert artifact["local_error_bound"] == pytest.approx(0.0)
    assert artifact["global_error_bound"] == pytest.approx(0.0)
    assert artifact["solve_time_s"] >= 0.0
    assert artifact["bound_tightness"] == pytest.approx(0.0)
    assert artifact["scale_blocker"] is None
    assert artifact["flagged_adversarial"] is False
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_principles"])
    assert "deterministic_formal_solver" in artifact["methodology_note"]
    assert "live_llm_inference" not in artifact["inference_substrate"]
    assert len(artifact["reproducibility_checksum"]) == 64
    assert "REQ-KAN-5091" in artifact["spec_refs"]
    mod.validate_artifact(artifact)


def test_deliverable_file_validates_for_req_kan_5091() -> None:
    """SCENARIO-KAN-5091: committed deliverable JSON satisfies the schema."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.SUCCESS_VERDICT
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["property_holds"] is True
