"""Tests for Exp 5098 bounded KAEM/PWA/MILP property-suite scaling.

Spec refs: REQ-KAN-5098, SCENARIO-KAN-5098.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from carnot import experiment_5098_kan_pwa_milp_scale_v2 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/kan/spec.md"
ARTIFACT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_kan_5098_spec_declares_property_suite_contract() -> None:
    """REQ-KAN-5098: OpenSpec anchors the suite before implementation."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-KAN-5098" in spec
    assert "SCENARIO-KAN-5098" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert mod.INFERENCE_SUBSTRATE in spec
    assert mod.SUCCESS_VERDICT in spec
    assert mod.BLOCKED_VERDICT in spec


def test_scenario_kan_5098_builds_required_bounded_property_suite() -> None:
    """SCENARIO-KAN-5098: suite has true, false, baseline, and budget-sensitive cases."""

    specs = mod.build_property_specs()
    roles = {spec.property_id: spec.suite_role for spec in specs}

    assert len(specs) >= 4
    assert roles["exp5091_baseline_two_unit_true"] == "baseline_expected_true"
    assert any(spec.expected_outcome == "verified" for spec in specs)
    assert any(spec.expected_outcome == "counterexample" for spec in specs)
    assert any(spec.suite_role == "approximation_error_sensitive" for spec in specs)
    assert any(spec.is_false_property_control for spec in specs)
    assert all(spec.threshold > 0.0 for spec in specs)
    assert mod.build_abstraction_for_property(specs[0]).binary_variable_count == 6


def test_req_kan_5098_solver_proves_true_cases_and_rejects_false_control() -> None:
    """REQ-KAN-5098: exact CPU solver reports status, counts, bounds, and controls."""

    if mod.detect_solver() != "z3":
        pytest.skip("Z3 is not available in this environment")

    results = mod.solve_property_suite(mod.build_property_specs())
    by_id = {result.property_id: result for result in results}

    baseline = by_id["exp5091_baseline_two_unit_true"]
    assert baseline.property_status == "verified"
    assert baseline.property_holds is True
    assert baseline.certified_upper_bound == pytest.approx(mod.EXP5091_PROPERTY_THRESHOLD)
    assert baseline.binary_variable_count == 6
    assert baseline.constraint_count == 43
    assert baseline.counterexample is None

    scaled = by_id["three_unit_composition_true"]
    assert scaled.property_status == "verified"
    assert scaled.property_holds is True
    assert scaled.binary_variable_count == 9
    assert scaled.constraint_count == 64
    assert scaled.certified_upper_bound == pytest.approx(1.8)

    false_control = by_id["adversarial_false_tight_bound"]
    assert false_control.property_status == "counterexample"
    assert false_control.property_holds is False
    assert false_control.counterexample is not None
    assert false_control.counterexample["inputs"] == pytest.approx([1.0, 1.0])
    assert false_control.budgeted_upper_bound > false_control.threshold

    sensitive = by_id["approximation_budget_sensitive_margin"]
    assert sensitive.property_status == "unproved_approximation_budget"
    assert sensitive.property_holds is None
    assert sensitive.certified_upper_bound <= sensitive.threshold
    assert sensitive.budgeted_upper_bound > sensitive.threshold
    assert sensitive.counterexample is None


def test_req_kan_5098_artifact_fields_principles_and_controls(tmp_path: Path) -> None:
    """REQ-KAN-5098: artifact emits required schema fields and principle notes."""

    if mod.detect_solver() != "z3":
        pytest.skip("Z3 is not available in this environment")

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.write_outputs(artifact_path=artifact_path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(payload)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(payload["field_principles"])
    assert artifact["honest_verdict"] == mod.SUCCESS_VERDICT
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["false_property_controls_passed"] is True
    assert artifact["properties_proved"] == [
        "exp5091_baseline_two_unit_true",
        "three_unit_composition_true",
    ]
    assert "adversarial_false_tight_bound" in artifact["counterexamples"]
    assert artifact["solver_statuses"]["adversarial_false_tight_bound"] == "optimal"
    assert artifact["binary_variable_counts"]["three_unit_composition_true"] == 9
    assert artifact["constraint_counts"]["three_unit_composition_true"] == 64
    assert artifact["approximation_error_budget"]["approximation_budget_sensitive_margin"] == pytest.approx(0.02)
    assert artifact["max_scale_reached"]["binary_variable_count"] == 9
    assert artifact["scale_blocker"] is None
    assert artifact["flagged_adversarial"] is False
    assert "live_llm" not in artifact["inference_substrate"]
    assert "hardware" not in artifact["methodology_note"].lower()
    assert len(artifact["reproducibility_checksum"]) == 64
    mod.validate_artifact(artifact)


def test_req_kan_5098_blocked_artifact_when_solver_forced_absent() -> None:
    """REQ-KAN-5098: absent solver dependencies fail closed without proving controls."""

    artifact = mod.build_artifact(solver_name="")

    assert artifact["honest_verdict"] == mod.BLOCKED_VERDICT
    assert artifact["scale_blocker"] == "blocked_kan_pwa_milp_solver_unavailable"
    assert artifact["properties_proved"] == []
    assert artifact["false_property_controls_passed"] is False
    assert set(artifact["solver_statuses"].values()) == {"blocked_solver_dependency"}
    assert artifact["counterexamples"] == {}
    assert artifact["flagged_adversarial"] is False
    mod.validate_artifact(artifact)


def test_req_kan_5098_defensive_scale_blocker_helpers() -> None:
    """REQ-KAN-5098: suite blocker helpers classify non-clean telemetry."""

    if mod.detect_solver() != "z3":
        pytest.skip("Z3 is not available in this environment")

    baseline, _scaled, _false_control, sensitive = mod.solve_property_suite(
        mod.build_property_specs()
    )

    assert mod._z3_float("3?") == pytest.approx(3.0)
    assert mod._max_scale_reached([]) == {}
    assert (
        mod._scale_blocker(
            [replace(baseline, solver_status="blocked_solver_dependency")],
            false_controls_passed=True,
        )
        == "blocked_kan_pwa_milp_solver_unavailable"
    )
    assert (
        mod._scale_blocker(
            [replace(baseline, solver_status="unknown")],
            false_controls_passed=True,
        )
        == "blocked_solver_status_unknown"
    )
    assert (
        mod._scale_blocker([baseline], false_controls_passed=False)
        == "false_property_control_not_counterexampled"
    )
    assert (
        mod._scale_blocker(
            [replace(baseline, property_status="counterexample")],
            false_controls_passed=True,
        )
        == "expected_true_not_proved:exp5091_baseline_two_unit_true"
    )
    assert (
        mod._scale_blocker(
            [baseline, replace(sensitive, property_status="verified")],
            false_controls_passed=True,
        )
        == "approximation_budget_control_failed:approximation_budget_sensitive_margin"
    )


def test_req_kan_5098_main_writes_default_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-KAN-5098: CLI entrypoint writes the configured result path."""

    if mod.detect_solver() != "z3":
        pytest.skip("Z3 is not available in this environment")

    monkeypatch.setenv("CARNOT_EXP5098_ROOT", str(tmp_path))

    assert mod.main() == 0
    payload = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert payload["honest_verdict"] == mod.SUCCESS_VERDICT
    mod.validate_artifact(payload)


def test_deliverable_file_validates_for_req_kan_5098() -> None:
    """SCENARIO-KAN-5098: committed deliverable JSON satisfies the schema."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.SUCCESS_VERDICT
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["false_property_controls_passed"] is True
