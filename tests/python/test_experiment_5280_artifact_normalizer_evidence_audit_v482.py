"""Tests for Exp 5280 artifact-normalizer evidence audit.

Spec refs: REQ-REPORT-5280, SCENARIO-REPORT-5280-EVIDENCE-AUDIT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5280_artifact_normalizer_evidence_audit_v482 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def _case(matrix: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(row for row in matrix if row["case"] == name)


def test_req_report_5280_spec_declares_evidence_audit_contract() -> None:
    """REQ-REPORT-5280: OpenSpec anchors the audit artifact and fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5280") : spec.index("### REQ-REPORT-5268")]

    for marker in (
        "REQ-REPORT-5280",
        "SCENARIO-REPORT-5280-EVIDENCE-AUDIT",
        str(mod.RESULT_RELATIVE_PATH),
        "aggregation_from_upstream_artifacts",
        "scripts/experiment_template.py:normalize_artifact_for_template_write",
        "scripts/research_conductor.py",
        "adversarial_verify_weakening.value` SHALL be false",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_report_5280_matrix_covers_required_artifact_shapes() -> None:
    """SCENARIO-REPORT-5280-EVIDENCE-AUDIT: matrix covers all requested cases."""

    matrix = mod.build_audit_matrix()

    assert {row["case"] for row in matrix} == set(mod.AUDIT_CASES)
    assert all(row["passed"] is True for row in matrix)

    valid = _case(matrix, "valid_shape_only_artifact")
    assert valid["ready_for_gated_consumers"] is True
    assert "top_level_wrapper_unwrapped" in valid["safe_repair_kinds"]

    wrapped = _case(matrix, "dict_wrapped_substrate_fields")
    assert wrapped["normalized"]["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert "DURATION_TOO_SHORT" not in wrapped["adversarial_flags"]

    aggregation = _case(matrix, "no_llm_aggregation_artifact")
    assert aggregation["duration_floor"]["reason"] == "aggregation"
    assert "DURATION_TOO_SHORT" not in aggregation["adversarial_flags"]
    assert "METHODOLOGY_MISSING" not in aggregation["adversarial_flags"]


def test_scenario_report_5280_bare_gates_and_missing_evidence_are_guarded() -> None:
    """SCENARIO-REPORT-5280-EVIDENCE-AUDIT: gates stay bare and evidence rejects."""

    matrix = mod.build_audit_matrix()

    gate_case = _case(matrix, "bare_gate_fields")
    assert gate_case["normalized"]["producer_normalizer_ready"] is True
    assert isinstance(gate_case["normalized"]["producer_normalizer_ready"], bool)
    assert gate_case["safe_repair_kinds"] == []
    assert gate_case["unsafe_rejection_kinds"] == []

    missing = _case(matrix, "missing_evidence")
    assert missing["ready_for_gated_consumers"] is False
    assert "missing_methodology_receipt" in missing["unsafe_rejection_kinds"]
    assert "model_specs" not in missing["normalized"]
    assert "random_seed" not in missing["normalized"]

    duration = _case(matrix, "sub_threshold_duration")
    assert duration["ready_for_gated_consumers"] is False
    assert "duration_too_short" in duration["unsafe_rejection_kinds"]
    assert "DURATION_TOO_SHORT" in duration["adversarial_flags"]


def test_req_report_5280_summary_reports_ready_without_verifier_weakening() -> None:
    """REQ-REPORT-5280: summary keeps old-pilot quarantine separate from readiness."""

    summary = mod.audit_summary(
        matrix=mod.build_audit_matrix(),
        producer_inventory=mod.enumerate_producers(),
    )

    assert summary["normalizer_evidence_ready"] is True
    assert summary["bare_gate_preservation_passed"] is True
    assert summary["missing_evidence_rejected"] is True
    assert summary["duration_substrate_regression_passed"] is True
    assert summary["adversarial_verify_weakening"] is False
    assert summary["producer_coverage"] == pytest.approx(1.0)
    assert summary["v481_quarantine_preserved"] is True


def test_req_report_5280_quarantine_check_reports_missing_inputs(tmp_path: Path) -> None:
    """REQ-REPORT-5280: absent upstream artifacts stay visible as missing evidence."""

    checks = mod.v481_quarantine_checks(tmp_path)

    assert {row["path"] for row in checks} == {
        str(path) for path in mod.V481_QUARANTINE_RELATIVE_PATHS
    }
    assert all(row["flags"] == ["artifact_missing"] for row in checks)
    assert all(row["passed"] is False for row in checks)


def test_req_report_5280_builds_and_validates_terminal_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-5280: terminal artifact exposes required wrapped fields."""

    artifact = mod.build_artifact(
        tests_run=[{"command": "unit fixture", "outcome": "PASS"}],
        duration_s=0.25,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert "producer evidence discipline is ready" in artifact["honest_verdict"]["value"]
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["normalizer_evidence_ready"]["value"] is True
    assert artifact["adversarial_verify_weakening"]["value"] is False
    assert artifact["tests_run"] == [{"command": "unit fixture", "outcome": "PASS"}]
    assert artifact["research_conductor_modified"]["value"] is False

    result_path = tmp_path / "experiment_5280.json"
    written = mod.write_artifact(
        output_path=result_path,
        tests_run=[{"command": "write fixture", "outcome": "PASS"}],
        duration_s=0.5,
    )
    assert json.loads(result_path.read_text(encoding="utf-8")) == written
    mod.validate_artifact(written)


def test_req_report_5280_validate_artifact_rejects_bad_required_shape() -> None:
    """REQ-REPORT-5280: schema validation rejects non-terminal audit artifacts."""

    artifact = mod.build_artifact(
        tests_run=[{"command": "unit fixture", "outcome": "PASS"}],
        duration_s=0.25,
    )
    bad = dict(artifact)
    bad["adversarial_verify_weakening"] = {
        "value": True,
        "principle": mod.FIELD_PRINCIPLES["adversarial_verify_weakening"],
    }

    with pytest.raises(ValueError, match="adversarial_verify_weakening"):
        mod.validate_artifact(bad)
